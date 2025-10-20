package moe

import (
	
	"fmt"
	"math/rand"
	"sort"

	. "nlptagger/neural/tensor"
)





// MoELayer implements a Mixture of Experts layer.
type MoELayer struct {
	GatingNetwork *GatingNetwork
	Experts       []Expert
	K             int // Number of top experts to select
	// InputDim      int // Add InputDim to MoELayer struct

	// Stored for backward pass
	inputTensor       *Tensor
	expertOutputs     []*Tensor
	expertActivations []*Tensor // Output of experts before combining
	selectedExperts   [][]int   // Indices of selected experts for each input in the batch
	gateOutputs       *Tensor   // Output of the gating network (probabilities)
	LoadBalancingLoss float64   // Load balancing loss
	Training          bool      // training mode
}

// NewMoELayer creates a new MoELayer.
// inputDim is the dimension of the input to the MoE layer.
// numExperts is the total number of experts.
// k is the number of top experts to select for each input.
// expertBuilder is a function that constructs an expert given its index.
func NewMoELayer(inputDim, numExperts, k int, expertBuilder func(int) (Expert, error)) (*MoELayer, error) {
	if k <= 0 || k > numExperts {
		return nil, fmt.Errorf("k (%d) must be between 1 and numExperts (%d)", k, numExperts)
	}

	gatingNetwork, err := NewGatingNetwork(inputDim, numExperts)
	if err != nil {
		return nil, fmt.Errorf("failed to create gating network: %w", err)
	}

	experts := make([]Expert, numExperts)
	for i := 0; i < numExperts; i++ {
		expert, err := expertBuilder(i)
		if err != nil {
			return nil, fmt.Errorf("failed to create expert %d: %w", i, err)
		}
		experts[i] = expert
	}

	return &MoELayer{
		GatingNetwork: gatingNetwork,
		Experts:       experts,
		K:             k,
		// InputDim:      inputDim, // Initialize InputDim
	}, nil
}

// Parameters returns all learnable parameters of the MoELayer.
func (moe *MoELayer) Parameters() []*Tensor {
	params := moe.GatingNetwork.Parameters()
	for _, expert := range moe.Experts {
		params = append(params, expert.Parameters()...)
	}
	return params
}

// Forward performs the forward pass of the MoELayer.
// It takes an input tensor and returns the combined output of selected experts.
func (moe *MoELayer) Forward(inputs ...*Tensor) (*Tensor, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("MoELayer.Forward expects 1 input, got %d", len(inputs))
	}
	input := inputs[0]
	moe.inputTensor = input

	batchSize := input.Shape[0]
	var seqLength int
	var embeddingDim int

	if len(input.Shape) == 3 {
		seqLength = input.Shape[1]
		embeddingDim = input.Shape[2]
	} else if len(input.Shape) == 2 {
		seqLength = 1 // Assuming each row is a single token's embedding
		embeddingDim = input.Shape[1]
	} else {
		return nil, fmt.Errorf("MoELayer.Forward expects 2D or 3D input, got shape %v", input.Shape)
	}

	reshapedInput, err := input.Reshape([]int{batchSize * seqLength, embeddingDim})
	if err != nil {
		return nil, fmt.Errorf("failed to reshape input for gating network: %w", err)
	}

	// 1. Gating Network (Router) forward pass to get logits
	gateLogitsReshaped, err := moe.GatingNetwork.Forward(reshapedInput)
	if err != nil {
		return nil, fmt.Errorf("moe layer gating network forward failed: %w", err)
	}

	// Add noise for soft selection during training
	if moe.Training {
		noise := make([]float64, len(gateLogitsReshaped.Data))
		for i := range noise {
			noise[i] = rand.NormFloat64()
		}
		noiseTensor := NewTensor(gateLogitsReshaped.Shape, noise, false)
		gateLogitsReshaped, err = gateLogitsReshaped.Add(noiseTensor)
		if err != nil {
			return nil, fmt.Errorf("failed to add noise to gate logits: %w", err)
		}
	}

	// Apply softmax to get probabilities
	gateOutputsReshaped, err := gateLogitsReshaped.Softmax(len(gateLogitsReshaped.Shape) - 1)
	if err != nil {
		return nil, fmt.Errorf("gating network softmax failed: %w", err)
	}

	// Reshape gateOutputs back to [batchSize, seqLength, numExperts]
	numExperts := len(moe.Experts)
	// Create a new Tensor with the desired shape and copy data to avoid potential aliasing issues
	gateOutputsData := make([]float64, batchSize*seqLength*numExperts)
	copy(gateOutputsData, gateOutputsReshaped.Data)
	moe.gateOutputs = NewTensor([]int{batchSize, seqLength, numExperts}, gateOutputsData, gateOutputsReshaped.RequiresGrad)

	// Ensure gateOutputs has the expected shape [batch_size, sequence_length, num_experts]
	if len(moe.gateOutputs.Shape) != 3 || moe.gateOutputs.Shape[0] != batchSize || moe.gateOutputs.Shape[1] != seqLength || moe.gateOutputs.Shape[2] != numExperts {
		return nil, fmt.Errorf("unexpected gateOutputs shape after explicit copy: %v, expected [%d, %d, %d]", moe.gateOutputs.Shape, batchSize, seqLength, numExperts)
	}

	// Prepare to store expert outputs and selected expert indices
	moe.expertOutputs = make([]*Tensor, numExperts)
	moe.expertActivations = make([]*Tensor, numExperts)
	moe.selectedExperts = make([][]int, batchSize*seqLength)

	// Output tensor will have the same shape as input
	outputShape := input.Shape
	combinedOutputData := make([]float64, batchSize*seqLength*embeddingDim)
	combinedOutput := NewTensor(outputShape, combinedOutputData, false)
	if input.RequiresGrad {
		combinedOutput.RequiresGrad = true
	}

	// Load balancing loss calculation
	tokensPerExpert := make([]float64, numExperts)
	routerProbPerExpert := make([]float64, numExperts)

	// Process each token in the batch
	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLength; s++ {
			tokenIdx := b*seqLength + s

			// Get gate scores for the current token
			scores := make([]float64, numExperts)
			for e := 0; e < numExperts; e++ {
				scores[e] = moe.gateOutputs.Data[tokenIdx*numExperts+e]
				routerProbPerExpert[e] += scores[e]
			}

			// Select top-K experts
			topKIndices := make([]int, numExperts)
			for i := range topKIndices {
				topKIndices[i] = i
			}

			sort.SliceStable(topKIndices, func(i, j int) bool {
				return scores[topKIndices[i]] > scores[topKIndices[j]]
			})

			moe.selectedExperts[tokenIdx] = topKIndices[:moe.K]

			// Extract input for the current token
			inputForExpertData := make([]float64, embeddingDim)
			copy(inputForExpertData, input.Data[tokenIdx*embeddingDim:(tokenIdx+1)*embeddingDim])
			inputForExpert := NewTensor([]int{1, embeddingDim}, inputForExpertData, false)
			if input.RequiresGrad {
				inputForExpert.RequiresGrad = true
			}

			// Run selected experts and combine their outputs
			for _, expertIdx := range moe.selectedExperts[tokenIdx] {
				tokensPerExpert[expertIdx]++
				expert := moe.Experts[expertIdx]

				expertOutput, err := expert.Forward(inputForExpert)
				if err != nil {
					return nil, fmt.Errorf("moe layer expert %d forward failed: %w", expertIdx, err)
				}
				moe.expertActivations[expertIdx] = expertOutput

				weight := scores[expertIdx]
				for i := 0; i < embeddingDim; i++ {
					combinedOutput.Data[tokenIdx*embeddingDim+i] += expertOutput.Data[i] * weight
				}
			}
		}
	}

	// Finalize load balancing loss
	if moe.Training {
		totalTokens := float64(batchSize * seqLength)
		for e := 0; e < numExperts; e++ {
			fractionTokens := tokensPerExpert[e] / totalTokens
			fractionRouterProb := routerProbPerExpert[e] / totalTokens
			moe.LoadBalancingLoss += fractionTokens * fractionRouterProb
		}
		moe.LoadBalancingLoss *= float64(numExperts)
	}

	combinedOutput.Creator = moe
	return combinedOutput, nil
}

// Backward performs the backward pass for the MoELayer.
func (moe *MoELayer) Backward(grad *Tensor) error {
	if grad == nil || grad.Data == nil {
		return nil
	}

	// Handle 2D gradient from decoder by processing only the last time step
	if len(grad.Shape) == 2 {
		// The grad is for the context vector, which corresponds to the last element of the sequence.
		// We will create a new grad tensor that has zeros everywhere except for the last time step.
		fullGrad := NewTensor(moe.inputTensor.Shape, make([]float64, len(moe.inputTensor.Data)), false)
		batchSize := grad.Shape[0]
		embeddingDim := grad.Shape[1]
		seqLength := fullGrad.Shape[1]
		for i := 0; i < batchSize; i++ {
			copy(fullGrad.Data[(i*seqLength+seqLength-1)*embeddingDim:(i*seqLength+seqLength)*embeddingDim], grad.Data[i*embeddingDim:(i+1)*embeddingDim])
		}
		grad = fullGrad
	}

	batchSize := moe.inputTensor.Shape[0]
	seqLength := moe.inputTensor.Shape[1]
	embeddingDim := moe.inputTensor.Shape[2]
	numExperts := len(moe.Experts)

	// Initialize gradients for the MoE layer's input
	if moe.inputTensor.RequiresGrad {
		if moe.inputTensor.Grad == nil {
			moe.inputTensor.Grad = NewTensor(moe.inputTensor.Shape, make([]float64, len(moe.inputTensor.Data)), false)
		}
	}

	// Reshape incoming gradient for Gating Network: [batchSize, seqLength, embeddingDim] -> [batchSize * seqLength, embeddingDim]
	gradReshaped, err := grad.Reshape([]int{batchSize * seqLength, embeddingDim})
	if err != nil {
		return fmt.Errorf("failed to reshape incoming gradient for gating network: %w", err)
	}

	// Gradient for the gating network's output (gateOutputs - softmax probabilities)
	gateGradData := make([]float64, batchSize*seqLength*numExperts)
	gateGradReshaped := NewTensor([]int{batchSize * seqLength, numExperts}, gateGradData, false)

	// Iterate through the batch and sequence to distribute gradients
	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLength; s++ {
			tokenIdx := b*seqLength + s

			// Get the gradient for the current token from the combined output
			gradForTokenData := make([]float64, embeddingDim)
			copy(gradForTokenData, gradReshaped.Data[tokenIdx*embeddingDim:(tokenIdx+1)*embeddingDim])
			gradForToken := NewTensor([]int{1, embeddingDim}, gradForTokenData, false)

			// Get the original input for this token
			inputForExpertData := make([]float64, embeddingDim)
			copy(inputForExpertData, moe.inputTensor.Data[tokenIdx*embeddingDim:(tokenIdx+1)*embeddingDim])
			inputForExpert := NewTensor([]int{1, embeddingDim}, inputForExpertData, false)
			if moe.inputTensor.RequiresGrad {
				inputForExpert.RequiresGrad = true
			}

			// Get gate scores for the current token (softmax probabilities)
			scores := make([]float64, numExperts)
			for e := 0; e < numExperts; e++ {
				scores[e] = moe.gateOutputs.Data[tokenIdx*numExperts+e]
			}

			// Distribute gradients to selected experts and accumulate gating network gradients
			for _, expertIdx := range moe.selectedExperts[tokenIdx] {
				expert := moe.Experts[expertIdx]
				weight := scores[expertIdx]

				// 1. Gradient for expert output: dL/dExpertOutput = dL/dCombinedOutput * weight
				gradForExpertOutputData := make([]float64, embeddingDim)
				for i := 0; i < embeddingDim; i++ {
					gradForExpertOutputData[i] = gradForToken.Data[i] * weight
				}
				gradForExpertOutput := NewTensor([]int{1, embeddingDim}, gradForExpertOutputData, false)

				// Backpropagate through the expert
				err := expert.Backward(gradForExpertOutput)
				if err != nil {
					return fmt.Errorf("moe layer expert %d backward failed: %w", expertIdx, err)
				}

				// Accumulate gradient for the input to the MoE layer from experts
				if moe.inputTensor.RequiresGrad && expert.Inputs() != nil && len(expert.Inputs()) > 0 && expert.Inputs()[0].Grad != nil {
					for i := 0; i < embeddingDim; i++ {
						moe.inputTensor.Grad.Data[tokenIdx*embeddingDim+i] += expert.Inputs()[0].Grad.Data[i]
					}
				}

				// 2. Accumulate gradient for the gating network
				gradForGateProb := 0.0
				if moe.expertActivations[expertIdx] != nil {
					for i := 0; i < embeddingDim; i++ {
						gradForGateProb += gradForToken.Data[i] * moe.expertActivations[expertIdx].Data[i]
					}
				}
				gateGradReshaped.Data[tokenIdx*numExperts+expertIdx] += gradForGateProb
			}
		}
	}

	// Finally, backpropagate through the gating network with the accumulated gateGrad.
	err = moe.GatingNetwork.Backward(gateGradReshaped)
	if err != nil {
		return err
	}

	

	return nil
}

// Inputs returns the input tensors of the MoELayer's last forward operation.
func (moe *MoELayer) Inputs() []*Tensor {
	if moe.inputTensor != nil {
		return []*Tensor{moe.inputTensor}
	}
	return []*Tensor{}
}

// SetMode sets the mode for the MoELayer and all its experts.
func (moe *MoELayer) SetMode(training bool) {
	moe.Training = training
	for _, expert := range moe.Experts {
		expert.SetMode(training)
	}
}

func (moe *MoELayer) GetOutputShape() []int {
	return moe.inputTensor.Shape
}
