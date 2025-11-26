package moe

import (
	"fmt"
	"sort"

	. "github.com/zendrulat/nlptagger/neural/tensor"
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
	expertActivations [][]*Tensor // Output of experts before combining
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

	// 1. Gating Network (Router) forward pass to get logits
	gateLogits, err := moe.GatingNetwork.Forward(input)
	if err != nil {
		return nil, fmt.Errorf("moe layer gating network forward failed: %w", err)
	}

	// Apply softmax to get probabilities
	gateOutputs, err := gateLogits.Softmax(len(gateLogits.Shape) - 1)
	if err != nil {
		return nil, fmt.Errorf("gating network softmax failed: %w", err)
	}
	moe.gateOutputs = gateOutputs

	batchSize := input.Shape[0]
	seqLength := input.Shape[1]
	embeddingDim := input.Shape[2]
	numExperts := len(moe.Experts)

	moe.selectedExperts = make([][]int, batchSize*seqLength)
	moe.expertActivations = make([][]*Tensor, batchSize*seqLength) // Initialize expertActivations
	finalOutput := NewTensor([]int{batchSize, seqLength, embeddingDim}, make([]float64, batchSize*seqLength*embeddingDim), true)

	for i := 0; i < batchSize*seqLength; i++ {
		scores := gateOutputs.Data[i*numExperts : (i+1)*numExperts]
		topKIndices := make([]int, numExperts)
		for j := range topKIndices {
			topKIndices[j] = j
		}
		sort.SliceStable(topKIndices, func(a, b int) bool {
			return scores[topKIndices[a]] > scores[topKIndices[b]]
		})
		moe.selectedExperts[i] = topKIndices[:moe.K]

		moe.expertActivations[i] = make([]*Tensor, numExperts) // Initialize inner slice for current token

		// Get the input for the current token by slicing the main input tensor
		// This preserves the computation graph for backpropagation.
		b := i / seqLength
		s := i % seqLength
		tokenInput3D, err := input.Slice(0, b, b+1)
		if err != nil {
			return nil, fmt.Errorf("failed to slice input for token (batch): %w", err)
		}
		tokenInput3D, err = tokenInput3D.Slice(1, s, s+1)
		if err != nil {
			return nil, fmt.Errorf("failed to slice input for token (seq): %w", err)
		}
		tokenInput, err := tokenInput3D.Reshape([]int{1, embeddingDim})
		if err != nil {
			return nil, fmt.Errorf("failed to reshape input for token: %w", err)
		}

		// Run selected experts and combine their outputs
		for _, expertIdx := range moe.selectedExperts[i] {
			expert := moe.Experts[expertIdx]
			expertOutput, err := expert.Forward(tokenInput)
			if err != nil {
				return nil, fmt.Errorf("moe layer expert %d forward failed: %w", expertIdx, err)
			}
			moe.expertActivations[i][expertIdx] = expertOutput // Store expert output

			weight := scores[expertIdx]
			for j := 0; j < embeddingDim; j++ {
				finalOutput.Data[i*embeddingDim+j] += expertOutput.Data[j] * weight
			}
		}
	}

	finalOutput.Creator = moe
	return finalOutput, nil
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

	// Initialize a temporary tensor to accumulate gradients for moe.inputTensor
	inputGradAccumulator := NewTensor(moe.inputTensor.Shape, make([]float64, len(moe.inputTensor.Data)), false)

	// Reshape grad to be [batchSize*seqLength, embeddingDim]
	gradReshaped, err := grad.Reshape([]int{batchSize * seqLength, embeddingDim})
	if err != nil {
		return fmt.Errorf("failed to reshape grad: %w", err)
	}

	// Initialize a tensor to accumulate gradients for the gating network
	gateGradReshaped := NewTensor([]int{batchSize * seqLength, numExperts}, make([]float64, batchSize*seqLength*numExperts), true)

	// Iterate through the batch and sequence to distribute gradients
	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLength; s++ {
			tokenIdx := b*seqLength + s

			// Get the gradient for the current token from the combined output
			gradForTokenData := make([]float64, embeddingDim)
			copy(gradForTokenData, gradReshaped.Data[tokenIdx*embeddingDim:(tokenIdx+1)*embeddingDim])
			gradForToken := NewTensor([]int{1, embeddingDim}, gradForTokenData, false)

			// Get the original input for this token
			tokenInputData := moe.inputTensor.Data[tokenIdx*embeddingDim : (tokenIdx+1)*embeddingDim]
			inputForExpert := NewTensor([]int{1, embeddingDim}, tokenInputData, false)
			inputForExpert.RequiresGrad = moe.inputTensor.RequiresGrad

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
				if moe.inputTensor.RequiresGrad && inputForExpert.Grad != nil {
					for i := 0; i < embeddingDim; i++ {
						inputGradAccumulator.Data[tokenIdx*embeddingDim+i] += inputForExpert.Grad.Data[i]
					}
				}

				// 2. Accumulate gradient for the gating network
				gradForGateProb := 0.0
				if moe.expertActivations[tokenIdx][expertIdx] != nil {
					for i := 0; i < embeddingDim; i++ {
						gradForGateProb += gradForToken.Data[i] * moe.expertActivations[tokenIdx][expertIdx].Data[i]
					}
				}
				gateGradReshaped.Data[tokenIdx*numExperts+expertIdx] += gradForGateProb
			}
		}
	}

	// After all experts have processed, add the accumulated input gradients to moe.inputTensor.Grad
	if moe.inputTensor.RequiresGrad {
		if moe.inputTensor.Grad == nil {
			moe.inputTensor.Grad = NewTensor(moe.inputTensor.Shape, make([]float64, len(moe.inputTensor.Data)), false)
		}
		for i := range moe.inputTensor.Grad.Data {
			moe.inputTensor.Grad.Data[i] += inputGradAccumulator.Data[i]
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
