package moe

import (
	
	"fmt"
	"github.com/zendrulat/nlptagger/neural/nn"
	"github.com/zendrulat/nlptagger/neural/tensor"
)


// GatingNetwork (Router) determines which experts to activate for a given input.
type GatingNetwork struct {
	Linear *nn.Linear
	// Stored for backward pass
	inputTensor *tensor.Tensor
	outputTensor *tensor.Tensor
}

// NewGatingNetwork creates a new GatingNetwork.
// inputDim is the dimension of the input to the gating network.
// numExperts is the number of experts in the MoE layer.
func NewGatingNetwork(inputDim, numExperts int) (*GatingNetwork, error) {
	linear, err := nn.NewLinear(inputDim, numExperts)
	if err != nil {
		return nil, fmt.Errorf("failed to create linear layer for gating network: %w", err)
	}
	return &GatingNetwork{Linear: linear},
		nil
}

// Forward performs the forward pass of the GatingNetwork.
// It takes an input tensor and returns a tensor of expert logits.
func (gn *GatingNetwork) Forward(inputs ...*tensor.Tensor) (*tensor.Tensor, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("GatingNetwork.Forward expects 1 input, got %d", len(inputs))
	}
	input := inputs[0]
	gn.inputTensor = input

	// Apply linear transformation
	logits, err := gn.Linear.Forward(input)
	if err != nil {
		return nil, fmt.Errorf("gating network linear forward failed: %w", err)
	}

	gn.outputTensor = logits

	return logits, nil
}

// Backward performs the backward pass for the GatingNetwork.
func (gn *GatingNetwork) Backward(grad *tensor.Tensor) error {
	if grad == nil || grad.Data == nil {
		return nil
	}

	// The creator of gn.outputTensor is the Linear layer (gn.Linear).
	// When we call Backward on gn.outputTensor, it will trigger the
	// backward pass of the Linear layer, which will compute the
	// gradients for its weights, biases, and its own input.

	// We just need to set the initial gradient for the output tensor
	// and start the backpropagation process.

	if gn.outputTensor.Grad == nil {
		gn.outputTensor.Grad = tensor.NewTensor(grad.Shape, make([]float64, len(grad.Data)), false)
	}
	// Accumulate the incoming gradient
	for i := range grad.Data {
		gn.outputTensor.Grad.Data[i] += grad.Data[i]
	}

	// Propagate the gradient further down the graph
	// The creator of gn.outputTensor is the Linear layer.
	// Calling Backward on gn.outputTensor.Creator will trigger the backward pass
	// for the Linear layer.
	if gn.outputTensor.Creator != nil {
		err := gn.outputTensor.Creator.Backward(gn.outputTensor.Grad)
		if err != nil {
			return fmt.Errorf("error during backward pass for GatingNetwork's output tensor: %w", err)
		}
	}

	// Clear gradients after use
	if gn.inputTensor != nil {
		gn.inputTensor.Grad = nil
	}
	if gn.outputTensor != nil {
		gn.outputTensor.Grad = nil
	}

	return nil
}

// Parameters returns all learnable parameters of the GatingNetwork.
func (gn *GatingNetwork) Parameters() []*tensor.Tensor {
	return gn.Linear.Parameters()
}

// Inputs returns the input tensors of the GatingNetwork's last forward operation.
func (gn *GatingNetwork) Inputs() []*tensor.Tensor {
	if gn.inputTensor != nil {
		return []*tensor.Tensor{gn.inputTensor}
	}
	return []*tensor.Tensor{}
}
