package moe

import (
	"fmt"
	. "nlptagger/neural/nn"
	. "nlptagger/neural/tensor"
)

// GatingNetwork (Router) determines which experts to activate for a given input.
type GatingNetwork struct {
	Linear *Linear
	// Stored for backward pass
	inputTensor *Tensor
	outputTensor *Tensor
}

// NewGatingNetwork creates a new GatingNetwork.
// inputDim is the dimension of the input to the gating network.
// numExperts is the number of experts in the MoE layer.
func NewGatingNetwork(inputDim, numExperts int) (*GatingNetwork, error) {
	linear, err := NewLinear(inputDim, numExperts)
	if err != nil {
		return nil, fmt.Errorf("failed to create linear layer for gating network: %w", err)
	}
	return &GatingNetwork{Linear: linear},
		nil
}

// Forward performs the forward pass of the GatingNetwork.
// It takes an input tensor and returns a tensor of expert weights (probabilities).
func (gn *GatingNetwork) Forward(inputs ...*Tensor) (*Tensor, error) {
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

	// Apply softmax to get probabilities (weights for each expert)
	weights, err := logits.Softmax(len(logits.Shape) - 1) // Softmax along the last dimension
	if err != nil {
		return nil, fmt.Errorf("gating network softmax failed: %w", err)
	}

	gn.outputTensor = weights
	// The creator of weights is the softmax operation, which is already set internally by weights.Softmax.
	// We do not need to set gn as the creator here, as gn is the operation that *uses* the softmax output,
	// not the one that *creates* it.

	return weights, nil
}

// Backward performs the backward pass for the GatingNetwork.
func (gn *GatingNetwork) Backward(grad *Tensor) error {
	if grad == nil || grad.Data == nil {
		return nil
	}

	// The creator of gn.outputTensor is the Softmax operation.
	// When we call Backward on gn.outputTensor, it will trigger the
	// backward pass of the Softmax operation, which will compute the
	// gradient with respect to its input (the logits).
	// Then, it will recursively call Backward on the logits tensor.
	// The creator of the logits tensor is the Linear layer (gn.Linear).
	// This will trigger the backward pass of the Linear layer, which will
	// compute the gradients for its weights, biases, and its own input.

	// We just need to set the initial gradient for the output tensor
	// and start the backpropagation process.

	if gn.outputTensor.Grad == nil {
		gn.outputTensor.Grad = NewTensor(grad.Shape, make([]float64, len(grad.Data)), false)
	}
	// Accumulate the incoming gradient
	for i := range grad.Data {
		gn.outputTensor.Grad.Data[i] += grad.Data[i]
	}

	// Propagate the gradient further down the graph
	// The creator of gn.outputTensor is the Softmax operation.
	// Calling Backward on gn.outputTensor.Creator will trigger the backward pass
	// for the Softmax, and then recursively for the Linear layer.
	if gn.outputTensor.Creator != nil {
		err := gn.outputTensor.Creator.Backward(gn.outputTensor.Grad)
		if err != nil {
			return fmt.Errorf("error during backward pass for GatingNetwork's output tensor: %w", err)
		}
	}

	return nil
}

// Parameters returns all learnable parameters of the GatingNetwork.
func (gn *GatingNetwork) Parameters() []*Tensor {
	return gn.Linear.Parameters()
}

// Inputs returns the input tensors of the GatingNetwork's last forward operation.
func (gn *GatingNetwork) Inputs() []*Tensor {
	if gn.inputTensor != nil {
		return []*Tensor{gn.inputTensor}
	}
	return []*Tensor{}
}
