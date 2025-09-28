package nn

import (
	"fmt"
	. "nlptagger/neural/tensor"
)

// FeedForward represents a simple feed-forward network.
type FeedForward struct {
	Linear1 *Linear
	Linear2 *Linear
	// Stored for backward pass
	inputTensor     *Tensor
	activatedHidden *Tensor
}

// NewFeedForward creates a new FeedForward layer.
func NewFeedForward(dimModel, hiddenDim int) (*FeedForward, error) {
	linear1, err := NewLinear(dimModel, hiddenDim)
	if err != nil {
		return nil, fmt.Errorf("failed to create feed-forward linear1 layer: %w", err)
	}
	linear2, err := NewLinear(hiddenDim, dimModel)
	if err != nil {
		return nil, fmt.Errorf("failed to create feed-forward linear2 layer: %w", err)
	}
	return &FeedForward{Linear1: linear1, Linear2: linear2}, nil
}

// Parameters returns all learnable parameters of the layer.
func (ff *FeedForward) Parameters() []*Tensor {
	return append(ff.Linear1.Parameters(), ff.Linear2.Parameters()...)
}

// Forward performs the forward pass of the FeedForward layer.
func (ff *FeedForward) Forward(inputs ...*Tensor) (*Tensor, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("FeedForward.Forward expects 1 input, got %d", len(inputs))
	}
	input := inputs[0]
	ff.inputTensor = input

	hidden, err := ff.Linear1.Forward(input)
	if err != nil {
		return nil, err
	}

	// ReLU activation
	ff.activatedHidden = NewTensor(hidden.Shape, make([]float64, len(hidden.Data)), true)
	ff.activatedHidden.RequiresGrad = true
	for i := range hidden.Data {
		if hidden.Data[i] > 0 {
			ff.activatedHidden.Data[i] = hidden.Data[i]
		}
	}

	output, err := ff.Linear2.Forward(ff.activatedHidden)
	if err != nil {
		return nil, err
	}
	output.Creator = ff
	return output, nil
}

// Inputs returns the input tensors of the FeedForward operation.
func (ff *FeedForward) Inputs() []*Tensor {
	if ff.inputTensor != nil {
		return []*Tensor{ff.inputTensor}
	}
	return []*Tensor{}
}

// Backward performs the backward pass for the FeedForward layer.
func (ff *FeedForward) Backward(grad *Tensor) error {
	// Backpropagate through Linear2
	err := ff.Linear2.Backward(grad)
	if err != nil {
		return err
	}

	// Backpropagate through ReLU
	gradHidden := ff.activatedHidden.Grad
	for i := range ff.activatedHidden.Data {
		if ff.activatedHidden.Data[i] == 0 {
			gradHidden.Data[i] = 0
		}
	}

	// Backpropagate through Linear1
	err = ff.Linear1.Backward(gradHidden)
	if err != nil {
		return err
	}

	return nil
}