package nn

import (
	"fmt"

	. "github.com/zendrulat/nlptagger/neural/tensor"
)

// FeedForward represents a simple feed-forward layer.
type FeedForward struct {
	Linear *Linear
}

// NewFeedForward creates a new FeedForward layer.
func NewFeedForward(inputDim, hiddenDim, outputDim int) (*FeedForward, error) {
	linear, err := NewLinear(inputDim, outputDim)
	if err != nil {
		return nil, fmt.Errorf("failed to create linear layer for feed-forward network: %w", err)
	}
	return &FeedForward{Linear: linear}, nil
}

// Forward performs the forward pass of the FeedForward layer.
func (f *FeedForward) Forward(inputs ...*Tensor) (*Tensor, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("FeedForward.Forward expects 1 input, got %d", len(inputs))
	}
	return f.Linear.Forward(inputs[0])
}

// Backward performs the backward pass for the FeedForward layer.
func (f *FeedForward) Backward(grad *Tensor) error {
	return f.Linear.Backward(grad)
}

// Parameters returns all learnable parameters of the FeedForward layer.
func (f *FeedForward) Parameters() []*Tensor {
	return f.Linear.Parameters()
}

// Inputs returns the input tensors of the FeedForward operation.
func (f *FeedForward) Inputs() []*Tensor {
	return f.Linear.Inputs()
}
