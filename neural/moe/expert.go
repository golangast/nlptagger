package moe

import (
	"github.com/zendrulat/nlptagger/neural/tensor"
)

// Expert is an interface for an expert network in a Mixture of Experts model.
type Expert interface {
	// Forward performs the forward pass of the expert network.
	// It takes a tensor of shape (batch_size, input_dim) and returns a tensor of shape (batch_size, output_dim).
	Forward(input *tensor.Tensor) (*tensor.Tensor, error)

	// Backward performs the backward pass of the expert network.
	Backward(grad *tensor.Tensor) error

	// Parameters returns all learnable parameters of the expert.
	Parameters() []*tensor.Tensor

	// Inputs returns the input tensors of the expert's last forward operation.
	Inputs() []*tensor.Tensor

	// Description returns a string description of the expert.
	Description() string

	// SetMode sets the mode of the expert (training or inference).
	SetMode(training bool)
}

