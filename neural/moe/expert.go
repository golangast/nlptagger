package moe

import (
	. "nlptagger/neural/tensor"
)

// Expert defines the interface for an expert network within an MoE layer.
// An expert typically processes a portion of the input and contributes to the final output.
type Expert interface {
	// Forward performs the forward pass of the expert.
	// It takes an input tensor and returns an output tensor.
	Forward(inputs ...*Tensor) (*Tensor, error)

	// Backward performs the backward pass of the expert.
	// It takes a gradient tensor from the output and propagates it backward.
	Backward(grad *Tensor) error

	// Parameters returns all learnable parameters of the expert.
	Parameters() []*Tensor

	// Inputs returns the input tensors of the expert's last forward operation.
	Inputs() []*Tensor
}
