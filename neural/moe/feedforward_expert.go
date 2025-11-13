package moe

import (
	
	"fmt"
	"github.com/zendrulat/nlptagger/neural/nn"
	"github.com/zendrulat/nlptagger/neural/tensor"
)



// FeedForwardExpert is a simple feed-forward neural network that implements the Expert interface.
type FeedForwardExpert struct {
	Layer1 *nn.Linear
	Layer2 *nn.Linear
	// Stored for backward pass
	inputTensor *tensor.Tensor
	intermediateOutput *tensor.Tensor
}

// NewFeedForwardExpert creates a new FeedForwardExpert.
// inputDim is the dimension of the input to the expert.
// hiddenDim is the dimension of the hidden layer.
// outputDim is the dimension of the output from the expert.
func NewFeedForwardExpert(inputDim, hiddenDim, outputDim int) (*FeedForwardExpert, error) {
	layer1, err := nn.NewLinear(inputDim, hiddenDim)
	if err != nil {
		return nil, fmt.Errorf("failed to create first linear layer for expert: %w", err)
	}
	layer2, err := nn.NewLinear(hiddenDim, outputDim)
	if err != nil {
		return nil, fmt.Errorf("failed to create second linear layer for expert: %w", err)
	}

	return &FeedForwardExpert{
		Layer1: layer1,
		Layer2: layer2,
	}, nil
}

// Forward performs the forward pass of the FeedForwardExpert.
func (e *FeedForwardExpert) Forward(input *tensor.Tensor) (*tensor.Tensor, error) {
	e.inputTensor = input

	// Layer 1: Linear -> ReLU (assuming ReLU is applied implicitly or as part of a custom activation)
	// For simplicity, let's just use linear layers for now. Add activation functions if needed.
	output1, err := e.Layer1.Forward(input)
	if err != nil {
		return nil, fmt.Errorf("expert layer 1 forward failed: %w", err)
	}
	e.intermediateOutput = output1 // Store for backward pass

	// Layer 2: Linear
	output2, err := e.Layer2.Forward(output1)
	if err != nil {
		return nil, fmt.Errorf("expert layer 2 forward failed: %w", err)
	}

	return output2, nil
}

// Backward performs the backward pass of the FeedForwardExpert.
func (e *FeedForwardExpert) Backward(grad *tensor.Tensor) error {
	if grad == nil || grad.Data == nil {
		return nil
	}

	// Backpropagate through Layer2
	err := e.Layer2.Backward(grad)
	if err != nil {
		return fmt.Errorf("expert layer 2 backward failed: %w", err)
	}

	// Backpropagate through Layer1
	// The gradient for Layer1 is the gradient of its output, which is e.intermediateOutput.Grad
	if e.intermediateOutput == nil || e.intermediateOutput.Grad == nil {
		return fmt.Errorf("expert intermediate output or its gradient is nil in backward")
	}
	err = e.Layer1.Backward(e.intermediateOutput.Grad)
	if err != nil {
		return fmt.Errorf("expert layer 1 backward failed: %w", err)
	}



	return nil
}

// Parameters returns all learnable parameters of the FeedForwardExpert.
func (e *FeedForwardExpert) Parameters() []*tensor.Tensor {
	params := e.Layer1.Parameters()
	params = append(params, e.Layer2.Parameters()...)
	return params
}

// Inputs returns the input tensors of the FeedForwardExpert's last forward operation.
func (e *FeedForwardExpert) Inputs() []*tensor.Tensor {
	if e.inputTensor != nil {
		return []*tensor.Tensor{e.inputTensor}
	}
	return []*tensor.Tensor{}
}

// Description returns a string description of the expert.
func (e *FeedForwardExpert) Description() string {
	return "FeedForwardExpert"
}

// SetMode sets the mode for the expert.
func (e *FeedForwardExpert) SetMode(training bool) {
	// No specific behavior for training/inference in this simple expert
}
