package bartsimple

// AddOperation represents an element-wise addition operation for autograd.
type AddOperation struct {
	Input1, Input2 *Tensor
	Output         *Tensor // Store the output tensor to access its shape and data
}

// EmbeddingLookupOperation represents an embedding lookup operation for autograd.
type EmbeddingLookupOperation struct {
	InputIDs *Tensor // Tensor of shape [batch_size, sequence_length]
	Weights  *Tensor // Tensor of shape [vocab_size, embedding_dim]
	Output   *Tensor // Tensor of shape [batch_size, sequence_length, embedding_dim]
}

// Implement the Inputs() method for AddOperation.
func (op *AddOperation) Inputs() []*Tensor {
	return []*Tensor{op.Input1, op.Input2}
}
func (op *addOperation) Backward(grad *Tensor) error {
	// For element-wise addition, the gradient is distributed to the inputs.
	if op.input1.requiresGrad {
		if op.input1.Grad == nil {
			// Initialize gradient tensor if nil
			op.input1.Grad = NewTensor(make([]float64, len(op.input1.Data)), op.input1.Shape, true)
		}
		// Accumulate gradients
		for i := range op.input1.Grad.Data {
			op.input1.Grad.Data[i] += grad.Data[i]
		}
	}
	if op.input2.requiresGrad {
		if op.input2.Grad == nil {
			// Initialize gradient tensor if nil
			op.input2.Grad = NewTensor(make([]float64, len(op.input2.Data)), op.input2.Shape, true)
		}
		// Accumulate gradients
		for i := range op.input2.Grad.Data {
			op.input2.Grad.Data[i] += grad.Data[i]
		}
	}
	return nil
}
