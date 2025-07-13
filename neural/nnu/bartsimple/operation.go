package bartsimple

import "fmt"

// Operation interface defines the methods required for an operation in the computation graph.
type Operation interface {
	// Forward performs the forward pass of the operation.
	Forward() *Tensor
	// Backward performs the backward pass of the operation.
	Backward(gradOutput *Tensor)
	// Inputs returns the input tensors of the operation.
	Inputs() []*Tensor
}

// AddOperation represents an element-wise addition operation for autograd.
type AddOperation struct {
	Input1, Input2 *Tensor
	Output         *Tensor // Store the output tensor to access its shape and data
}

// MatMulOperation represents a matrix multiplication operation for autograd.
type MatMulOperation struct {
	Input1, Input2 *Tensor
	Output         *Tensor // Store the output tensor to access its shape and data
}

// Forward performs the matrix multiplication of the input tensors and returns the result.
// Forward performs the forward pass for MatMulOperation.
func (op *MatMulOperation) Forward() *Tensor {
	// Perform matrix multiplication of Input1 and Input2
	result, err := op.Input1.MatMul(op.Input2)
	if err != nil {
		// Handle the error appropriately, perhaps panic or return an error
		panic(fmt.Sprintf("Error during MatMulOperation forward pass: %v", err))
	}
	// Store the output tensor in the operation
	op.Output = result
	return result
}

// Backward performs the backward pass for MatMulOperation.
func (op *MatMulOperation) Backward(gradOutput *Tensor) {
	// If Input1 required gradient, compute and accumulate its gradient.
	if op.Input1.requiresGrad {
		// Compute gradient for Input1: gradOutput @ Input2^T
		// This is done by matrix multiplying the output gradient with the transpose of Input2.
		input2Transposed, err := op.Input2.Transpose(len(op.Input2.Shape)-2, len(op.Input2.Shape)-1) // Transpose the last two dimensions
		if err != nil {
			// Handle error appropriately, e.g., panic or return error
			panic(fmt.Sprintf("Error transposing Input2 in MatMulOperation Backward: %v", err))
		}
		// Perform the matrix multiplication: gradOutput @ Input2^T
		gradInput1, err := gradOutput.MatMul(input2Transposed)
		if err != nil {
			panic(fmt.Sprintf("Error computing gradInput1 in MatMulOperation Backward: %v", err))
		}
		// If the gradient for Input1 is nil, initialize it. Otherwise, accumulate the new gradient.
		if op.Input1.Grad == nil {
			op.Input1.Grad = NewTensor(gradInput1.Data, gradInput1.Shape) // Initialize Grad with the computed gradient
		} else {
			op.Input1.Grad, err = op.Input1.Grad.Add(gradInput1) // Accumulate gradient by adding to the existing gradient
			if err != nil {
				// Handle error during gradient accumulation
				panic(fmt.Sprintf("Error accumulating gradInput1 in MatMulOperation Backward: %v", err))
			}
		}
	}

	// If Input2 required gradient, compute and accumulate its gradient.
	if op.Input2.requiresGrad {
		// Compute gradient for Input2: Input1^T @ gradOutput
		// This is done by matrix multiplying the transpose of Input1 with the output gradient.
		input1Transposed, err := op.Input1.Transpose(len(op.Input1.Shape)-2, len(op.Input1.Shape)-1) // Transpose the last two dimensions
		if err != nil {
			// Handle error appropriately
			panic(fmt.Sprintf("Error transposing Input1 in MatMulOperation Backward: %v", err))
		}
		// Perform the matrix multiplication: Input1^T @ gradOutput
		gradInput2, err := input1Transposed.MatMul(gradOutput)
		if err != nil {
			panic(fmt.Sprintf("Error computing gradInput2 in MatMulOperation Backward: %v", err))
		}
		// If the gradient for Input2 is nil, initialize it. Otherwise, accumulate the new gradient.
		if op.Input2.Grad == nil {
			op.Input2.Grad = NewTensor(gradInput2.Data, gradInput2.Shape) // Initialize Grad with the computed gradient
		} else {
			op.Input2.Grad, err = op.Input2.Grad.Add(gradInput2) // Accumulate gradient by adding to the existing gradient
			if err != nil {
				// Handle error during gradient accumulation
				panic(fmt.Sprintf("Error accumulating gradInput2 in MatMulOperation Backward: %v", err))
			}
		}
	}

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

// Implement the Inputs() method for MatMulOperation.
func (op *MatMulOperation) Inputs() []*Tensor {
	return []*Tensor{op.Input1, op.Input2}
}
