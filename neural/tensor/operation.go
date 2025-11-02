package tensor

import (
	"math"
)

// EmbeddingLookupOperation represents an embedding lookup operation for autograd.
type EmbeddingLookupOperation struct {
	InputIDs *Tensor // Tensor of shape [batch_size, sequence_length]
	Weights  *Tensor // Tensor of shape [vocab_size, embedding_dim]
	Output   *Tensor // Tensor of shape [batch_size, sequence_length, embedding_dim]
}

// Softmax applies the softmax function to the last dimension of the tensor.
func Softmax(tensor *Tensor) *Tensor {
	shape := tensor.Shape
	lastDim := shape[len(shape)-1]
	output := NewTensor(shape, make([]float64, len(tensor.Data)), false)

	for i := 0; i < len(tensor.Data); i += lastDim {
		maxVal := math.Inf(-1)
		for j := 0; j < lastDim; j++ {
			if tensor.Data[i+j] > maxVal {
				maxVal = tensor.Data[i+j]
			}
		}

		sumExp := 0.0
		for j := 0; j < lastDim; j++ {
			sumExp += math.Exp(tensor.Data[i+j] - maxVal)
		}

		for j := 0; j < lastDim; j++ {
			output.Data[i+j] = math.Exp(tensor.Data[i+j]-maxVal) / sumExp
		}
	}

	return output
}

