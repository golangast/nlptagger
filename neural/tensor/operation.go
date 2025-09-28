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

// CrossEntropyLoss calculates the cross-entropy loss.
func CrossEntropyLoss(logits *Tensor, targetIDs []int, padID int) (float64, *Tensor) {
	probs := Softmax(logits)
	loss := 0.0
	activeTokens := 0
	epsilon := 1e-9 // Small value to avoid log(0)

	grad := NewTensor(logits.Shape, make([]float64, len(logits.Data)), false)

	for i := 0; i < logits.Shape[0]; i++ {
		targetID := targetIDs[i]
		if targetID == padID {
			continue
		}
		activeTokens++

		// Add epsilon to the probability to avoid log(0)
		loss -= math.Log(probs.Data[i*logits.Shape[1]+targetID] + epsilon)

		for j := 0; j < logits.Shape[1]; j++ {
			if j == targetID {
				grad.Data[i*logits.Shape[1]+j] = probs.Data[i*logits.Shape[1]+j] - 1
			} else {
				grad.Data[i*logits.Shape[1]+j] = probs.Data[i*logits.Shape[1]+j]
			}
		}
	}

	if activeTokens > 0 {
		loss /= float64(activeTokens)
		for i := range grad.Data {
			grad.Data[i] /= float64(activeTokens)
		}
	}

	return loss, grad
}