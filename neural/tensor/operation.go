package tensor

import (
	"fmt"
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
	// Reshape logits to 2D if it's 3D (batch_size * seq_len, vocab_size)
	originalShape := logits.Shape
	var reshapedLogits *Tensor
	var numClasses int

	if len(originalShape) == 3 {
		batchSize := originalShape[0]
		seqLen := originalShape[1]
		numClasses = originalShape[2]
		var err error
		reshapedLogits, err = logits.Reshape([]int{batchSize * seqLen, numClasses})
		if err != nil {
			panic(fmt.Sprintf("Failed to reshape logits: %v", err))
		}
	} else if len(originalShape) == 2 {
		reshapedLogits = logits
		numClasses = originalShape[1]
	} else {
		// Handle other dimensions or return an error
		panic("Unsupported logits dimension for CrossEntropyLoss")
	}

	probs := Softmax(reshapedLogits)
	loss := 0.0
	activeTokens := 0
	epsilon := 1e-9 // Small value to avoid log(0)

	grad := NewTensor(reshapedLogits.Shape, make([]float64, len(reshapedLogits.Data)), false)

	for i := 0; i < reshapedLogits.Shape[0]; i++ {
		targetID := targetIDs[i]
		if targetID == padID {
			continue
		}
		activeTokens++

		// Add epsilon to the probability to avoid log(0)
		loss -= math.Log(probs.Data[i*numClasses+targetID] + epsilon)

		for j := 0; j < numClasses; j++ {
			if j == targetID {
				grad.Data[i*numClasses+j] = probs.Data[i*numClasses+j] - 1
			} else {
				grad.Data[i*numClasses+j] = probs.Data[i*numClasses+j]
			}
		}
	}

	if activeTokens > 0 {
		loss /= float64(activeTokens)
		for i := range grad.Data {
			grad.Data[i] /= float64(activeTokens)
		}
	}

	// Reshape grad back to original shape if it was reshaped
	if len(originalShape) == 3 {
		var err error
		grad, err = grad.Reshape(originalShape)
		if err != nil {
			panic(fmt.Sprintf("Failed to reshape gradient: %v", err))
		}
	}

	return loss, grad
}