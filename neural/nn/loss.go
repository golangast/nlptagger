package nn

import (
	"fmt"
	"math"

	"github.com/zendrulat/nlptagger/neural/tensor"
)

// CrossEntropyLoss calculates the cross-entropy loss and its gradient.
// logits: A tensor of shape [batch_size, num_classes] or [batch_size, sequence_length, num_classes]
// targets: A slice of integers representing the true class indices.
// paddingID: The ID of the padding token to ignore in loss calculation.
// Returns the scalar loss and a slice of gradient tensors for each logit tensor.
func CrossEntropyLoss(logits *tensor.Tensor, targets []int, paddingID int) (float64, *tensor.Tensor) {
	if len(logits.Shape) < 2 || len(logits.Shape) > 3 {
		panic(fmt.Sprintf("CrossEntropyLoss expects logits to be 2D or 3D, got shape %v", logits.Shape))
	}

	numClasses := logits.Shape[len(logits.Shape)-1]
	batchSize := logits.Shape[0]

	var sequenceLength int
	if len(logits.Shape) == 3 {
		sequenceLength = logits.Shape[1]
	} else {
		sequenceLength = 1 // For 2D logits, treat as sequence length 1
	}

	if len(targets) != batchSize*sequenceLength {
		panic(fmt.Sprintf("Mismatched target and logit dimensions: targets %d, logits batch*seq %d", len(targets), batchSize*sequenceLength))
	}

	loss := 0.0
	gradData := make([]float64, len(logits.Data))

	for b := 0; b < batchSize; b++ {
		for s := 0; s < sequenceLength; s++ {
			targetIndex := b*sequenceLength + s
			target := targets[targetIndex]

			// Ignore padding tokens in loss calculation
			if target == paddingID {
				continue
			}

			// Calculate softmax probabilities
			startIdx := (b*sequenceLength + s) * numClasses
			endIdx := startIdx + numClasses
			logitSlice := logits.Data[startIdx:endIdx]

			maxLogit := -math.MaxFloat64
			for _, val := range logitSlice {
				if val > maxLogit {
					maxLogit = val
				}
			}

			sumExp := 0.0
			expLogits := make([]float64, numClasses)
			for i, val := range logitSlice {
				expLogits[i] = math.Exp(val - maxLogit) // For numerical stability
				sumExp += expLogits[i]
			}

			probabilities := make([]float64, numClasses)
			for i := range probabilities {
				probabilities[i] = expLogits[i] / sumExp
			}

			// Calculate loss
			if target >= 0 && target < numClasses {
				loss -= math.Log(probabilities[target] + 1e-9) // Add small epsilon for numerical stability
			} else {
				// Handle out-of-bounds target, e.g., by skipping or returning an error
				// For now, we'll just skip, but a more robust solution might be needed.
				continue
			}

			// Calculate gradient for this slice
			for i := 0; i < numClasses; i++ {
				if i == target {
					gradData[startIdx+i] = probabilities[i] - 1.0
				} else {
					gradData[startIdx+i] = probabilities[i]
				}
			}
		}
	}

	// Average loss over non-padding elements
	numNonPadding := 0
	for _, target := range targets {
		if target != paddingID {
			numNonPadding++
		}
	}
	if numNonPadding > 0 {
		loss /= float64(numNonPadding)
	} else {
		loss = 0.0 // No non-padding elements, so no loss
	}

	// The gradient should also be scaled by the number of non-padding elements
	if numNonPadding > 0 {
		for i := range gradData {
			gradData[i] /= float64(numNonPadding)
		}
	}

	gradTensor := tensor.NewTensor(logits.Shape, gradData, false)
	return loss, gradTensor
}
