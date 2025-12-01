package nn

import (
	"fmt"
	"math"

	. "github.com/zendrulat/nlptagger/neural/tensor"
)

// LayerNorm represents a layer normalization module
type LayerNorm struct {
	NormalizedShape int
	Gamma           *Tensor // Learnable scale parameter
	Beta            *Tensor // Learnable shift parameter
	Eps             float64

	// Stored for backward pass
	input      *Tensor
	normalized *Tensor
	mean       *Tensor
	variance   *Tensor
}

// NewLayerNorm creates a new LayerNorm module
func NewLayerNorm(normalizedShape int) *LayerNorm {
	// Initialize gamma to ones and beta to zeros
	gamma := NewTensor([]int{normalizedShape}, make([]float64, normalizedShape), true)
	beta := NewTensor([]int{normalizedShape}, make([]float64, normalizedShape), true)

	for i := range gamma.Data {
		gamma.Data[i] = 1.0
		beta.Data[i] = 0.0
	}

	return &LayerNorm{
		NormalizedShape: normalizedShape,
		Gamma:           gamma,
		Beta:            beta,
		Eps:             1e-5,
	}
}

// Forward performs the forward pass of LayerNorm
// Input shape: [batchSize, normalizedShape]
func (ln *LayerNorm) Forward(input *Tensor) (*Tensor, error) {
	if len(input.Shape) != 2 {
		return nil, fmt.Errorf("LayerNorm expects 2D input, got shape %v", input.Shape)
	}
	if input.Shape[1] != ln.NormalizedShape {
		return nil, fmt.Errorf("LayerNorm expects last dimension %d, got %d", ln.NormalizedShape, input.Shape[1])
	}

	batchSize := input.Shape[0]
	ln.input = input

	// Calculate mean and variance for each sample
	mean := NewTensor([]int{batchSize}, make([]float64, batchSize), false)
	variance := NewTensor([]int{batchSize}, make([]float64, batchSize), false)

	for i := 0; i < batchSize; i++ {
		// Calculate mean
		sum := 0.0
		for j := 0; j < ln.NormalizedShape; j++ {
			sum += input.Data[i*ln.NormalizedShape+j]
		}
		mean.Data[i] = sum / float64(ln.NormalizedShape)

		// Calculate variance
		varSum := 0.0
		for j := 0; j < ln.NormalizedShape; j++ {
			diff := input.Data[i*ln.NormalizedShape+j] - mean.Data[i]
			varSum += diff * diff
		}
		variance.Data[i] = varSum / float64(ln.NormalizedShape)
	}

	ln.mean = mean
	ln.variance = variance

	// Normalize
	normalized := NewTensor(input.Shape, make([]float64, len(input.Data)), input.RequiresGrad)
	for i := 0; i < batchSize; i++ {
		std := math.Sqrt(variance.Data[i] + ln.Eps)
		for j := 0; j < ln.NormalizedShape; j++ {
			normalized.Data[i*ln.NormalizedShape+j] = (input.Data[i*ln.NormalizedShape+j] - mean.Data[i]) / std
		}
	}
	ln.normalized = normalized

	// Scale and shift
	output := NewTensor(input.Shape, make([]float64, len(input.Data)), input.RequiresGrad)
	for i := 0; i < batchSize; i++ {
		for j := 0; j < ln.NormalizedShape; j++ {
			output.Data[i*ln.NormalizedShape+j] = ln.Gamma.Data[j]*normalized.Data[i*ln.NormalizedShape+j] + ln.Beta.Data[j]
		}
	}

	return output, nil
}

// Backward performs the backward pass of LayerNorm
func (ln *LayerNorm) Backward(gradOutput *Tensor) error {
	batchSize := ln.input.Shape[0]

	// Initialize gradients
	if ln.Gamma.Grad == nil {
		ln.Gamma.Grad = NewTensor(ln.Gamma.Shape, make([]float64, len(ln.Gamma.Data)), false)
	}
	if ln.Beta.Grad == nil {
		ln.Beta.Grad = NewTensor(ln.Beta.Shape, make([]float64, len(ln.Beta.Data)), false)
	}
	if ln.input.Grad == nil {
		ln.input.Grad = NewTensor(ln.input.Shape, make([]float64, len(ln.input.Data)), false)
	}

	// Gradient w.r.t. gamma and beta
	for i := 0; i < batchSize; i++ {
		for j := 0; j < ln.NormalizedShape; j++ {
			ln.Gamma.Grad.Data[j] += gradOutput.Data[i*ln.NormalizedShape+j] * ln.normalized.Data[i*ln.NormalizedShape+j]
			ln.Beta.Grad.Data[j] += gradOutput.Data[i*ln.NormalizedShape+j]
		}
	}

	// Gradient w.r.t. input (simplified version)
	for i := 0; i < batchSize; i++ {
		std := math.Sqrt(ln.variance.Data[i] + ln.Eps)
		for j := 0; j < ln.NormalizedShape; j++ {
			ln.input.Grad.Data[i*ln.NormalizedShape+j] += gradOutput.Data[i*ln.NormalizedShape+j] * ln.Gamma.Data[j] / std
		}
	}

	return nil
}

// Parameters returns the learnable parameters of LayerNorm
func (ln *LayerNorm) Parameters() []*Tensor {
	return []*Tensor{ln.Gamma, ln.Beta}
}
