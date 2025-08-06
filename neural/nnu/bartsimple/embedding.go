package bartsimple

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

const tenThousand = 10000.0


// Embedding represents a simple token embedding layer.
type Embedding struct {
	Weights   *Tensor // Embedding weights (learnable)
	DimModel  int     // Add this field
	VocabSize int

	// Stored values from forward pass for backward calculation
	inputTokenIDs []int // Store input token IDs
	inputShape    []int // Store input shape
}

// PositionalEmbedding represents a simple positional embedding layer.
type PositionalEmbedding struct {
	MaxSequenceLength  int
	DimModel           int
	PositionEmbeddings *Tensor // Positional embeddings (typically not learnable)

	// Stored values from forward pass for backward calculation
	inputShape  []int   // Store input shape
	inputTensor *Tensor // Add this field
}

// Example constructor (modify based on your actual constructor)
func NewEmbedding(vocabSize, dimModel int) *Embedding {
	// Initialize weights with appropriate shape [vocabSize, dimModel]
	weightsShape := []int{vocabSize, dimModel}
	weightsData := make([]float64, vocabSize*dimModel)    // Initialize with random values or zeros
	weights := NewTensor(weightsData, weightsShape, true) // Embedding weights require gradients

	return &Embedding{
		Weights:  weights,
		DimModel: dimModel, // Initialize the DimModel field
		VocabSize: vocabSize, // Initialize the VocabSize field
	}
}

// Parameters returns all learnable parameters of the layer.
func (e *Embedding) Parameters() []*Tensor {
	return []*Tensor{e.Weights}
}

// Backward performs the backward pass for the token embedding layer.
// grad is the gradient from the subsequent layer.
func (e *Embedding) Backward(grad *Tensor) {
	if grad == nil || grad.Data == nil {
		// No gradient to propagate
		return
	}
	if e.Weights == nil || !e.Weights.requiresGrad {
		// Embedding weights are not learnable or do not require gradients
		return
	}
	if e.inputTokenIDs == nil || len(e.inputTokenIDs) != len(grad.Data)/e.Weights.Shape[1] {
		panic("Embedding backward called before forward or with mismatched input/gradient shapes")
	}

	vocabSize := e.Weights.Shape[0]
	dimModel := e.Weights.Shape[1]
	batchSize := e.inputShape[0]
	seqLength := e.inputShape[1]

	// Ensure gradient for weights is initialized
	if e.Weights.Grad == nil {
		e.Weights.Grad = NewTensor(make([]float64, len(e.Weights.Data)), e.Weights.Shape, false)
	}

	// Accumulate gradients to the embedding weights.
	// For each gradient element in 'grad', add it to the corresponding
	// position in e.Weights.Grad based on the input token ID.
	// The incoming gradient 'grad' has shape [batch_size, sequence_length, dim_model].
	// We need to add the gradient for each embedded vector back to the
	// corresponding row in the embedding weights matrix.

	// Iterate through the incoming gradient (which has the shape of the output from forward)
	gradFlatIndex := 0
	for i := 0; i < batchSize; i++ {
		for j := 0; j < seqLength; j++ {
			tokenID := e.inputTokenIDs[i*seqLength+j]
			if tokenID < 0 || tokenID >= vocabSize {
				// Should not happen if forward passed without error, but add a check
				fmt.Printf("Warning: Invalid token ID %d encountered during Embedding backward.\n", tokenID)
				continue // Skip invalid token IDs
			}

			// Add the gradient for this embedded vector (grad[i, j, :])
			// to the gradient of the corresponding embedding vector in Weights.Grad (Weights.Grad[tokenID, :]).
			embeddingGradStart := tokenID * dimModel
			gradVectorStart := gradFlatIndex // Current position in flattened grad data

			for k := 0; k < dimModel; k++ {
				if embeddingGradStart+k >= len(e.Weights.Grad.Data) || gradVectorStart+k >= len(grad.Data) {
					panic("gradient data index out of bounds during Embedding backward accumulation")
				}
				e.Weights.Grad.Data[embeddingGradStart+k] += grad.Data[gradVectorStart+k] // Accumulate gradient
			}

			gradFlatIndex += dimModel // Move to the next embedded vector in grad
		}
	}
}

// Backward performs the backward pass for the positional embedding layer.
// Since positional embeddings are typically not learnable, this method
// only propagates the gradient to the input if the input requires gradients.
func (pe *PositionalEmbedding) Backward(grad *Tensor) {
	if grad == nil || grad.Data == nil {
		// No gradient to propagate
		return
	}

	if pe.PositionEmbeddings != nil && pe.PositionEmbeddings.requiresGrad {

		fmt.Println("Warning: PositionalEmbedding.Backward accumulation for trainable embeddings not fully implemented.")
		// Implementation would go here if PositionEmbeddings.requiresGrad is true.
	}

}

// Parameters returns all learnable parameters of the layer.
// Positional embeddings are typically not learned, so this returns an empty slice
// unless requiresGrad is explicitly set on them.
func (pe *PositionalEmbedding) Parameters() []*Tensor {
	if pe.PositionEmbeddings != nil && pe.PositionEmbeddings.requiresGrad {
		return []*Tensor{pe.PositionEmbeddings}
	}
	return []*Tensor{}
}

// Inputs returns the input tensors of the Embedding operation.
// Assuming the input tensor is stored in the struct.
func (e *Embedding) Inputs() []*Tensor {

	return []*Tensor{} // Embedding input is not a tensor for backprop graph traversal
}

func (pe *PositionalEmbedding) Inputs() []*Tensor {

	return []*Tensor{pe.inputTensor} // Assuming inputTensor field exists
}

// Forward performs a simplified embedding lookup.
// It takes a tensor of token IDs and returns a tensor of corresponding embeddings.
func (e *Embedding) Forward(inputIDs *Tensor) (*Tensor, error) {
	// Assuming inputIDs is a 2D tensor [batch_size, sequence_length]
	// where each element is a token ID (integer).
	if len(inputIDs.Shape) != 2 {
		return nil, fmt.Errorf("embedding input must be 2D, got %v", inputIDs.Shape)
	}

	// Store input shape and token IDs for the backward pass
	e.inputShape = inputIDs.Shape
	if e.inputTokenIDs == nil || len(e.inputTokenIDs) != len(inputIDs.Data) {
		e.inputTokenIDs = make([]int, len(inputIDs.Data))
	}

	// Extract dimensions from the input tensor shape.
	// inputIDs.Shape[0] is the batch size.
	batchSize := inputIDs.Shape[0]
	seqLength := inputIDs.Shape[1]

	outputShape := []int{batchSize, seqLength, e.DimModel}
	outputData := make([]float64, batchSize*seqLength*e.DimModel)

	// Iterate over the flattened input IDs for efficiency and clarity.
	for i, idAsFloat := range inputIDs.Data {
		tokenID := int(idAsFloat)
		e.inputTokenIDs[i] = tokenID // Store for backward pass

		// Validate the token ID is within the vocabulary range.
		if tokenID < 0 || tokenID >= e.VocabSize {
			return nil, fmt.Errorf("token ID %d is out of vocabulary range [0, %d)", tokenID, e.VocabSize)
		}

		// Copy the embedding vector for the current token ID from the weights tensor.
		weightsOffset := tokenID * e.DimModel
		outputOffset := i * e.DimModel
		copy(outputData[outputOffset:outputOffset+e.DimModel], e.Weights.Data[weightsOffset:weightsOffset+e.DimModel])
	}
	return NewTensor(outputData, outputShape, true), nil
}

// NewPositionalEmbedding creates a basic positional embedding layer.
// It initializes the positional embeddings using a sinusoidal pattern.
func NewPositionalEmbedding(maxSequenceLength, dimModel int) *PositionalEmbedding {
	// Placeholder: Create a dummy position embeddings tensor
	positionEmbeddingsData := make([]float64, maxSequenceLength*dimModel)
	// Initialize with sinusoidal positional embeddings
	for pos := 0; pos < maxSequenceLength; pos++ {
		for i := 0; i < dimModel; i++ {
			if i%2 == 0 {
				positionEmbeddingsData[pos*dimModel+i] = math.Sin(float64(pos) / math.Pow(tenThousand, float64(i)/float64(dimModel)))
			} else {
				positionEmbeddingsData[pos*dimModel+i] = math.Cos(float64(pos) / math.Pow(tenThousand, float64(i-1)/float64(dimModel)))
			}
		}
	}
	positionEmbeddings := NewTensor(positionEmbeddingsData, []int{maxSequenceLength, dimModel}, false)

	return &PositionalEmbedding{
		MaxSequenceLength:  maxSequenceLength,
		DimModel:           dimModel,
		PositionEmbeddings: positionEmbeddings,
	}
}

// Forward performs a simplified positional embedding addition.
// It adds the positional embeddings to the input tensor (which is assumed
// to be token embeddings or similar).
func (pe *PositionalEmbedding) Forward(inputTensor *Tensor) (*Tensor, error) {
	pe.inputTensor = inputTensor // Add this line

	// Assuming inputTensor is [batch_size, sequence_length, embedding_dim]
	// It should be the output of a token embedding layer.
	if len(inputTensor.Shape) != 3 || inputTensor.Shape[2] != pe.DimModel {
		return nil, fmt.Errorf("positional embedding input must be 3D with embedding_dim %d, got %v", pe.DimModel, inputTensor.Shape)
	}

	// Extract dimensions from the input tensor shape.
	batchSize := inputTensor.Shape[0]
	seqLength := inputTensor.Shape[1]
	dimModel := inputTensor.Shape[2]

	outputData := make([]float64, len(inputTensor.Data))
	copy(outputData, inputTensor.Data) // Start with the input tensor data

	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLength; s++ {
			// Add a check to prevent out-of-bounds access on positional embeddings
			if s >= pe.MaxSequenceLength {
				// Depending on the model's design, you might want to log a warning,
				// return an error, or simply stop applying positional embeddings.
				// Here, we'll just continue, effectively not adding embeddings for long sequences.
				continue
			}

			outputOffset := (b*seqLength + s) * dimModel
			posEmbOffset := s * dimModel

			// Add positional embedding for position 's'
			for d := 0; d < dimModel; d++ {
				outputData[outputOffset+d] += pe.PositionEmbeddings.Data[posEmbOffset+d]
			}
		}
	}

	return NewTensor(outputData, inputTensor.Shape, false), nil
}

func init() {
	rand.Seed(time.Now().UnixNano())
}
