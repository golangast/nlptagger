package nn

import (
	"encoding/gob"
	"fmt"
	"log"
	"math"
	"math/rand"
	"time"

	. "github.com/zendrulat/nlptagger/neural/tensor"
)

const tenThousand = 10000.0

// Embedding represents a simple token embedding layer.
type Embedding struct {
	RequiresGrad bool
	DimModel     int
	VocabSize    int
	Weight       *Tensor // Embedding weights

	// Stored values from forward pass for backward calculation
	inputTokenIDs []int   // Store input token IDs
	inputShape    []int   // Store input shape
	inputTensor   *Tensor // Store the input tensor for backward pass
}

// PositionalEmbedding represents a simple positional embedding layer.
type PositionalEmbedding struct {
	MaxSequenceLength  int
	DimModel           int
	PositionEmbeddings *Tensor // Positional embeddings (typically not learnable)

	// Stored values from forward pass for backward calculation
	inputTensor *Tensor // Add this field
}

// NewEmbedding creates a new Embedding layer with random initialization.
func NewEmbedding(vocabSize, dimModel int) *Embedding {
	// Initialize weights with appropriate shape [vocabSize, dimModel]
	weightsShape := []int{vocabSize, dimModel}
	weightsData := make([]float64, vocabSize*dimModel)
	for i := range weightsData {
		weightsData[i] = (rand.Float64()*2 - 1) * 0.1 // Random between -0.1 and 0.1
	}
	weights := NewTensor(weightsShape, weightsData, true)
	weights.RequiresGrad = true // Embedding weights require gradients

	return &Embedding{
		Weight:    weights,
		DimModel:  dimModel,  // Initialize the DimModel field
		VocabSize: vocabSize, // Initialize the VocabSize field
	}
}

// LoadPretrainedWeights loads pretrained embedding weights.
func (e *Embedding) LoadPretrainedWeights(weights map[int][]float64) {
	for tokenID, vector := range weights {
		if tokenID >= e.VocabSize {
			continue // Or handle error
		}
		offset := tokenID * e.DimModel
		if len(vector) != e.DimModel {
			continue // Or handle error
		}
		copy(e.Weight.Data[offset:offset+e.DimModel], vector)
	}
}

// Parameters returns all learnable parameters of the layer.
func (e *Embedding) Parameters() []*Tensor {
	return []*Tensor{e.Weight}
}

// SetInput sets the input tensor for the backward pass.
// This is useful when the embedding layer is reused in a loop (e.g. RNN).
func (e *Embedding) SetInput(input *Tensor) {
	e.inputShape = input.Shape
	e.inputTensor = input
	if e.inputTokenIDs == nil || len(e.inputTokenIDs) != len(input.Data) {
		e.inputTokenIDs = make([]int, len(input.Data))
	}
	for i, v := range input.Data {
		e.inputTokenIDs[i] = int(v)
	}
}

// Backward performs the backward pass for the token embedding layer.
// grad is the gradient from the subsequent layer.
func (e *Embedding) Backward(grad *Tensor) error {
	if grad == nil || grad.Data == nil {
		log.Printf("Embedding.Backward: Incoming grad is nil or has no data. Returning.\n")
		// No gradient to propagate
		return nil
	}

	if e.Weight == nil || !e.Weight.RequiresGrad {
		log.Printf("Embedding.Backward: Embedding weights do not require gradients. Returning.\n")
		// Embedding weights are not learnable or do not require gradients
		return nil
	}
	if e.inputTokenIDs == nil || len(e.inputTokenIDs) != len(grad.Data)/e.Weight.Shape[1] {
		return fmt.Errorf("Embedding backward called before forward or with mismatched input/gradient shapes")
	}

	vocabSize := e.Weight.Shape[0]
	dimModel := e.Weight.Shape[1]
	batchSize := e.inputShape[0]
	seqLength := e.inputShape[1]

	// Ensure gradient for weights is initialized
	if e.Weight.Grad == nil {
		log.Printf("Embedding.Backward: Initializing e.Weight.Grad.\n")
		e.Weight.Grad = NewTensor(e.Weight.Shape, make([]float64, len(e.Weight.Data)), false)
		e.Weight.Grad.RequiresGrad = false
	}

	// Accumulate gradients to the embedding weights.
	// For each gradient element in 'grad', add it to the corresponding
	// position in e.Weight.Grad based on the input token ID.
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
				log.Printf("Warning: Invalid token ID %d encountered during Embedding backward.\n", tokenID)
				continue // Skip invalid token IDs
			}

			// Add the gradient for this embedded vector (grad[i, j, :])
			// to the gradient of the corresponding embedding vector in Weight.Grad (Weight.Grad[tokenID, :]).
			embeddingGradStart := tokenID * dimModel
			gradVectorStart := gradFlatIndex // Current position in flattened grad data

			for k := 0; k < dimModel; k++ {
				if embeddingGradStart+k >= len(e.Weight.Grad.Data) || gradVectorStart+k >= len(grad.Data) {
					return fmt.Errorf("gradient data index out of bounds during Embedding backward accumulation")
				}
				e.Weight.Grad.Data[embeddingGradStart+k] += grad.Data[gradVectorStart+k] // Accumulate gradient
			}

			gradFlatIndex += dimModel // Move to the next embedded vector in grad
		}
	}
	return nil
}

// Backward performs the backward pass for the positional embedding layer.
// Since positional embeddings are typically not learnable, this method
// only propagates the gradient to the input if the input requires gradients.
func (pe *PositionalEmbedding) Backward(grad *Tensor) error {
	if grad == nil || grad.Data == nil {
		return nil
	}

	if pe.inputTensor != nil && pe.inputTensor.RequiresGrad {
		if pe.inputTensor.Grad == nil {
			pe.inputTensor.Grad = NewTensor(pe.inputTensor.Shape, make([]float64, len(pe.inputTensor.Data)), false)
		}
		for i := range grad.Data {
			pe.inputTensor.Grad.Data[i] += grad.Data[i]
		}
	}

	return nil
}

// Parameters returns all learnable parameters of the layer.
// Positional embeddings are typically not learned, so this returns an empty slice
// unless RequiresGrad is explicitly set on them.
func (pe *PositionalEmbedding) Parameters() []*Tensor {
	if pe.PositionEmbeddings != nil && pe.PositionEmbeddings.RequiresGrad {
		return []*Tensor{pe.PositionEmbeddings}
	}
	return []*Tensor{}
}

// Inputs returns the input tensors of the Embedding operation.
// Assuming the input tensor is stored in the struct.
func (e *Embedding) Inputs() []*Tensor {
	if e.inputTensor != nil {
		return []*Tensor{e.inputTensor}
	}
	return []*Tensor{} // Embedding input is not a tensor for backprop graph traversal
}

func (pe *PositionalEmbedding) Inputs() []*Tensor {
	return []*Tensor{pe.inputTensor}
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
	e.inputTensor = inputIDs // Store the input tensor
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
			log.Printf("Error: token ID %d is out of vocabulary range [0, %d)", tokenID, e.VocabSize)
			return nil, fmt.Errorf("token ID %d is out of vocabulary range [0, %d)", tokenID, e.VocabSize)
		}

		// Copy the embedding vector for the current token ID from the weights tensor.
		weightsOffset := tokenID * e.DimModel
		outputOffset := i * e.DimModel
		copy(outputData[outputOffset:outputOffset+e.DimModel], e.Weight.Data[weightsOffset:weightsOffset+e.DimModel])
	}
	outputTensor := NewTensor(outputShape, outputData, e.Weight.RequiresGrad)
	if outputTensor.RequiresGrad {
		outputTensor.Creator = e
	}
	return outputTensor, nil
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
	positionEmbeddings := NewTensor([]int{maxSequenceLength, dimModel}, positionEmbeddingsData, false)
	positionEmbeddings.RequiresGrad = false

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

	outputTensor := NewTensor(inputTensor.Shape, outputData, inputTensor.RequiresGrad)
	if outputTensor.RequiresGrad {
		outputTensor.Creator = pe
	}
	return outputTensor, nil
}

func init() {
	gob.Register(&Embedding{})
	rand.Seed(time.Now().UnixNano())
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
