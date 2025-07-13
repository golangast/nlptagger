package bartsimple

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

const tenThousand = 10000.0

// Embedding represents a basic token embedding layer.
type Embedding struct {
	VocabSize int
	DimModel  int
	Weights   *Tensor // Placeholder for embedding weights
}

// NewEmbedding creates a basic embedding layer.
func NewEmbedding(vocabSize, dimModel int) *Embedding {
	weightsData := make([]float64, vocabSize*dimModel)
	// Initialize weights with small random values
	for i := range weightsData {
		weightsData[i] = (rand.Float64()*2 - 1) * 0.1 // Random values between -0.1 and 0.1
	}
	weights := NewTensor(weightsData, []int{vocabSize, dimModel})

	return &Embedding{
		VocabSize: vocabSize,
		DimModel:  dimModel,
		Weights:   weights,
	}
}

// Forward performs a simplified embedding lookup.
func (e *Embedding) Forward(inputIDs *Tensor) (*Tensor, error) {
	// Assuming inputIDs is a 2D tensor [batch_size, sequence_length]
	if len(inputIDs.Shape) != 2 {
		return nil, fmt.Errorf("embedding input must be 2D, got %v", inputIDs.Shape)
	}

	batchSize := inputIDs.Shape[0]
	seqLength := inputIDs.Shape[1]

	// Perform embedding lookup
	outputShape := []int{batchSize, seqLength, e.DimModel}
	outputData := make([]float64, batchSize*seqLength*e.DimModel)

	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLength; s++ {
			tokenID := int(inputIDs.Data[b*seqLength+s])
			if tokenID < 0 || tokenID >= e.VocabSize {
				return nil, fmt.Errorf("token ID %d is out of vocabulary range [0, %d)", tokenID, e.VocabSize)
			}
			startIndex := tokenID * e.DimModel
			endIndex := (tokenID + 1) * e.DimModel
			copy(outputData[(b*seqLength+s)*e.DimModel:(b*seqLength+s+1)*e.DimModel], e.Weights.Data[startIndex:endIndex])
		}
	}
	return NewTensor(outputData, outputShape), nil
}

// PositionalEmbedding represents a basic positional embedding layer.
type PositionalEmbedding struct {
	MaxSequenceLength  int
	DimModel           int
	PositionEmbeddings *Tensor // Placeholder for positional embeddings
}

// NewPositionalEmbedding creates a basic positional embedding layer.
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
	positionEmbeddings := NewTensor(positionEmbeddingsData, []int{maxSequenceLength, dimModel})

	return &PositionalEmbedding{
		MaxSequenceLength:  maxSequenceLength,
		DimModel:           dimModel,
		PositionEmbeddings: positionEmbeddings,
	}
}

// Forward performs a simplified positional embedding addition.
func (pe *PositionalEmbedding) Forward(inputTensor *Tensor) (*Tensor, error) {
	// Assuming inputTensor is [batch_size, sequence_length, embedding_dim]
	if len(inputTensor.Shape) != 3 || inputTensor.Shape[2] != pe.DimModel {
		return nil, fmt.Errorf("positional embedding input must be 3D with embedding_dim %d, got %v", pe.DimModel, inputTensor.Shape)
	}

	batchSize := inputTensor.Shape[0]
	seqLength := inputTensor.Shape[1]
	dimModel := inputTensor.Shape[2]

	outputData := make([]float64, len(inputTensor.Data))
	copy(outputData, inputTensor.Data) // Start with the input tensor data

	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLength; s++ {
			// Add positional embedding for position 's'
			for d := 0; d < dimModel; d++ {
				outputData[(b*seqLength+s)*dimModel+d] += pe.PositionEmbeddings.Data[s*dimModel+d]
			}
		}
	}

	return NewTensor(outputData, inputTensor.Shape), nil
}

func init() {
	rand.Seed(time.Now().UnixNano())
}
