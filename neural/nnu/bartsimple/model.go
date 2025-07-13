package bartsimple

import (
	"encoding/gob"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
)

// BARTEncoder represents a simplified encoder.
type BARTEncoder struct {
	Layer *BARTEncoderLayer // Simplified: only one layer
}

// BARTDecoder represents a simplified decoder.
type BARTDecoder struct {
	Layer *BARTDecoderLayer // Simplified: only one layer
}

// BARTEncoderLayer represents a single simplified encoder layer.
type BARTEncoderLayer struct {
	SelfAttention *MultiHeadAttention
	FeedForward   *FeedForward
	Norm1, Norm2  *LayerNormalization
}

// NewBARTEncoderLayer creates a new simplified encoder layer.
func NewBARTEncoderLayer(dimModel, numHeads int) (*BARTEncoderLayer, error) {
	selfAttention, err := NewMultiHeadAttention(dimModel, numHeads, numHeads) // Assuming numQHeads == numKVHeads
	if err != nil {
		return nil, fmt.Errorf("failed to create encoder self-attention: %w", err)
	}
	feedForward, err := NewFeedForward(dimModel)
	if err != nil {
		return nil, fmt.Errorf("failed to create encoder feed-forward: %w", err)
	}
	norm1 := NewLayerNormalization(dimModel)
	norm2 := NewLayerNormalization(dimModel)

	return &BARTEncoderLayer{
		SelfAttention: selfAttention,
		FeedForward:   feedForward,
		Norm1:         norm1,
		Norm2:         norm2,
	}, nil
}

// Forward performs the forward pass of the simplified encoder layer.
func (l *BARTEncoderLayer) Forward(inputTensor *Tensor, mask *Tensor) (*Tensor, error) {
	// Self-Attention
	attentionOutput, err := l.SelfAttention.Forward(inputTensor, inputTensor, inputTensor, mask) // Q, K, V are the same for self-attention
	if err != nil {
		return nil, fmt.Errorf("encoder self-attention failed: %w", err)
	}

	// Add and Normalize
	norm1Input, err := inputTensor.AddWithBroadcast(attentionOutput) // Residual connection
	if err != nil {
		return nil, fmt.Errorf("encoder self-attention residual failed: %w", err)
	}
	norm1Output, err := l.Norm1.Forward(norm1Input) // Layer Normalization
	if err != nil {
		return nil, fmt.Errorf("encoder self-attention normalization failed: %w", err)
	}

	// Feed-Forward
	feedForwardOutput, err := l.FeedForward.Forward(norm1Output)
	if err != nil {
		return nil, fmt.Errorf("encoder feed-forward failed: %w", err)
	}

	// Add and Normalize
	norm2Input, err := norm1Output.AddWithBroadcast(feedForwardOutput) // Residual connection
	if err != nil {
		return nil, fmt.Errorf("encoder feed-forward residual failed: %w", err)
	}
	norm2Output, err := l.Norm2.Forward(norm2Input) // Layer Normalization
	if err != nil {
		return nil, fmt.Errorf("encoder feed-forward normalization failed: %w", err)
	}

	return norm2Output, nil
}

// BARTDecoderLayer represents a single simplified decoder layer.
type BARTDecoderLayer struct {
	SelfAttention       *MultiHeadAttention
	CrossAttention      *MultiHeadCrossAttention
	FeedForward         *FeedForward
	Norm1, Norm2, Norm3 *LayerNormalization
}

// NewBARTDecoderLayer creates a new simplified decoder layer.
func NewBARTDecoderLayer(dimModel, numHeads int) (*BARTDecoderLayer, error) {
	selfAttention, err := NewMultiHeadAttention(dimModel, numHeads, numHeads) // Assuming numQHeads == numKVHeads
	if err != nil {
		return nil, fmt.Errorf("failed to create decoder self-attention: %w", err)
	}
	crossAttention, err := NewMultiHeadCrossAttention(dimModel, numHeads, numHeads) // Assuming numQHeads == numKVHeads
	if err != nil {
		return nil, fmt.Errorf("failed to create decoder cross-attention: %w", err)
	}
	feedForward, err := NewFeedForward(dimModel)
	if err != nil {
		return nil, fmt.Errorf("failed to create decoder feed-forward: %w", err)
	}
	norm1 := NewLayerNormalization(dimModel)
	norm2 := NewLayerNormalization(dimModel)
	norm3 := NewLayerNormalization(dimModel)

	return &BARTDecoderLayer{
		SelfAttention:  selfAttention,
		CrossAttention: crossAttention,
		FeedForward:    feedForward,
		Norm1:          norm1,
		Norm2:          norm2,
		Norm3:          norm3,
	}, nil
}

// Forward performs the forward pass of the simplified decoder layer.
func (l *BARTDecoderLayer) Forward(inputTensor *Tensor, encoderOutput *Tensor, selfAttentionMask *Tensor, crossAttentionMask *Tensor) (*Tensor, error) {
	// Self-Attention
	selfAttentionOutput, err := l.SelfAttention.Forward(inputTensor, inputTensor, inputTensor, selfAttentionMask) // Q, K, V are the same for self-attention
	if err != nil {
		return nil, fmt.Errorf("decoder self-attention failed: %w", err)
	}

	// Add and Normalize
	norm1Input, err := inputTensor.AddWithBroadcast(selfAttentionOutput) // Residual connection
	if err != nil {
		return nil, fmt.Errorf("decoder self-attention residual failed: %w", err)
	}
	norm1Output, err := l.Norm1.Forward(norm1Input) // Layer Normalization
	if err != nil {
		return nil, fmt.Errorf("decoder self-attention normalization failed: %w", err)
	}

	// Cross-Attention
	crossAttentionOutput, err := l.CrossAttention.Forward(norm1Output, encoderOutput, encoderOutput, crossAttentionMask) // Query is normalized self-attention output, K/V are encoder output
	if err != nil {
		return nil, fmt.Errorf("decoder cross-attention failed: %w", err)
	}

	// Add and Normalize
	norm2Input, err := norm1Output.AddWithBroadcast(crossAttentionOutput) // Residual connection
	if err != nil {
		return nil, fmt.Errorf("decoder cross-attention residual failed: %w", err)
	}
	norm2Output, err := l.Norm2.Forward(norm2Input) // Layer Normalization
	if err != nil {
		return nil, fmt.Errorf("decoder cross-attention normalization failed: %w", err)
	}

	// Feed-Forward
	feedForwardOutput, err := l.FeedForward.Forward(norm2Output)
	if err != nil {
		return nil, fmt.Errorf("decoder feed-forward failed: %w", err)
	}

	// Add and Normalize
	norm3Input, err := norm2Output.AddWithBroadcast(feedForwardOutput) // Residual connection
	if err != nil {
		return nil, fmt.Errorf("decoder feed-forward residual failed: %w", err)
	}

	norm3Output, err := l.Norm3.Forward(norm3Input) // Layer Normalization
	if err != nil {
		return nil, fmt.Errorf("decoder final normalization failed: %w", err)
	}

	return norm3Output, nil
}

// SimplifiedBARTModel represents a simplified BART model with one encoder and one decoder layer.
type SimplifiedBARTModel struct {
	Encoder             *BARTEncoder
	Decoder             *BARTDecoder
	Tokenizer           *Tokenizer           // We'll add a simple one later
	TokenEmbedding      *Embedding           // We'll add a simple one later
	PositionalEmbedding *PositionalEmbedding // We'll add a simple one later
	OutputLinear        *Linear
	TokenIds            []int // Final linear layer for output
	VocabSize           int
	MaxSequenceLength   int
	Vocabulary          *Vocabulary
}

// NewSimplifiedBARTModel creates a new simplified BART model.
func NewSimplifiedBARTModel(tokenizer *Tokenizer, vocabulary *Vocabulary, dimModel, numHeads, maxSequenceLength int) (*SimplifiedBARTModel, error) {
	vocabSize := len(vocabulary.WordToToken)

	// Create simplified encoder and decoder with one layer each
	encoderLayer, err := NewBARTEncoderLayer(dimModel, numHeads)
	if err != nil {
		return nil, fmt.Errorf("failed to create simplified encoder layer: %w", err)
	}
	encoder := &BARTEncoder{Layer: encoderLayer}

	decoderLayer, err := NewBARTDecoderLayer(dimModel, numHeads)
	if err != nil {
		return nil, fmt.Errorf("failed to create simplified decoder layer: %w", err)
	}
	decoder := &BARTDecoder{Layer: decoderLayer}

	// Initialize embedding layers and output linear layer (simplified)
	// These will need to be implemented in separate simple files
	tokenEmbedding := NewEmbedding(len(vocabulary.WordToToken), dimModel)

	// Assuming NewEmbedding exists
	positionalEmbedding := NewPositionalEmbedding(maxSequenceLength, dimModel) // Assuming NewPositionalEmbedding exists
	outputLinear, err := NewLinear(dimModel, vocabSize)                        // Assuming NewLinear exists
	if err != nil {
		return nil, fmt.Errorf("failed to create output linear layer: %w", err)
	}

	return &SimplifiedBARTModel{
		Encoder:             encoder,
		Decoder:             decoder,
		TokenEmbedding:      tokenEmbedding,
		PositionalEmbedding: positionalEmbedding,
		OutputLinear:        outputLinear,
		VocabSize:           vocabSize,
		MaxSequenceLength:   maxSequenceLength,
	}, nil
}
func Reshape(tensor []float64, originalShape, newShape []int) ([]float64, error) {
	originalSize := 1
	for _, dim := range originalShape {
		originalSize *= dim
	}

	newSize := 1
	for _, dim := range newShape {
		newSize *= dim
	}

	if originalSize != newSize {
		return nil, fmt.Errorf("cannot reshape tensor from %v to %v: sizes do not match (%d vs %d)", originalShape, newShape, originalSize, newSize)
	}

	// In a simple case where the underlying data layout is compatible,
	// you might just return the original slice with the new shape metadata.
	// However, for more complex reshapes (e.g., transposing), you might
	// need to create a new slice and copy/rearrange the data.
	// This basic implementation just checks for size compatibility.
	// A more complete tensor library would handle the data rearrangement.

	// If your tensor representation is a struct containing the data slice and shape:
	// func (t *Tensor) Reshape(newShape []int) (*Tensor, error) {
	//     // ... size check ...
	//     return &Tensor{Data: t.Data, Shape: newShape}, nil // Or create a new Tensor with reordered data
	// }

	// Assuming you are working with a simple float64 slice:
	return tensor, nil // This is a placeholder, you might need to reorder data
}

// SaveSimplifiedBARTModelToGOB saves the simplified BART model to a file in Gob format.
func SaveSimplifiedBARTModelToGOB(model *SimplifiedBARTModel, filePath string) error {
	// Create the directory if it doesn't exist
	dir := filepath.Dir(filePath)                  // Need to import "path/filepath"
	if err := os.MkdirAll(dir, 0755); err != nil { // Need to import "os"
		return fmt.Errorf("failed to create directory %s: %w", dir, err)
	}

	// 1. Check if the file exists
	_, err := os.Stat(filePath)
	if err == nil {
		// File exists, proceed to delete
		fmt.Printf("File '%s' exists. Deleting...\n", filePath)
		err = os.Remove(filePath)
		if err != nil {
			log.Fatalf("Error deleting file '%s': %v\n", filePath, err)
		}
		fmt.Printf("File '%s' deleted successfully.\n", filePath)
	} else if !errors.Is(err, os.ErrNotExist) {
		// Handle other potential errors during Stat
		log.Fatalf("Error checking file existence for '%s': %v\n", filePath, err)
	} else {
		fmt.Printf("File '%s' does not exist. Creating a new one.\n", filePath)
	}

	// 2. Create a new file (or truncate and open if it existed)
	file, err := os.Create(filePath) // Need to import "os"
	if err != nil {
		return fmt.Errorf("failed to create file for saving simplified model: %w", err)
	}
	defer file.Close()

	fmt.Printf("File '%s' created/replaced and written successfully.\n", filePath)

	// Register all custom types used in your simplified model
	// This is crucial for gob encoding/decoding.
	gob.Register(&SimplifiedBARTModel{}) // Need to import "encoding/gob"
	gob.Register(&BARTEncoder{})
	gob.Register(&BARTDecoder{})
	gob.Register(&BARTEncoderLayer{})
	gob.Register(&BARTDecoderLayer{})
	gob.Register(&MultiHeadAttention{})
	gob.Register(&MultiHeadCrossAttention{})
	gob.Register(&Linear{})
	gob.Register(&Embedding{})
	gob.Register(&PositionalEmbedding{})
	gob.Register(&FeedForward{})
	gob.Register(&LayerNormalization{})
	gob.Register(&Tensor{}) // Make sure Tensor is registered

	encoder := gob.NewEncoder(file)
	err = encoder.Encode(model)
	if err != nil {
		return fmt.Errorf("failed to encode simplified BART model to Gob: %w", err)
	}

	return nil
}

// LoadSimplifiedBARTModelFromGOB loads a simplified BART model from a file in Gob format.
func LoadSimplifiedBARTModelFromGOB(filePath string) (*SimplifiedBARTModel, error) {
	file, err := os.Open(filePath) // Need to import "os"
	if err != nil {
		// If the file doesn't exist, return a specific error
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("simplified BART model gob file not found at %s", filePath)
		}
		// For other errors, return the original error
		return nil, fmt.Errorf("error opening simplified BART model gob file: %w", err)
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)

	// Register all custom types used in your simplified model (same as in Save)
	gob.Register(&SimplifiedBARTModel{})
	gob.Register(&BARTEncoder{})
	gob.Register(&BARTDecoder{})
	gob.Register(&BARTEncoderLayer{})
	gob.Register(&BARTDecoderLayer{})
	gob.Register(&MultiHeadAttention{})
	gob.Register(&MultiHeadCrossAttention{})
	gob.Register(&Linear{})
	gob.Register(&Embedding{})
	gob.Register(&PositionalEmbedding{})
	gob.Register(&FeedForward{})
	gob.Register(&LayerNormalization{})
	gob.Register(&Tensor{})

	var loadedModel SimplifiedBARTModel
	err = decoder.Decode(&loadedModel)
	if err != nil {
		return nil, fmt.Errorf("error decoding simplified BART model from gob: %w", err)
	}
	// Debugging: Inspect loaded Embedding weights
	if loadedModel.TokenEmbedding == nil {
		fmt.Println("Debug Load: loadedModel.TokenEmbedding is nil after decoding!")
	} else {
		if loadedModel.TokenEmbedding.Weights == nil {
			fmt.Println("Debug Load: loadedModel.TokenEmbedding.Weights is nil after decoding!")
		}
	}

	return &loadedModel, nil
}
