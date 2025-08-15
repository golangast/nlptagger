package bartsimple

import (
	"encoding/gob"
	"errors"
	"fmt"
	"math/rand"
	"os"
	"sort"

	"github.com/golangast/nlptagger/tagger"
	"github.com/golangast/nlptagger/tagger/nertagger"
	"github.com/golangast/nlptagger/tagger/postagger"
)

// BARTEncoder represents a simplified encoder.
type BARTEncoder struct {
	Layer *BARTEncoderLayer // Simplified: only one layer
}

// BARTDecoder represents a simplified decoder.
type BARTDecoder struct {
	Layer *BARTDecoderLayer // Simplified: only one layer
}

// Parameters returns all learnable parameters of the layer.

// Parameters returns all learnable parameters of the layer.
func (l *BARTEncoderLayer) Parameters() []*Tensor {
	params := l.SelfAttention.Parameters()
	params = append(params, l.FeedForward.Parameters()...)
	params = append(params, l.Norm1.Parameters()...)
	params = append(params, l.Norm2.Parameters()...)
	return params
}

// Parameters returns all learnable parameters of the entire model.
func (m *SimplifiedBARTModel) Parameters() []*Tensor {
	params := m.TokenEmbedding.Parameters()
	params = append(params, m.PositionalEmbedding.Parameters()...)
	params = append(params, m.PosTagEmbedding.Parameters()...)
	params = append(params, m.NerTagEmbedding.Parameters()...)
	params = append(params, m.Encoder.Layer.Parameters()...)
	params = append(params, m.Decoder.Layer.Parameters()...)
	params = append(params, m.OutputLinear.Parameters()...)
	return params
}

// Parameters returns all learnable parameters of the layer.
func (l *BARTDecoderLayer) Parameters() []*Tensor {
	params := l.SelfAttention.Parameters()
	params = append(params, l.CrossAttention.Parameters()...)
	params = append(params, l.FeedForward.Parameters()...)
	params = append(params, l.Norm1.Parameters()...)
	params = append(params, l.Norm2.Parameters()...)
	params = append(params, l.Norm3.Parameters()...)
	return params
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
	},
		nil
}

// Forward performs the forward pass of the simplified encoder layer.
func (l *BARTEncoderLayer) Forward(inputTensor *Tensor, mask *Tensor) (*Tensor, error) {
	// Self-Attention
	inputTensor.Mask = mask
	attentionOutput, err := l.SelfAttention.Forward(inputTensor) // Q, K, V are the same for self-attention
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
	},
		nil
}

// Forward performs the forward pass of the simplified decoder layer.
func (l *BARTDecoderLayer) Forward(inputTensor *Tensor, encoderOutput *Tensor, selfAttentionMask *Tensor, crossAttentionMask *Tensor) (*Tensor, error) {
	// Self-Attention
	inputTensor.Mask = selfAttentionMask
	selfAttentionOutput, err := l.SelfAttention.Forward(inputTensor) // Q, K, V are the same for self-attention
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
	Tokenizer           *Tokenizer
	TokenEmbedding      *Embedding
	PositionalEmbedding *PositionalEmbedding
	PosTagEmbedding     *Embedding
	NerTagEmbedding     *Embedding
	OutputLinear        *Linear
	TokenIds            []int
	VocabSize           int
	MaxSequenceLength   int
	Vocabulary          *Vocabulary
}

// NewSimplifiedBARTModel creates a new simplified BART model.
func NewSimplifiedBARTModel(tokenizer *Tokenizer, vocabulary *Vocabulary, dimModel, numHeads, maxSequenceLength int, word2VecEmbeddings map[string][]float64) (*SimplifiedBARTModel, error) {
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
	tokenEmbedding := NewEmbeddingWithPretrained(len(vocabulary.WordToToken), dimModel, vocabulary, word2VecEmbeddings)
	posTagEmbedding := NewEmbedding(len(postagger.PosTagToIDMap()), dimModel)
	nerTagEmbedding := NewEmbedding(len(nertagger.NerTagToIDMap()), dimModel)
	positionalEmbedding := NewPositionalEmbedding(maxSequenceLength, dimModel)
	outputLinear, err := NewLinear(dimModel, vocabSize)
	if err != nil {
		return nil, fmt.Errorf("failed to create output linear layer: %w", err)
	}

	return &SimplifiedBARTModel{
		Encoder:             encoder,
		Decoder:             decoder,
		Tokenizer:           tokenizer,
		TokenEmbedding:      tokenEmbedding,
		PositionalEmbedding: positionalEmbedding,
		PosTagEmbedding:     posTagEmbedding,
		NerTagEmbedding:     nerTagEmbedding,
		OutputLinear:        outputLinear,
		VocabSize:           vocabSize,
		MaxSequenceLength:   maxSequenceLength,
		Vocabulary:          vocabulary,
	},
		nil
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

	return tensor, nil
}

// SaveSimplifiedBARTModelToGOB saves the simplified BART model to a file in Gob format.
func SaveSimplifiedBARTModelToGOB(model *SimplifiedBARTModel, filePath string) error {
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to create file for saving simplified model: %w", err)
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	err = encoder.Encode(model)
	if err != nil {
		return fmt.Errorf("failed to encode simplified BART model to Gob: %w", err)
	}

	return nil
}

// LoadSimplifiedBARTModelFromGOB loads a simplified BART model from a file in Gob format.
func LoadSimplifiedBARTModelFromGOB(filePath string) (*SimplifiedBARTModel, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("error opening simplified BART model gob file: %w", err)
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	var loadedModel SimplifiedBARTModel
	err = decoder.Decode(&loadedModel)
	if err != nil {
		return nil, fmt.Errorf("error decoding simplified BART model from gob: %w", err)
	}
	return &loadedModel, nil
}

// ForwardForTraining performs the forward pass of the BART model for training.
func (m *SimplifiedBARTModel) ForwardForTraining(inputTensor, targetTensor, posTagTensor, nerTagTensor *Tensor) (*Tensor, error) {
	// Ensure necessary components are initialized
	if m.Encoder == nil || m.Decoder == nil || m.TokenEmbedding == nil || m.PositionalEmbedding == nil || m.PosTagEmbedding == nil || m.NerTagEmbedding == nil || m.OutputLinear == nil {
		return nil, errors.New("model component is not initialized")
	}

	// --- Encoder Forward Pass ---
	encoderInputEmbeddings, err := m.TokenEmbedding.Forward(inputTensor)
	if err != nil {
		return nil, fmt.Errorf("encoder token embedding failed: %w", err)
	}

	posTagEmbeddings, err := m.PosTagEmbedding.Forward(posTagTensor)
	if err != nil {
		return nil, fmt.Errorf("encoder pos tag embedding failed: %w", err)
	}

	nerTagEmbeddings, err := m.NerTagEmbedding.Forward(nerTagTensor)
	if err != nil {
		return nil, fmt.Errorf("encoder ner tag embedding failed: %w", err)
	}

	encoderInputEmbeddings, err = encoderInputEmbeddings.Add(posTagEmbeddings)
	if err != nil {
		return nil, fmt.Errorf("failed to add pos tag embeddings: %w", err)
	}

	encoderInputEmbeddings, err = encoderInputEmbeddings.Add(nerTagEmbeddings)
	if err != nil {
		return nil, fmt.Errorf("failed to add ner tag embeddings: %w", err)
	}

	encoderInputWithPos, err := m.PositionalEmbedding.Forward(encoderInputEmbeddings)
	if err != nil {
		return nil, fmt.Errorf("encoder positional embedding failed: %w", err)
	}

	var encoderMask *Tensor = nil
	encoderOutput, err := m.Encoder.Layer.Forward(encoderInputWithPos, encoderMask)
	if err != nil {
		return nil, fmt.Errorf("encoder forward pass failed: %w", err)
	}

	// --- Decoder Forward Pass ---
	decoderInputEmbeddings, err := m.TokenEmbedding.Forward(targetTensor)
	if err != nil {
		return nil, fmt.Errorf("decoder token embedding failed: %w", err)
	}

	decoderInputWithPos, err := m.PositionalEmbedding.Forward(decoderInputEmbeddings)
	if err != nil {
		return nil, fmt.Errorf("decoder positional embedding failed: %w", err)
	}

	var selfAttentionMask *Tensor = nil
	var crossAttentionMask *Tensor = nil
	decoderOutput, err := m.Decoder.Layer.Forward(decoderInputWithPos, encoderOutput, selfAttentionMask, crossAttentionMask)
	if err != nil {
		return nil, fmt.Errorf("decoder forward pass failed: %w", err)
	}

	// --- Final Linear Layer ---
	outputLogits, err := m.OutputLinear.Forward(decoderOutput)
	if err != nil {
		return nil, fmt.Errorf("output linear layer failed: %w", err)
	}

	return outputLogits, nil
}

// Reply generates a response based on the input text.
func (m *SimplifiedBARTModel) Reply(inputText string) (string, error) {
	// 1. Tag the input text
	taggedInput := tagger.Tagging(inputText)

	// 2. Filter the tags
	filteredTokens := []string{}
	filteredPosTags := []string{}
	filteredNerTags := []string{}
	filterTags := map[string]bool{
		"DT": true, "PRP": true, "IN": true, "CC": true, "RP": true, "ADP": true, "WDT": true, "DET": true, "WP": true,
	}
	for i, token := range taggedInput.Tokens {
		if !filterTags[taggedInput.PosTag[i]] {
			filteredTokens = append(filteredTokens, token)
			filteredPosTags = append(filteredPosTags, taggedInput.PosTag[i])
			filteredNerTags = append(filteredNerTags, taggedInput.NerTag[i])
		}
	}

	// 3. Encode the input text from the filtered tokens
	inputTokenIDs := make([]int, len(filteredTokens))
	for i, token := range filteredTokens {
		if id, ok := m.Vocabulary.WordToToken[token]; ok {
			inputTokenIDs[i] = id
		} else {
			inputTokenIDs[i] = m.Vocabulary.UnknownTokenID
		}
	}

	// Convert inputTokenIDs to a Tensor
	inputData := make([]float64, len(inputTokenIDs))
	for i, id := range inputTokenIDs {
		inputData[i] = float64(id)
	}
	inputTensor := NewTensor(inputData, []int{1, len(inputTokenIDs)}, false)

	// Create POS and NER tag tensors
	posTagIDs := make([]float64, len(filteredPosTags))
	for i, tag := range filteredPosTags {
		posTagIDs[i] = float64(postagger.PosTagToIDMap()[tag])
	}
	posTagTensor := NewTensor(posTagIDs, []int{1, len(posTagIDs)}, false)

	nerTagIDs := make([]float64, len(filteredNerTags))
	for i, tag := range filteredNerTags {
		nerTagIDs[i] = float64(nertagger.NerTagToIDMap()[tag])
	}
	nerTagTensor := NewTensor(nerTagIDs, []int{1, len(nerTagIDs)}, false)

	// 4. Pass through the encoder
	encoderInputEmbeddings, err := m.TokenEmbedding.Forward(inputTensor)
	if err != nil {
		return "", fmt.Errorf("encoder token embedding failed: %w", err)
	}

	posTagEmbeddings, err := m.PosTagEmbedding.Forward(posTagTensor)
	if err != nil {
		return "", fmt.Errorf("encoder pos tag embedding failed: %w", err)
	}

	nerTagEmbeddings, err := m.NerTagEmbedding.Forward(nerTagTensor)
	if err != nil {
		return "", fmt.Errorf("encoder ner tag embedding failed: %w", err)
	}

	encoderInputEmbeddings, err = encoderInputEmbeddings.Add(posTagEmbeddings)
	if err != nil {
		return "", fmt.Errorf("failed to add pos tag embeddings: %w", err)
	}

	encoderInputEmbeddings, err = encoderInputEmbeddings.Add(nerTagEmbeddings)
	if err != nil {
		return "", fmt.Errorf("failed to add ner tag embeddings: %w", err)
	}

	encoderInputWithPos, err := m.PositionalEmbedding.Forward(encoderInputEmbeddings)
	if err != nil {
		return "", fmt.Errorf("encoder positional embedding failed: %w", err)
	}
	encoderOutput, err := m.Encoder.Layer.Forward(encoderInputWithPos, nil) // No mask for now
	if err != nil {
		return "", fmt.Errorf("encoder forward pass failed: %w", err)
	}

	// 5. Initialize decoder input with BOS token
	decoderInputIDs := []int{m.Tokenizer.BosID}

	// 6. Iteratively generate tokens
	for i := 0; i < m.MaxSequenceLength; i++ {
		decoderInputData := make([]float64, len(decoderInputIDs))
		for j, id := range decoderInputIDs {
			decoderInputData[j] = float64(id)
		}
		decoderInputTensor := NewTensor(decoderInputData, []int{1, len(decoderInputIDs)}, false)

		decoderInputEmbeddings, err := m.TokenEmbedding.Forward(decoderInputTensor)
		if err != nil {
			return "", fmt.Errorf("decoder token embedding failed: %w", err)
		}
		decoderInputWithPos, err := m.PositionalEmbedding.Forward(decoderInputEmbeddings)
		if err != nil {
			return "", fmt.Errorf("decoder positional embedding failed: %w", err)
		}

		// Create causal mask for self-attention in decoder
		seqLen := len(decoderInputIDs)
		maskData := make([]float64, seqLen*seqLen)
		for r := 0; r < seqLen; r++ {
			for c := 0; c < seqLen; c++ {
				if c > r {
					maskData[r*seqLen+c] = -1e9 // Large negative value for masking
				}
			}
		}
		causalMask := NewTensor(maskData, []int{1, 1, seqLen, seqLen}, false) // Shape for broadcasting

		decoderOutput, err := m.Decoder.Layer.Forward(decoderInputWithPos, encoderOutput, causalMask, nil)
		if err != nil {
			return "", fmt.Errorf("decoder forward pass failed: %w", err)
		}

		// Get the last token's output for prediction
		lastTokenOutput, err := decoderOutput.Slice(1, len(decoderInputIDs)-1, len(decoderInputIDs))
		if err != nil {
			return "", fmt.Errorf("slicing tensor failed: %w", err)
		}
		logits, err := m.OutputLinear.Forward(lastTokenOutput)
		if err != nil {
			return "", fmt.Errorf("output linear layer failed: %w", err)
		}

		// Apply softmax and sample the next token using top-k sampling
		probabilities, err := logits.Softmax(2)
		if err != nil {
			return "", fmt.Errorf("softmax failed: %w", err)
		}

		// Prevent BOS token from being sampled after the first token
		bosTokenID := m.Tokenizer.BosID
		if bosTokenID >= 0 && bosTokenID < len(probabilities.Data) {
			probabilities.Data[bosTokenID] = 1e-10 // Set probability to a very small number
		}

		// Apply no-repeat n-gram penalty
		if len(decoderInputIDs) > 1 {
			// Penalize 2-grams
			lastToken := decoderInputIDs[len(decoderInputIDs)-1]
			for i := 0; i < len(decoderInputIDs)-1; i++ {
				if decoderInputIDs[i] == lastToken {
					probabilities.Data[decoderInputIDs[i+1]] *= 0.1
				}
			}
		}
		if len(decoderInputIDs) > 2 {
			// Penalize 3-grams
			lastTwoTokens := [2]int{decoderInputIDs[len(decoderInputIDs)-2], decoderInputIDs[len(decoderInputIDs)-1]}
			for i := 0; i < len(decoderInputIDs)-2; i++ {
				if decoderInputIDs[i] == lastTwoTokens[0] && decoderInputIDs[i+1] == lastTwoTokens[1] {
					probabilities.Data[decoderInputIDs[i+2]] *= 0.1
				}
			}
		}

		k := 10 // Increased top-k value for more diversity
		predictedTokenID, err := TopKSampling(probabilities, k, m.Vocabulary)
		if err != nil {
			return "", fmt.Errorf("top-k sampling failed: %w", err)
		}

		// Stop if EOS token is predicted
		if predictedTokenID == m.Tokenizer.EosID {
			break
		}

		decoderInputIDs = append(decoderInputIDs, predictedTokenID)
	}

	// Decode the generated token IDs into a string
	generatedSentence, err := m.Tokenizer.Decode(decoderInputIDs)
	if err != nil {
		return "", fmt.Errorf("failed to decode generated tokens: %w", err)
	}

	return generatedSentence, nil
}

// TopKSampling performs top-k sampling on a probability distribution.
type tokenProb struct {
	ID    int
	Prob  float64
	Token string
}

func TopKSampling(probabilities *Tensor, k int, vocabulary *Vocabulary) (int, error) {
	if probabilities == nil || len(probabilities.Data) == 0 {
		return 0, fmt.Errorf("probabilities tensor is empty or nil")
	}
	if k <= 0 {
		return 0, fmt.Errorf("k must be positive")
	}

	// Create a slice of tokenProb structs
	probs := make([]tokenProb, len(probabilities.Data))
	for i, p := range probabilities.Data {
		token, ok := vocabulary.TokenToWord[i]
		if !ok {
			token = "[UNK]"
		}
		probs[i] = tokenProb{ID: i, Prob: p, Token: token}
	}

	// Sort by probability in descending order
	sort.Slice(probs, func(i, j int) bool {
		return probs[i].Prob > probs[j].Prob
	})

	// Take the top k
	if k > len(probs) {
		k = len(probs)
	}
	topKProbs := probs[:k]

	// Normalize the probabilities of the top k tokens
	var sum float64
	for _, p := range topKProbs {
		sum += p.Prob
	}
	if sum == 0 {
		// If all top-k probabilities are zero, return the first one (highest prob)
		if len(topKProbs) > 0 {
			return topKProbs[0].ID, nil
		}
		return 0, fmt.Errorf("no valid tokens to sample from")
	}

	// Sample from the normalized distribution
	r := rand.Float64() * sum
	var cumulativeProb float64
	for _, p := range topKProbs {
		cumulativeProb += p.Prob
		if r < cumulativeProb {
			return p.ID, nil
		}
	}

	// Fallback to the most probable token if something goes wrong
	if len(topKProbs) > 0 {
		return topKProbs[0].ID, nil
	}

	return 0, fmt.Errorf("sampling failed: no tokens in top-k")
}