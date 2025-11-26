package moe

import (
	"encoding/gob"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"nlptagger/neural/nn"
	"nlptagger/neural/nnu/word2vec"
	"nlptagger/neural/tensor"
	"os"
	"sort"
)

func init() {
	gob.Register(&IntentMoE{})
	gob.Register(&RNNDecoder{})
	gob.Register(&MoELayer{})
	gob.Register(&FeedForwardExpert{})
	gob.Register(&GatingNetwork{})
	gob.Register(&nn.Linear{})
	gob.Register(&nn.Embedding{})
	gob.Register(&nn.LSTM{})
	gob.Register(&nn.LSTMCell{})
	gob.Register([]*nn.LSTMCell{})
	gob.Register([][]*nn.LSTMCell{})
	gob.Register(&tensor.ConcatOperation{})
	gob.Register(&tensor.DivScalarOperation{})
	gob.Register(&tensor.AddOperation{})
	gob.Register(&tensor.MatmulOperation{})
	gob.Register(&tensor.AddWithBroadcastOperation{})
	gob.Register(&tensor.SoftmaxOperation{})
	gob.Register(&tensor.MulScalarOperation{})
	gob.Register(&tensor.SelectOperation{})
	gob.Register(&tensor.TanhOperation{})
	gob.Register(&tensor.SigmoidOperation{})
	gob.Register(&tensor.LogOperation{})
	gob.Register(&tensor.MulOperation{})
	gob.Register(&tensor.SumOperation{})
	gob.Register(&tensor.SplitOperation{})
}

// sampleFromLogits samples a token ID from logits using temperature, top-k, and top-p sampling.
func sampleFromLogits(logits *tensor.Tensor, temperature float64, topK int, topP float64) (int, error) {
	// logits shape: [batchSize, vocabSize]
	// We assume batchSize = 1 for inference
	if logits.Shape[0] != 1 {
		return 0, fmt.Errorf("sampleFromLogits expects batch size 1, got %d", logits.Shape[0])
	}

	vocabSize := logits.Shape[1]
	logitsData := logits.Data

	// Apply temperature scaling
	if temperature <= 0.0 {
		temperature = 1.0 // Default to 1.0 if invalid
	}

	scaledLogits := make([]float64, vocabSize)
	for i := 0; i < vocabSize; i++ {
		scaledLogits[i] = logitsData[i] / temperature
	}

	// Convert to probabilities using softmax
	maxLogit := scaledLogits[0]
	for i := 1; i < vocabSize; i++ {
		if scaledLogits[i] > maxLogit {
			maxLogit = scaledLogits[i]
		}
	}

	expSum := 0.0
	probs := make([]float64, vocabSize)
	for i := 0; i < vocabSize; i++ {
		probs[i] = math.Exp(scaledLogits[i] - maxLogit)
		expSum += probs[i]
	}
	for i := 0; i < vocabSize; i++ {
		probs[i] /= expSum
	}

	// Apply top-k filtering if specified
	if topK > 0 && topK < vocabSize {
		// Create index-probability pairs
		type indexProb struct {
			index int
			prob  float64
		}
		pairs := make([]indexProb, vocabSize)
		for i := 0; i < vocabSize; i++ {
			pairs[i] = indexProb{index: i, prob: probs[i]}
		}

		// Sort by probability descending
		sort.Slice(pairs, func(i, j int) bool {
			return pairs[i].prob > pairs[j].prob
		})

		// Zero out probabilities outside top-k
		topKIndices := make(map[int]bool)
		for i := 0; i < topK; i++ {
			topKIndices[pairs[i].index] = true
		}
		for i := 0; i < vocabSize; i++ {
			if !topKIndices[i] {
				probs[i] = 0.0
			}
		}

		// Renormalize
		probSum := 0.0
		for i := 0; i < vocabSize; i++ {
			probSum += probs[i]
		}
		if probSum > 0 {
			for i := 0; i < vocabSize; i++ {
				probs[i] /= probSum
			}
		}
	}

	// Apply top-p (nucleus) filtering if specified
	if topP > 0.0 && topP < 1.0 {
		// Create index-probability pairs
		type indexProb struct {
			index int
			prob  float64
		}
		pairs := make([]indexProb, vocabSize)
		for i := 0; i < vocabSize; i++ {
			pairs[i] = indexProb{index: i, prob: probs[i]}
		}

		// Sort by probability descending
		sort.Slice(pairs, func(i, j int) bool {
			return pairs[i].prob > pairs[j].prob
		})

		// Find cumulative probability cutoff
		cumProb := 0.0
		cutoffIdx := vocabSize
		for i := 0; i < vocabSize; i++ {
			cumProb += pairs[i].prob
			if cumProb >= topP {
				cutoffIdx = i + 1
				break
			}
		}

		// Zero out probabilities outside nucleus
		nucleusIndices := make(map[int]bool)
		for i := 0; i < cutoffIdx; i++ {
			nucleusIndices[pairs[i].index] = true
		}
		for i := 0; i < vocabSize; i++ {
			if !nucleusIndices[i] {
				probs[i] = 0.0
			}
		}

		// Renormalize
		probSum := 0.0
		for i := 0; i < vocabSize; i++ {
			probSum += probs[i]
		}
		if probSum > 0 {
			for i := 0; i < vocabSize; i++ {
				probs[i] /= probSum
			}
		}
	}

	// Sample from the probability distribution
	r := rand.Float64()
	cumProb := 0.0
	for i := 0; i < vocabSize; i++ {
		cumProb += probs[i]
		if r <= cumProb {
			return i, nil
		}
	}

	// Fallback: return the last token (should rarely happen)
	return vocabSize - 1, nil
}

// IntentMoE represents a Mixture of Experts model for intent classification.
type IntentMoE struct {
	Encoder           *MoELayer
	Decoder           *RNNDecoder
	Embedding         *nn.Embedding
	SentenceVocabSize int
}

// NewIntentMoE creates a new IntentMoE model.
func NewIntentMoE(vocabSize, embeddingDim, numExperts, parentVocabSize, childVocabSize, sentenceVocabSize, maxAttentionHeads int, word2vecModel *word2vec.SimpleWord2Vec) (*IntentMoE, error) {
	if word2vecModel != nil {
		vocabSize = word2vecModel.VocabSize
		// embeddingDim = word2vecModel.VectorSize // Commented out to allow explicit embeddingDim
	}
	embedding := nn.NewEmbedding(vocabSize, embeddingDim)
	if word2vecModel != nil {
		embedding.LoadPretrainedWeights(word2vecModel.WordVectors)
	}

	// Define the expert builder function
	expertBuilder := func(expertIdx int) (Expert, error) {
		return NewFeedForwardExpert(embeddingDim, embeddingDim, embeddingDim) // Example: inputDim, hiddenDim, outputDim
	}

	// Initialize the MoE encoder
	// Assuming k=1 (select top 1 expert) for simplicity, adjust as needed
	encoder, err := NewMoELayer(embeddingDim, numExperts, 1, expertBuilder)
	if err != nil {
		return nil, fmt.Errorf("failed to create MoE encoder: %w", err)
	}

	// Initialize the RNN Decoder (legacy code - using defaults: 1 layer, no dropout)
	decoder, err := NewRNNDecoder(embeddingDim, sentenceVocabSize, embeddingDim, maxAttentionHeads, 1, 0.0)
	if err != nil {
		return nil, fmt.Errorf("failed to create RNN decoder: %w", err)
	}

	return &IntentMoE{
			Encoder:           encoder,
			Decoder:           decoder,
			Embedding:         embedding,
			SentenceVocabSize: sentenceVocabSize,
		},
		nil
}

// Forward performs the forward pass of the IntentMoE model.
// scheduledSamplingProb: probability of using model predictions instead of ground truth (0.0 for inference)
func (m *IntentMoE) Forward(scheduledSamplingProb float64, inputs ...*tensor.Tensor) ([]*tensor.Tensor, *tensor.Tensor, error) {
	if len(inputs) != 2 {
		return nil, nil, fmt.Errorf("IntentMoE.Forward expects 2 inputs (query token IDs, target token IDs), got %d", len(inputs))
	}
	queryTokenIDs := inputs[0]
	targetTokenIDs := inputs[1]

	// Pass token IDs through embedding layer
	queryEmbeddings, err := m.Embedding.Forward(queryTokenIDs)
	if err != nil {
		return nil, nil, fmt.Errorf("embedding layer forward failed: %w", err)
	}

	// Encoder forward pass
	contextVector, err := m.Encoder.Forward(queryEmbeddings)
	if err != nil {
		return nil, nil, fmt.Errorf("MoE encoder forward failed: %w", err)
	}

	// Decoder forward pass with scheduled sampling
	sentenceLogits, err := m.Decoder.Forward(contextVector, targetTokenIDs, scheduledSamplingProb)
	if err != nil {
		return nil, nil, fmt.Errorf("decoder forward failed: %w", err)
	}

	return sentenceLogits, contextVector, nil
}

// Backward performs the backward pass for the IntentMoE model.
func (m *IntentMoE) Backward(grads ...*tensor.Tensor) error {
	sentenceGrads := grads

	// Backward pass for the decoder
	if err := m.Decoder.Backward(sentenceGrads); err != nil {
		return fmt.Errorf("decoder backward failed: %w", err)
	}
	contextVectorGrad := m.Decoder.InitialHiddenState.Grad

	// Backward pass for the encoder
	if err := m.Encoder.Backward(contextVectorGrad); err != nil {
		return fmt.Errorf("MoE encoder backward failed: %w", err)
	}

	// Backward pass for the embedding layer
	embeddingGrad := m.Encoder.Inputs()[0].Grad
	if err := m.Embedding.Backward(embeddingGrad); err != nil {
		return fmt.Errorf("embedding layer backward failed: %w", err)
	}

	return nil
}

// Parameters returns all learnable parameters of the IntentMoE model.
func (m *IntentMoE) Parameters() []*tensor.Tensor {
	params := []*tensor.Tensor{}
	params = append(params, m.Embedding.Parameters()...)
	params = append(params, m.Encoder.Parameters()...)
	params = append(params, m.Decoder.Parameters()...)
	return params
}

func (m *IntentMoE) GreedySearchDecode(contextVector *tensor.Tensor, maxLen, sosToken, eosToken int, repetitionPenalty float64) ([]int, error) {
	var decodedIDs []int
	decoderInputIDs := tensor.NewTensor([]int{1, 1}, []float64{float64(sosToken)}, false)

	// Take the first element of the batch
	contextVector, err := contextVector.Slice(0, 0, 1)
	if err != nil {
		return nil, fmt.Errorf("failed to slice context vector: %w", err)
	}

	batchSize := contextVector.Shape[0]
	hiddenSize := m.Decoder.LSTM.HiddenSize

	initialHidden, err := contextVector.Mean(1)
	if err != nil {
		return nil, fmt.Errorf("failed to get mean of context vector for initial hidden state: %w", err)
	}

	if initialHidden.Shape[1] != hiddenSize {
		if initialHidden.Shape[1] > hiddenSize {
			initialHidden, err = initialHidden.Slice(1, 0, hiddenSize)
			if err != nil {
				return nil, fmt.Errorf("failed to slice initial hidden state: %w", err)
			}
		} else if initialHidden.Shape[1] < hiddenSize {
			padding := tensor.NewTensor([]int{batchSize, hiddenSize - initialHidden.Shape[1]}, make([]float64, batchSize*(hiddenSize-initialHidden.Shape[1])), false)
			initialHidden, err = tensor.Concat([]*tensor.Tensor{initialHidden, padding}, 1)
			if err != nil {
				return nil, fmt.Errorf("failed to pad initial hidden state: %w", err)
			}
		}
	}

	hiddenState := initialHidden
	cellState := tensor.NewTensor([]int{batchSize, hiddenSize}, make([]float64, batchSize*hiddenSize), false)

	for i := 0; i < maxLen; i++ {
		outputLogits, newHidden, newCell, err := m.Decoder.DecodeStep(decoderInputIDs, hiddenState, cellState, contextVector)
		if err != nil {
			return nil, fmt.Errorf("decoder step failed: %w", err)
		}

		hiddenState = newHidden
		cellState = newCell

		// Apply repetition penalty
		if repetitionPenalty != 1.0 {
			for _, id := range decodedIDs {
				if id < len(outputLogits.Data) {
					if outputLogits.Data[id] < 0 {
						outputLogits.Data[id] *= repetitionPenalty
					} else {
						outputLogits.Data[id] /= repetitionPenalty
					}
				}
			}
		}

		argmax, err := outputLogits.Argmax(1)
		if err != nil {
			return nil, fmt.Errorf("argmax failed: %w", err)
		}
		predictedID := int(argmax.Data[0])

		log.Printf("Step %d: Predicted ID %d (EOS: %d)\n", i, predictedID, eosToken) // Debug logging

		if predictedID == eosToken {
			break
		}

		decodedIDs = append(decodedIDs, predictedID)

		decoderInputIDs = tensor.NewTensor([]int{1, 1}, []float64{float64(predictedID)}, false)
	}

	return decodedIDs, nil
}

// SampleDecode performs sampling-based decoding with temperature, top-k, and top-p (nucleus) sampling.
// temperature: controls randomness (0.0 = deterministic, 1.0 = normal, >1.0 = more random)
// topK: if > 0, only sample from top K tokens
// topP: if > 0.0 and < 1.0, only sample from tokens whose cumulative probability is <= topP
func (m *IntentMoE) SampleDecode(contextVector *tensor.Tensor, maxLen, sosToken, eosToken int, temperature float64, topK int, topP float64, repetitionPenalty float64) ([]int, error) {
	var decodedIDs []int
	decoderInputIDs := tensor.NewTensor([]int{1, 1}, []float64{float64(sosToken)}, false)

	// Take the first element of the batch
	contextVector, err := contextVector.Slice(0, 0, 1)
	if err != nil {
		return nil, fmt.Errorf("failed to slice context vector: %w", err)
	}

	batchSize := contextVector.Shape[0]
	hiddenSize := m.Decoder.LSTM.HiddenSize

	initialHidden, err := contextVector.Mean(1)
	if err != nil {
		return nil, fmt.Errorf("failed to get mean of context vector for initial hidden state: %w", err)
	}

	if initialHidden.Shape[1] != hiddenSize {
		if initialHidden.Shape[1] > hiddenSize {
			initialHidden, err = initialHidden.Slice(1, 0, hiddenSize)
			if err != nil {
				return nil, fmt.Errorf("failed to slice initial hidden state: %w", err)
			}
		} else if initialHidden.Shape[1] < hiddenSize {
			padding := tensor.NewTensor([]int{batchSize, hiddenSize - initialHidden.Shape[1]}, make([]float64, batchSize*(hiddenSize-initialHidden.Shape[1])), false)
			initialHidden, err = tensor.Concat([]*tensor.Tensor{initialHidden, padding}, 1)
			if err != nil {
				return nil, fmt.Errorf("failed to pad initial hidden state: %w", err)
			}
		}
	}

	hiddenState := initialHidden
	cellState := tensor.NewTensor([]int{batchSize, hiddenSize}, make([]float64, batchSize*hiddenSize), false)

	for i := 0; i < maxLen; i++ {
		outputLogits, newHidden, newCell, err := m.Decoder.DecodeStep(decoderInputIDs, hiddenState, cellState, contextVector)
		if err != nil {
			return nil, fmt.Errorf("decoder step failed: %w", err)
		}

		hiddenState = newHidden
		cellState = newCell

		// Apply repetition penalty
		if repetitionPenalty != 1.0 {
			for _, id := range decodedIDs {
				if id < len(outputLogits.Data) {
					if outputLogits.Data[id] < 0 {
						outputLogits.Data[id] *= repetitionPenalty
					} else {
						outputLogits.Data[id] /= repetitionPenalty
					}
				}
			}
		}

		// Sample from the logits with temperature, top-k, and top-p
		predictedID, err := sampleFromLogits(outputLogits, temperature, topK, topP)
		if err != nil {
			return nil, fmt.Errorf("sampling failed: %w", err)
		}

		log.Printf("Step %d: Sampled ID %d (EOS: %d)\n", i, predictedID, eosToken) // Debug logging

		if predictedID == eosToken {
			break
		}

		decodedIDs = append(decodedIDs, predictedID)

		decoderInputIDs = tensor.NewTensor([]int{1, 1}, []float64{float64(predictedID)}, false)
	}

	return decodedIDs, nil
}

// SaveIntentMoEModelToGOB saves the IntentMoE to a file in Gob format.
func SaveIntentMoEModelToGOB(model *IntentMoE, writer io.Writer) error {
	encoder := gob.NewEncoder(writer)
	err := encoder.Encode(model)
	if err != nil {
		return fmt.Errorf("failed to encode IntentMoE model to Gob: %w", err)
	}

	return nil
}

// LoadIntentMoEModelFromGOB loads a IntentMoE from a file in Gob format.
func LoadIntentMoEModelFromGOB(filePath string) (*IntentMoE, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("error opening IntentMoE model gob file: %w", err)
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	var loadedModel IntentMoE
	err = decoder.Decode(&loadedModel)
	if err != nil {
		return nil, fmt.Errorf("error decoding IntentMoE model from gob: %w", err)
	}

	if loadedModel.Encoder == nil {
		return nil, fmt.Errorf("loaded IntentMoE model has a nil Encoder after decoding")
	}

	if loadedModel.Decoder == nil {
		return nil, fmt.Errorf("loaded IntentMoE model's Decoder has a nil Decoder after decoding")
	}

	return &loadedModel, nil
}
