package moe

import (
	"encoding/gob"
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
	"sort"

	"github.com/zendrulat/nlptagger/neural/nn"
	"github.com/zendrulat/nlptagger/neural/nnu/word2vec"
	"github.com/zendrulat/nlptagger/neural/tensor"
)

func init() {
	gob.Register(&FeedForwardExpert{})
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
		embeddingDim = word2vecModel.VectorSize
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

	// Initialize the RNN Decoder
	decoder, err := NewRNNDecoder(embeddingDim, sentenceVocabSize, embeddingDim, maxAttentionHeads)
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

// Inference performs the forward pass of the IntentMoE model for inference.
func (m *IntentMoE) Inference(inputs ...*tensor.Tensor) (*tensor.Tensor, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("IntentMoE.Inference expects 1 input (query token IDs), got %d", len(inputs))
	}
	queryTokenIDs := inputs[0]

	// Pass token IDs through embedding layer
	queryEmbeddings, err := m.Embedding.Forward(queryTokenIDs)
	if err != nil {
		return nil, fmt.Errorf("embedding layer forward failed: %w", err)
	}

	// Encoder forward pass
	contextVector, err := m.Encoder.Forward(queryEmbeddings)
	if err != nil {
		return nil, fmt.Errorf("MoE encoder forward failed: %w", err)
	}

	return contextVector, nil
}

// Forward performs the forward pass of the IntentMoE model.
func (m *IntentMoE) Forward(inputs ...*tensor.Tensor) ([]*tensor.Tensor, *tensor.Tensor, error) {
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

	// Decoder forward pass
	sentenceLogits, err := m.Decoder.Forward(contextVector, targetTokenIDs)
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

type probIndex struct {
	prob  float64
	index int
}

func sampleTopK(probabilities []float64, k int) int {
	// Create a slice of probIndex to store probabilities and their original indices
	probIndices := make([]probIndex, len(probabilities))
	for i, p := range probabilities {
		probIndices[i] = probIndex{prob: p, index: i}
	}

	// Sort the probabilities in descending order
	sort.Slice(probIndices, func(i, j int) bool {
		return probIndices[i].prob > probIndices[j].prob
	})

	// Take the top k probabilities
	topK := probIndices[:k]

	// Normalize the top k probabilities
	var sumProb float64
	for _, pi := range topK {
		sumProb += pi.prob
	}
	for i := range topK {
		topK[i].prob /= sumProb
	}

	// Sample from the normalized top k probabilities
	r := rand.Float64()
	var cumulativeProb float64
	for _, pi := range topK {
		cumulativeProb += pi.prob
		if r < cumulativeProb {
			return pi.index
		}
	}

	// Fallback to the most likely token
	return topK[0].index
}

func (m *IntentMoE) GreedySearchDecode(contextVector *tensor.Tensor, maxLen, sosToken, eosToken int) ([]int, error) {
	var decodedIDs []int
	decoderInputIDs := tensor.NewTensor([]int{1, 1}, []float64{float64(sosToken)}, false)

	// Take the first element of the batch
	contextVector, err := contextVector.Slice(0, 0, 1)
	if err != nil {
		return nil, fmt.Errorf("failed to slice context vector: %w", err)
	}

	batchSize := contextVector.Shape[0]
	hiddenSize := m.Decoder.LSTM.HiddenSize

	// Take the mean of the context vector as the initial hidden state
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

		// Apply softmax to get probabilities
		probabilities, err := outputLogits.Softmax(1)
		if err != nil {
			return nil, fmt.Errorf("softmax failed: %w", err)
		}

		// Sample from the top-k probabilities
		predictedID := sampleTopK(probabilities.Data, 5)

		log.Printf("GreedySearchDecode: Iteration %d, predictedID: %d, eosToken: %d", i, predictedID, eosToken)
		log.Printf("GreedySearchDecode: outputLogits: %v", outputLogits.Data)
		if predictedID == eosToken {
			log.Printf("GreedySearchDecode: EOS token encountered, breaking.")
			break
		}
		decodedIDs = append(decodedIDs, predictedID)

		decoderInputIDs.Data[0] = float64(predictedID)
	}

	return decodedIDs, nil
}

// GreedySearchDecode ... (existing code)

// SaveIntentMoEModelToGOB saves the IntentMoE to a file in Gob format.
func SaveIntentMoEModelToGOB(model *IntentMoE, writer io.Writer) error {
	encoder := gob.NewEncoder(writer)
	log.Printf("Saving model of type %T", model)
	if err := encoder.Encode(model); err != nil {
		return fmt.Errorf("failed to encode model: %w", err)
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

	var model IntentMoE
	log.Printf("Loading model of type %T", &model)
	if err := decoder.Decode(&model); err != nil {
		return nil, fmt.Errorf("failed to decode model: %w", err)
	}

	return &model, nil
}
