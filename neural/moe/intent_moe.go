package moe

import (
	"encoding/gob"
	"fmt"
	"io"
	"log"
	"os"

	"github.com/zendrulat/nlptagger/neural/nn"
	"github.com/zendrulat/nlptagger/neural/nnu/word2vec"
	t "github.com/zendrulat/nlptagger/neural/tensor"
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
		embeddingDim = word2vecModel.VectorSize
		// Ensure embeddingDim is divisible by maxAttentionHeads
		if embeddingDim%maxAttentionHeads != 0 {
			embeddingDim = (embeddingDim/maxAttentionHeads + 1) * maxAttentionHeads
			log.Printf("Adjusted embeddingDim to %d to be divisible by maxAttentionHeads (%d)", embeddingDim, maxAttentionHeads)
		}
	}
	log.Printf("NewIntentMoE: Initializing Embedding layer with vocabSize: %d, embeddingDim: %d", vocabSize, embeddingDim)
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
func (m *IntentMoE) Inference(inputs ...*t.Tensor) (*t.Tensor, error) {
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
func (m *IntentMoE) Forward(inputs ...*t.Tensor) ([]*t.Tensor, *t.Tensor, error) {
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
func (m *IntentMoE) Backward(grads ...*t.Tensor) error {
	sentenceGrads := grads

	// Backward pass for the decoder
	if err := m.Decoder.Backward(sentenceGrads); err != nil {
		return fmt.Errorf("decoder backward failed: %w", err)
	}
	contextVectorGrad := m.Decoder.ContextVector.Grad

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
func (m *IntentMoE) Parameters() []*t.Tensor {
	params := []*t.Tensor{}
	params = append(params, m.Embedding.Parameters()...)
	params = append(params, m.Encoder.Parameters()...)
	params = append(params, m.Decoder.Parameters()...)
	return params
}


func (m *IntentMoE) GreedySearchDecode(contextVector *t.Tensor, maxLen, sosToken, eosToken int) ([]int, error) {
	fmt.Fprintf(os.Stderr, "GreedySearchDecode: Function entered. maxLen: %d, sosToken: %d, eosToken: %d\n", maxLen, sosToken, eosToken)
	if contextVector.Shape[0] > 1 {
		var err error
		contextVector, err = contextVector.Slice(0, 0, 1)
		if err != nil {
			return nil, fmt.Errorf("failed to slice context vector for single-item decoding: %w", err)
		}
	}
	var decodedIDs []int
	decoderInputIDs := t.NewTensor([]int{1, 1}, []float64{float64(sosToken)}, false)

	// Average the contextVector along the sequence length dimension to get a single vector
	// for initializing the decoder's hidden state.
	initialHidden, err := contextVector.Mean(1) // Average along the sequence length dimension
	if err != nil {
		return nil, fmt.Errorf("failed to average context vector for initial hidden state: %w", err)
	}
	// Reshape to [batchSize, embeddingDim]
	initialHidden, err = initialHidden.Reshape([]int{contextVector.Shape[0], contextVector.Shape[2]})
	if err != nil {
		return nil, fmt.Errorf("failed to reshape initial hidden state: %w", err)
	}

	batchSize := contextVector.Shape[0]
	hiddenSize := m.Decoder.LSTM.HiddenSize

	// If the hiddenSize of LSTM is different from embeddingDim, we need to adjust initialHidden
	if initialHidden.Shape[1] != hiddenSize {
		if initialHidden.Shape[1] > hiddenSize {
			initialHidden, err = initialHidden.Slice(1, 0, hiddenSize)
			if err != nil {
				return nil, fmt.Errorf("failed to slice initial hidden state: %w", err)
			}
		} else if initialHidden.Shape[1] < hiddenSize {
			padding := t.NewTensor([]int{batchSize, hiddenSize - initialHidden.Shape[1]}, make([]float64, batchSize*(hiddenSize-initialHidden.Shape[1])), false)
			initialHidden, err = t.Concat([]*t.Tensor{initialHidden, padding}, 1)
			if err != nil {
				return nil, fmt.Errorf("failed to pad initial hidden state: %w", err)
			}
		}
	}

	hiddenState := initialHidden
	cellState := t.NewTensor([]int{batchSize, hiddenSize}, make([]float64, batchSize*hiddenSize), false)

	for i := 0; i < maxLen; i++ {
		outputProbabilities, newHidden, newCell, err := m.Decoder.DecodeStep(decoderInputIDs, hiddenState, cellState, contextVector)
		if err != nil {
			fmt.Fprintf(os.Stderr, "GreedySearchDecode: Decoder step failed at iteration %d: %v\n", i, err)
			return nil, fmt.Errorf("decoder step failed: %w", err)
		}

		hiddenState = newHidden
		cellState = newCell

		// Get the predicted ID (greedy approach)
		if i == 0 {
			// Prevent EOS token from being the first token
			if len(outputProbabilities.Data) > eosToken {
				outputProbabilities.Data[eosToken] = -1e9
			}
		}
		predictedIDTensor, err := outputProbabilities.Argmax(1)
		if err != nil {
			return nil, fmt.Errorf("failed to get argmax of output probabilities: %w", err)
		}
		predictedID := int(predictedIDTensor.Data[0])

		fmt.Fprintf(os.Stderr, "GreedySearchDecode: Iteration %d, Predicted ID: %d, EOS Token: %d\n", i, predictedID, eosToken)

		if predictedID == eosToken {
			break
		}
		decodedIDs = append(decodedIDs, predictedID)

		decoderInputIDs.Data[0] = float64(predictedID)
	}
	fmt.Fprintf(os.Stderr, "GreedySearchDecode: Function exited. Decoded IDs: %v\n", decodedIDs)
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
