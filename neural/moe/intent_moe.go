package moe

import (
	"encoding/gob"
	"fmt"
	"io"
	"nlptagger/neural/nn"
	"nlptagger/neural/nnu/word2vec"
	"nlptagger/neural/tensor"
	"os"
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

		argmax, err := outputLogits.Argmax(1)
		if err != nil {
			return nil, fmt.Errorf("argmax failed: %w", err)
		}
		predictedID := int(argmax.Data[0])

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
