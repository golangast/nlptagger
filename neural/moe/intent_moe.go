package moe

import (
	"encoding/gob"
	"fmt"
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
	ParentHead        *nn.Linear
	ChildHead         *nn.Linear
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

	// Initialize the MoE encoder
	expertBuilder := func(expertIdx int) (Expert, error) {
		return NewFeedForwardExpert(embeddingDim, embeddingDim, embeddingDim)
	}
	encoder, err := NewMoELayer(embeddingDim, numExperts, 1, expertBuilder)
	if err != nil {
		return nil, fmt.Errorf("failed to create MoE encoder: %w", err)
	}

	// Initialize the parent and child heads
	parentHead, err := nn.NewLinear(embeddingDim, parentVocabSize)
	if err != nil {
		return nil, fmt.Errorf("failed to create parent head: %w", err)
	}
	childHead, err := nn.NewLinear(embeddingDim, childVocabSize)
	if err != nil {
		return nil, fmt.Errorf("failed to create child head: %w", err)
	}

	// Initialize the RNN Decoder
	decoder, err := NewRNNDecoder(embeddingDim, sentenceVocabSize, embeddingDim, maxAttentionHeads)
	if err != nil {
		return nil, fmt.Errorf("failed to create RNN decoder: %w", err)
	}

	return &IntentMoE{
		Encoder:           encoder,
		ParentHead:        parentHead,
		ChildHead:         childHead,
		Decoder:           decoder,
		Embedding:         embedding,
		SentenceVocabSize: sentenceVocabSize,
	},
		nil
}

// Forward performs the forward pass of the IntentMoE model.
func (m *IntentMoE) Forward(inputs ...*tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, []*tensor.Tensor, *tensor.Tensor, error) {
	if len(inputs) != 2 {
		return nil, nil, nil, nil, fmt.Errorf("IntentMoE.Forward expects 2 inputs (query token IDs, target token IDs), got %d", len(inputs))
	}
	queryTokenIDs := inputs[0]
	targetTokenIDs := inputs[1]

	// Pass token IDs through embedding layer
	queryEmbeddings, err := m.Embedding.Forward(queryTokenIDs)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("embedding layer forward failed: %w", err)
	}

	// Encoder forward pass
	contextVector, err := m.Encoder.Forward(queryEmbeddings)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("MoE encoder forward failed: %w", err)
	}

	// Average the context vectors over the sequence length for classification heads
	avgContextVector, err := contextVector.Mean(1)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("failed to average context vector: %w", err)
	}

	// Parent and child heads
	parentLogits, err := m.ParentHead.Forward(avgContextVector)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("parent head forward failed: %w", err)
	}
	childLogits, err := m.ChildHead.Forward(avgContextVector)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("child head forward failed: %w", err)
	}

	// Decoder forward pass
	sentenceLogits, err := m.Decoder.Forward(contextVector, targetTokenIDs)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("decoder forward failed: %w", err)
	}

	return parentLogits, childLogits, sentenceLogits, contextVector, nil
}

// Backward performs the backward pass for the IntentMoE model.
func (m *IntentMoE) Backward(grads ...*tensor.Tensor) error {
	if len(grads) < 2 {
		return fmt.Errorf("IntentMoE.Backward expects at least 2 gradients (parent, child), got %d", len(grads))
	}
	parentGrad := grads[0]
	childGrad := grads[1]
	sentenceGrads := grads[2:]

	// Backward pass for the parent and child heads
	if err := m.ParentHead.Backward(parentGrad); err != nil {
		return fmt.Errorf("parent head backward failed: %w", err)
	}
	if err := m.ChildHead.Backward(childGrad); err != nil {
		return fmt.Errorf("child head backward failed: %w", err)
	}

	// The gradient from the heads is the gradient for the context vector.
	// We need to get this gradient and pass it to the encoder.
	contextVectorGrad := m.ParentHead.Input().Grad
	contextVectorGrad.Add(m.ChildHead.Input().Grad)

	// Backward pass for the decoder
	if err := m.Decoder.Backward(sentenceGrads); err != nil {
		return fmt.Errorf("decoder backward failed: %w", err)
	}
	contextVectorGrad.Add(m.Decoder.InitialHiddenState.Grad)

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
	params = append(params, m.ParentHead.Parameters()...)
	params = append(params, m.ChildHead.Parameters()...)
	params = append(params, m.Decoder.Parameters()...)
	return params
}

func (m *IntentMoE) GreedySearchDecode(contextVector *tensor.Tensor, maxLen, sosToken, eosToken int) ([]int, error) {
	var decodedIDs []int
	decoderInput := tensor.NewTensor([]int{1, 1}, []float64{float64(sosToken)}, false)

	// Take the first element of the batch
	contextVector, err := contextVector.Slice(0, 0, 1)
	if err != nil {
		return nil, fmt.Errorf("failed to slice context vector: %w", err)
	}

	for i := 0; i < maxLen; i++ {
		// Decoder forward pass for one step
		output, err := m.Decoder.Forward(contextVector, decoderInput)
		if err != nil {
			return nil, fmt.Errorf("decoder forward failed: %w", err)
		}

		// Get the token with the highest probability
		lastOutput := output[len(output)-1]
		argmax, err := lastOutput.Argmax(1)
		if err != nil {
			return nil, fmt.Errorf("argmax failed: %w", err)
		}
		predictedID := int(argmax.Data[0])

		if predictedID == eosToken {
			break
		}

		decodedIDs = append(decodedIDs, predictedID)

		// Use the predicted token as the next input
		decoderInput = tensor.NewTensor([]int{1, 1}, []float64{float64(predictedID)}, false)
	}

	return decodedIDs, nil
}

// SaveIntentMoEModelToGOB saves the IntentMoE to a file in Gob format.
func SaveIntentMoEModelToGOB(model *IntentMoE, filePath string) error {
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to create file for saving IntentMoE model: %w", err)
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	err = encoder.Encode(model)
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
	if loadedModel.Encoder.GatingNetwork == nil {
		return nil, fmt.Errorf("loaded IntentMoE model's Encoder has a nil GatingNetwork after decoding")
	}

	if loadedModel.Decoder == nil {
		return nil, fmt.Errorf("loaded IntentMoE model's Decoder has a nil Decoder after decoding")
	}

	return &loadedModel, nil
}