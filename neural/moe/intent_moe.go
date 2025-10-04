package moe

import (
	"encoding/gob"
	"fmt"
	"nlptagger/neural/nn"
	. "nlptagger/neural/tensor"
	"os"
)

func init() {
	gob.Register(&IntentMoE{})
}

// IntentMoE represents a Mixture of Experts model for intent classification.
type IntentMoE struct {
	Encoder     *MoELayer
	ParentHead  *nn.Linear
	ChildHead   *nn.Linear
	Embedding   *nn.Embedding
}

// NewIntentMoE creates a new IntentMoE model.
func NewIntentMoE(vocabSize, embeddingDim, numExperts, parentVocabSize, childVocabSize int) (*IntentMoE, error) {
	embedding := nn.NewEmbedding(vocabSize, embeddingDim)

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

	return &IntentMoE{
		Encoder:     encoder,
		ParentHead:  parentHead,
		ChildHead:   childHead,
		Embedding:   embedding,
	},
		nil
}

// Forward performs the forward pass of the IntentMoE model.
func (m *IntentMoE) Forward(inputs ...*Tensor) (*Tensor, *Tensor, error) {
	if len(inputs) != 1 {
		return nil, nil, fmt.Errorf("IntentMoE.Forward expects 1 input (query token IDs), got %d", len(inputs))
	}
	queryTokenIDs := inputs[0]

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

	// Average the context vectors over the sequence length
	contextVector, err = contextVector.Mean(1)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to average context vector: %w", err)
	}

	// Parent and child heads
	parentLogits, err := m.ParentHead.Forward(contextVector)
	if err != nil {
		return nil, nil, fmt.Errorf("parent head forward failed: %w", err)
	}
	childLogits, err := m.ChildHead.Forward(contextVector)
	if err != nil {
		return nil, nil, fmt.Errorf("child head forward failed: %w", err)
	}

	return parentLogits, childLogits, nil
}

// Backward performs the backward pass for the IntentMoE model.
func (m *IntentMoE) Backward(grads ...*Tensor) error {
	if len(grads) != 2 {
		return fmt.Errorf("IntentMoE.Backward expects 2 gradients (parent, child), got %d", len(grads))
	}
	parentGrad := grads[0]
	childGrad := grads[1]

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
func (m *IntentMoE) Parameters() []*Tensor {
	params := []*Tensor{}
	params = append(params, m.Embedding.Parameters()...)
	params = append(params, m.Encoder.Parameters()...)
	params = append(params, m.ParentHead.Parameters()...)
	params = append(params, m.ChildHead.Parameters()...)
	return params
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

	return &loadedModel, nil
}
