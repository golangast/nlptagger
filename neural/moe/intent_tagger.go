package moe

import (
	"fmt"

	"github.com/zendrulat/nlptagger/neural/nn"
	"github.com/zendrulat/nlptagger/neural/tensor"
)

// IntentTagger represents a model that predicts intent and tags for a sequence.
type IntentTagger struct {
	Encoder    *MoELayer
	Embedding  *nn.Embedding
	IntentHead *nn.Linear
	TagHead    *nn.Linear
}

// NewIntentTagger creates a new IntentTagger model.
func NewIntentTagger(vocabSize, embeddingDim, numExperts, intentVocabSize, tagVocabSize int) (*IntentTagger, error) {
	embedding := nn.NewEmbedding(vocabSize, embeddingDim)

	// Define the expert builder function
	expertBuilder := func(expertIdx int) (Expert, error) {
		return NewFeedForwardExpert(embeddingDim, embeddingDim, embeddingDim)
	}

	// Initialize the MoE encoder
	encoder, err := NewMoELayer(embeddingDim, numExperts, 1, expertBuilder)
	if err != nil {
		return nil, fmt.Errorf("failed to create MoE encoder: %w", err)
	}

	// Initialize the output heads
	intentHead, err := nn.NewLinear(embeddingDim, intentVocabSize)
	if err != nil {
		return nil, fmt.Errorf("failed to create intent head: %w", err)
	}
	tagHead, err := nn.NewLinear(embeddingDim, tagVocabSize)
	if err != nil {
		return nil, fmt.Errorf("failed to create tag head: %w", err)
	}

	return &IntentTagger{
		Encoder:    encoder,
		Embedding:  embedding,
		IntentHead: intentHead,
		TagHead:    tagHead,
	}, nil
}

// Forward performs the forward pass of the IntentTagger model.
func (m *IntentTagger) Forward(inputs ...*tensor.Tensor) (*tensor.Tensor, []*tensor.Tensor, error) {
	if len(inputs) != 1 {
		return nil, nil, fmt.Errorf("IntentTagger.Forward expects 1 input (query token IDs), got %d", len(inputs))
	}
	queryTokenIDs := inputs[0]

	// Pass token IDs through embedding layer
	queryEmbeddings, err := m.Embedding.Forward(queryTokenIDs)
	if err != nil {
		return nil, nil, fmt.Errorf("embedding layer forward failed: %w", err)
	}

	// Encoder forward pass
	encodedSequence, err := m.Encoder.Forward(queryEmbeddings)
	if err != nil {
		return nil, nil, fmt.Errorf("MoE encoder forward failed: %w", err)
	}

	// For intent prediction, we can take the mean of the encoded sequence
	contextVector, err := encodedSequence.Mean(1)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to get mean of encoded sequence: %w", err)
	}

	// Intent head
	intentLogits, err := m.IntentHead.Forward(contextVector)
	if err != nil {
		return nil, nil, fmt.Errorf("intent head forward failed: %w", err)
	}

	// Tag head - apply to each token in the sequence
	batchSize := encodedSequence.Shape[0]
	seqLength := encodedSequence.Shape[1]
	hiddenSize := encodedSequence.Shape[2]

	tagLogits := make([]*tensor.Tensor, seqLength)
	for t := 0; t < seqLength; t++ {
		// Get the output for the current time step
		timeStepOutput, err := encodedSequence.Slice(1, t, t+1)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to slice encoded sequence: %w", err)
		}
		// Reshape to (batchSize, hiddenSize)
		timeStepOutput.Reshape([]int{batchSize, hiddenSize})

		// Pass through the tag head
		tagLogit, err := m.TagHead.Forward(timeStepOutput)
		if err != nil {
			return nil, nil, fmt.Errorf("tag head forward failed for time step %d: %w", t, err)
		}
		tagLogits[t] = tagLogit
	}

	return intentLogits, tagLogits, nil
}

// Parameters returns all learnable parameters of the IntentTagger model.
func (m *IntentTagger) Parameters() []*tensor.Tensor {
	params := []*tensor.Tensor{}
	params = append(params, m.Embedding.Parameters()...)
	params = append(params, m.Encoder.Parameters()...)
	params = append(params, m.IntentHead.Parameters()...)
	params = append(params, m.TagHead.Parameters()...)
	return params
}

// Backward performs the backward pass for the IntentTagger model.
func (m *IntentTagger) Backward(intentGrad, tagGrads *tensor.Tensor) error {
    // Backward pass for the heads
    if err := m.IntentHead.Backward(intentGrad); err != nil {
        return fmt.Errorf("intent head backward failed: %w", err)
    }
    if err := m.TagHead.Backward(tagGrads); err != nil {
        return fmt.Errorf("tag head backward failed: %w", err)
    }

    // Combine gradients for the encoder
    tagEncoderGrad := m.TagHead.Inputs()[0].Grad

    // Backward pass for the encoder
    if err := m.Encoder.Backward(tagEncoderGrad); err != nil {
        return fmt.Errorf("MoE encoder backward failed: %w", err)
    }

    // Backward pass for the embedding layer
    embeddingGrad := m.Encoder.Inputs()[0].Grad
    if err := m.Embedding.Backward(embeddingGrad); err != nil {
        return fmt.Errorf("embedding layer backward failed: %w", err)
    }

    return nil
}
