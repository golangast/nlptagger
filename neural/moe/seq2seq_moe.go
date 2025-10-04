package moe

import (
	"encoding/gob"
	"fmt"
	"nlptagger/neural/nn"
	. "nlptagger/neural/tensor"
	"os"
)

func init() {
	gob.Register(&Seq2SeqMoE{})
	gob.Register(&RNNDecoder{})
}


// Seq2SeqMoE represents a Sequence-to-Sequence model with a Mixture of Experts encoder.
type Seq2SeqMoE struct {
	Encoder           *MoELayer
	Decoder           *RNNDecoder // Use RNNDecoder
	Embedding         *nn.Embedding
	MaxAttentionHeads int
	// Add other necessary components like embedding layers, output layers, etc.
}

// NewSeq2SeqMoE creates a new Seq2SeqMoE model.
func NewSeq2SeqMoE(vocabSize, embeddingDim, numExperts, outputVocabSize, maxAttentionHeads int) (*Seq2SeqMoE, error) {
	embedding := nn.NewEmbedding(vocabSize, embeddingDim)

	// Initialize the MoE encoder
	expertBuilder := func(expertIdx int) (Expert, error) {
		return NewFeedForwardExpert(embeddingDim, embeddingDim, embeddingDim) // Experts operate on embeddingDim
	}
	encoder, err := NewMoELayer(embeddingDim, numExperts, 1, expertBuilder) // MoELayer input is embeddingDim
	if err != nil {
		return nil, fmt.Errorf("failed to create MoE encoder: %w", err)
	}

	// Initialize the RNN Decoder
	decoder, err := NewRNNDecoder(embeddingDim, outputVocabSize, embeddingDim, maxAttentionHeads) // Decoder input is embeddingDim
	if err != nil {
		return nil, fmt.Errorf("failed to create RNN decoder: %w", err)
	}

	return &Seq2SeqMoE{
		Encoder:           encoder,
		Decoder:           decoder,
		Embedding:         embedding,
		MaxAttentionHeads: maxAttentionHeads,
	},
		nil
}

// Forward performs the forward pass of the Seq2SeqMoE model.
// It takes an input tensor (e.g., query embeddings) and aims to produce a sequence (description).
func (m *Seq2SeqMoE) Forward(inputs ...*Tensor) ([]*Tensor, error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("Seq2SeqMoE.Forward expects 2 inputs (query token IDs, target token IDs), got %d", len(inputs))
	}
	queryTokenIDs := inputs[0]
	targetTokenIDs := inputs[1]

	// Pass token IDs through embedding layer
	queryEmbeddings, err := m.Embedding.Forward(queryTokenIDs)
	if err != nil {
		return nil, fmt.Errorf("embedding layer forward failed: %w", err)
	}

	// Encoder forward pass: MoE processes the query embeddings to produce a context vector
	contextVector, err := m.Encoder.Forward(queryEmbeddings)
	if err != nil {
		return nil, fmt.Errorf("MoE encoder forward failed: %w", err)
	}

	// Decoder forward pass: Decoder takes the context vector and generates the description sequence
	descriptionSequence, err := m.Decoder.Forward(contextVector, targetTokenIDs)
	if err != nil {
		return nil, fmt.Errorf("decoder forward failed: %w", err)
	}

	return descriptionSequence, nil
}

// Backward performs the backward pass for the Seq2SeqMoE model.
func (m *Seq2SeqMoE) Backward(grads []*Tensor) error {
	// Backward pass for the decoder
	err := m.Decoder.Backward(grads)
	if err != nil {
		return fmt.Errorf("decoder backward failed: %w", err)
	}

	// The gradient from the decoder is the gradient for the context vector.
	// We need to get this gradient and pass it to the encoder.
	contextVectorGrad := m.Decoder.HiddenState.Grad

	// Backward pass for the encoder
	err = m.Encoder.Backward(contextVectorGrad)
	if err != nil {
		return fmt.Errorf("MoE encoder backward failed: %w", err)
	}

	// Backward pass for the embedding layer
	embeddingGrad := m.Encoder.Inputs()[0].Grad
	err = m.Embedding.Backward(embeddingGrad)
	if err != nil {
		return fmt.Errorf("embedding layer backward failed: %w", err)
	}

	return nil
}

// Parameters returns all learnable parameters of the Seq2SeqMoE model.
func (m *Seq2SeqMoE) Parameters() []*Tensor {
	params := []*Tensor{}
	params = append(params, m.Embedding.Parameters()...)
	params = append(params, m.Encoder.Parameters()...)
	params = append(params, m.Decoder.Parameters()...)
	return params
}

// RNNDecoder is a simple RNN-based decoder for sequence generation.
type RNNDecoder struct {
	// Input to hidden layer
	Wh *nn.Linear
	// Hidden to output layer
	Wo *nn.Linear
	// Hidden state
	HiddenState *Tensor
	// Output vocabulary size
	OutputVocabSize int
	// Embedding layer for the decoder input
	Embedding *nn.Embedding
	MaxAttentionHeads int
	Attention *nn.MultiHeadCrossAttention
}

// NewRNNDecoder creates a new RNNDecoder.
// inputDim is the dimension of the context vector from the encoder.
// outputVocabSize is the size of the target vocabulary.
func NewRNNDecoder(inputDim, outputVocabSize, embeddingDim, maxAttentionHeads int) (*RNNDecoder, error) {
	// For simplicity, let's assume hiddenDim is the same as inputDim for now.
	hiddenDim := inputDim

	wh, err := nn.NewLinear(inputDim+embeddingDim, hiddenDim)
	if err != nil {
		return nil, fmt.Errorf("failed to create Wh linear layer for decoder: %w", err)
	}
	wo, err := nn.NewLinear(hiddenDim, outputVocabSize)
	if err != nil {
		return nil, fmt.Errorf("failed to create Wo linear layer for decoder: %w", err)
	}

	embedding := nn.NewEmbedding(outputVocabSize, embeddingDim)

	attention, err := nn.NewMultiHeadCrossAttention(embeddingDim, maxAttentionHeads, maxAttentionHeads)
	if err != nil {
		return nil, fmt.Errorf("failed to create multi-head attention for decoder: %w", err)
	}

	return &RNNDecoder{
		Wh:              wh,
		Wo:              wo,
		OutputVocabSize: outputVocabSize,
		Embedding:       embedding,
		MaxAttentionHeads: maxAttentionHeads,
		Attention: attention,
	},
		nil
}

// Forward performs the forward pass of the RNNDecoder.
// It takes the context vector from the encoder and the target sequence (for teacher forcing) and generates a sequence of tokens.
func (d *RNNDecoder) Forward(contextVector, targetSequence *Tensor) ([]*Tensor, error) {
	// Initialize hidden state with context vector (or a transformation of it)
	batchSize := targetSequence.Shape[0]
	maxSequenceLength := targetSequence.Shape[1]
	hiddenDim := d.Wo.Weights.Shape[0]
	decoderHiddenState := NewTensor([]int{batchSize, hiddenDim}, make([]float64, batchSize*hiddenDim), false)

	// Create a tensor to hold the decoder outputs
	var outputs []*Tensor

	// Start with a start-of-sequence token (e.g., token ID 0)
	decoderInput := NewTensor([]int{batchSize, 1}, make([]float64, batchSize), false)

	for t := 0; t < maxSequenceLength; t++ {
		// Embed the decoder input
		embeddedInput, err := d.Embedding.Forward(decoderInput)
		if err != nil {
			return nil, fmt.Errorf("decoder embedding failed: %w", err)
		}

		// Apply attention
		attentionOutput, err := d.Attention.Forward(embeddedInput, contextVector, contextVector)
		if err != nil {
			return nil, fmt.Errorf("decoder attention failed: %w", err)
		}

		// Concatenate embedded input and attention output
		combinedInput, err := Concat([]*Tensor{embeddedInput, attentionOutput}, 1)
		if err != nil {
			return nil, fmt.Errorf("decoder concat failed: %w", err)
		}

		// Reshape combinedInput from [batchSize, 2, embeddingDim] to [batchSize, 2 * embeddingDim]
		reshapedCombinedInput, err := combinedInput.Reshape([]int{batchSize, 2 * d.Embedding.DimModel})
		if err != nil {
			return nil, fmt.Errorf("failed to reshape combined input for linear layer: %w", err)
		}
		reshapedCombinedInput.RequiresGrad = true

		// Update hidden state
		decoderHiddenState, err = d.Wh.Forward(reshapedCombinedInput)
		if err != nil {
			return nil, fmt.Errorf("decoder hidden state update failed: %w", err)
		}

		// Hidden state to output
		output, err := d.Wo.Forward(decoderHiddenState)
		if err != nil {
			return nil, fmt.Errorf("decoder output layer forward failed: %w", err)
		}

		outputs = append(outputs, output)

		// Use the target token as the next input (teacher forcing)
		slicedTensor, err := targetSequence.Slice(1, t, t+1)
		if err != nil {
			return nil, fmt.Errorf("error slicing target sequence: %w", err)
		}
		decoderInput = slicedTensor
	}

	d.HiddenState = decoderHiddenState // Store the final hidden state
	return outputs, nil
}

// Backward performs the backward pass of the RNNDecoder.
func (d *RNNDecoder) Backward(grads []*Tensor) error {
	var hiddenGrad *Tensor

	for t := len(grads) - 1; t >= 0; t-- {
		outputGrad := grads[t]

		// Backpropagate through the output layer
		err := d.Wo.Backward(outputGrad)
		if err != nil {
			return fmt.Errorf("decoder output layer backward failed: %w", err)
		}

		// Accumulate gradients for the hidden state
		if hiddenGrad == nil {
			hiddenGrad = d.Wo.Input().Grad
		} else {
			hiddenGrad.Add(d.Wo.Input().Grad)
		}

		// Backpropagate through the hidden layer
		err = d.Wh.Backward(hiddenGrad)
		if err != nil {
			return fmt.Errorf("decoder hidden layer backward failed: %w", err)
		}

		// Split the gradient for the concatenated input
		splitGrads, err := Split(d.Wh.Input().Grad, 1, []int{d.Embedding.DimModel, d.Attention.DimModel})
		if err != nil {
			return fmt.Errorf("decoder split failed: %w", err)
		}
		embeddedGrad := splitGrads[0]
		attentionOutputGrad := splitGrads[1]

		// Reshape attentionOutputGrad to 3D before passing to MultiHeadCrossAttention.Backward
		batchSize := attentionOutputGrad.Shape[0]
		reshapedAttentionOutputGrad, err := attentionOutputGrad.Reshape([]int{batchSize, 1, d.Attention.DimModel})
		if err != nil {
			return fmt.Errorf("failed to reshape attention output gradient for attention backward: %w", err)
		}

		// Backpropagate through the attention layer
		err = d.Attention.Backward(reshapedAttentionOutputGrad)
		if err != nil {
			return fmt.Errorf("decoder attention backward failed: %w", err)
		}

		// Backpropagate through the embedding layer
		err = d.Embedding.Backward(embeddedGrad)
		if err != nil {
			return fmt.Errorf("decoder embedding backward failed: %w", err)
		}
	}

	// The final hidden gradient is the gradient for the context vector
	d.HiddenState.Grad = hiddenGrad

	return nil
}

// Parameters returns all learnable parameters of the RNNDecoder.
func (d *RNNDecoder) Parameters() []*Tensor {
	params := []*Tensor{}
	params = append(params, d.Embedding.Parameters()...)
	params = append(params, d.Wh.Parameters()...)
	params = append(params, d.Wo.Parameters()...)
	params = append(params, d.Attention.Parameters()...)
	return params
}

// SaveSeq2SeqMoEModelToGOB saves the Seq2SeqMoE to a file in Gob format.
func SaveSeq2SeqMoEModelToGOB(model *Seq2SeqMoE, filePath string) error {
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to create file for saving Seq2SeqMoE model: %w", err)
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	err = encoder.Encode(model)
	if err != nil {
		return fmt.Errorf("failed to encode Seq2SeqMoE model to Gob: %w", err)
	}

	return nil
}

// LoadSeq2SeqMoEModelFromGOB loads a Seq2SeqMoE from a file in Gob format.
func LoadSeq2SeqMoEModelFromGOB(filePath string) (*Seq2SeqMoE, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("error opening Seq2SeqMoE model gob file: %w", err)
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	var loadedModel Seq2SeqMoE
	err = decoder.Decode(&loadedModel)
	if err != nil {
		return nil, fmt.Errorf("error decoding Seq2SeqMoE model from gob: %w", err)
	}

	// Re-initialize layers
	if loadedModel.Decoder != nil {
		if loadedModel.Decoder.Wh != nil {
			whWeightsData := loadedModel.Decoder.Wh.Weights.Data
			whBiasesData := loadedModel.Decoder.Wh.Biases.Data
			whInputDim := loadedModel.Decoder.Wh.Weights.Shape[0]
			whOutputDim := loadedModel.Decoder.Wh.Weights.Shape[1]
			newWh, err := nn.NewLinear(whInputDim, whOutputDim)
			if err != nil {
				return nil, fmt.Errorf("failed to re-initialize decoder Wh layer: %w", err)
			}
			newWh.Weights.Data = whWeightsData
			newWh.Biases.Data = whBiasesData
			loadedModel.Decoder.Wh = newWh
		}

		if loadedModel.Decoder.Wo != nil {
			woWeightsData := loadedModel.Decoder.Wo.Weights.Data
			woBiasesData := loadedModel.Decoder.Wo.Biases.Data
			woInputDim := loadedModel.Decoder.Wo.Weights.Shape[0]
			woOutputDim := loadedModel.Decoder.Wo.Weights.Shape[1]
			newWo, err := nn.NewLinear(woInputDim, woOutputDim)
			if err != nil {
				return nil, fmt.Errorf("failed to re-initialize decoder Wo layer: %w", err)
			}
			newWo.Weights.Data = woWeightsData
			newWo.Biases.Data = woBiasesData
			loadedModel.Decoder.Wo = newWo
		}
	}

	return &loadedModel, nil
}
