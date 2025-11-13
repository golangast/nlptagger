package moe

import (
	"encoding/gob"
	"fmt"
	"github.com/zendrulat/nlptagger/neural/nn"
	tn "github.com/zendrulat/nlptagger/neural/tensor"
)

func init() {
	gob.Register(&RNNDecoder{})
}

// RNNDecoder is a simple RNN-based decoder for sequence generation.
type RNNDecoder struct {
	// LSTM layer for recurrent processing
	LSTM *nn.LSTM
	// Linear layer to project LSTM output to vocabulary size
	OutputLayer *nn.Linear
	// Attention layer
	Attention *nn.MultiHeadCrossAttention
	// Output vocabulary size
	OutputVocabSize int
	// Embedding layer for the decoder input
	Embedding *nn.Embedding
	// Initial hidden and cell states for the LSTM
	InitialHiddenState *tn.Tensor
	InitialCellState   *tn.Tensor
	ContextVector      *tn.Tensor
}

// NewRNNDecoder creates a new RNNDecoder.
// inputDim is the dimension of the context vector from the encoder.
// outputVocabSize is the size of the target vocabulary.
func NewRNNDecoder(inputDim, outputVocabSize, embeddingDim, maxAttentionHeads int) (*RNNDecoder, error) {
	// For simplicity, let's assume hiddenDim is the same as inputDim for now.
	hiddenDim := inputDim

	// LSTM input dimension will be embeddingDim + attention output dim
	lstmInputDim := embeddingDim * 2

	// numLayers for LSTM is 1 for a single-layer LSTM
	lstm, err := nn.NewLSTM(lstmInputDim, hiddenDim, 1)
	if err != nil {
		return nil, fmt.Errorf("failed to create LSTM for decoder: %w", err)
	}

	outputLayer, err := nn.NewLinear(hiddenDim, outputVocabSize)
	if err != nil {
		return nil, fmt.Errorf("failed to create output linear layer for decoder: %w", err)
	}

	embedding := nn.NewEmbedding(outputVocabSize, embeddingDim)

	attention, err := nn.NewMultiHeadCrossAttention(embeddingDim, maxAttentionHeads, maxAttentionHeads)
	if err != nil {
		return nil, fmt.Errorf("failed to create attention layer for decoder: %w", err)
	}

	return &RNNDecoder{
		LSTM:              lstm,
		OutputLayer:       outputLayer,
		OutputVocabSize: outputVocabSize,
		Embedding:       embedding,
		Attention:         attention,
	},
	nil
}

// Forward performs the forward pass of the RNNDecoder.
// It takes the context vector from the encoder and the target sequence (for teacher forcing) and generates a sequence of tokens.
func (d *RNNDecoder) Forward(contextVector, targetSequence *tn.Tensor) ([]*tn.Tensor, error) {
	d.ContextVector = contextVector
	batchSize := targetSequence.Shape[0]
	maxSequenceLength := targetSequence.Shape[1]
	hiddenSize := d.LSTM.HiddenSize

	// Initialize hidden and cell states for the LSTM
	// Use the contextVector to initialize the hidden state
	// The contextVector is [batchSize, sequenceLength, embeddingDim]. We need [batchSize, hiddenSize]
	// For now, let's take the mean of the contextVector along the sequence length dimension
	initialHidden, err := contextVector.Mean(1)
	if err != nil {
		return nil, fmt.Errorf("failed to get mean of context vector for initial hidden state: %w", err)
	}

	// If the hiddenSize of LSTM is different from embeddingDim, we need a linear projection
	if initialHidden.Shape[1] != hiddenSize {
		// This case needs a projection layer, which is not currently in the decoder.
		// For simplicity, let's assume hiddenSize == embeddingDim for now, or handle this with a linear layer.
		// For now, we'll just resize if possible, or error if not compatible.
		if initialHidden.Shape[1] > hiddenSize {
			initialHidden, err = initialHidden.Slice(1, 0, hiddenSize)
			if err != nil {
				return nil, fmt.Errorf("failed to slice initial hidden state: %w", err)
			}
		} else if initialHidden.Shape[1] < hiddenSize {
			// Pad with zeros if hiddenSize is larger
			padding := tn.NewTensor([]int{batchSize, hiddenSize - initialHidden.Shape[1]}, make([]float64, batchSize*(hiddenSize-initialHidden.Shape[1])), false)
			initialHidden, err = tn.Concat([]*tn.Tensor{initialHidden, padding}, 1)
			if err != nil {
				return nil, fmt.Errorf("failed to pad initial hidden state: %w", err)
			}
		}
	}

	hiddenState := initialHidden
	cellState := tn.NewTensor([]int{batchSize, hiddenSize}, make([]float64, batchSize*hiddenSize), false) // Initialize cell state to zeros

	// Create a tensor to hold the decoder outputs
	var outputs []*tn.Tensor

	// Start with a start-of-sequence token (e.g., token ID 0)
	for t := 0; t < maxSequenceLength-1; t++ {
		decoderInput, err := targetSequence.Slice(1, t, t+1)
		if err != nil {
			return nil, fmt.Errorf("error slicing target sequence: %w", err)
		}
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
		combinedInput, err := tn.Concat([]*tn.Tensor{embeddedInput, attentionOutput}, 2)
		if err != nil {
			return nil, fmt.Errorf("decoder concat failed: %w", err)
		}

		// Reshape combinedInput from [batchSize, 1, 2 * embeddingDim] to [batchSize, 2 * embeddingDim]
		reshapedCombinedInput, err := combinedInput.Reshape([]int{batchSize, combinedInput.Shape[2]})
		if err != nil {
			return nil, fmt.Errorf("failed to reshape combined input for LSTM: %w", err)
		}

		// Pass through LSTM
		hiddenState, cellState, err = d.LSTM.Forward(reshapedCombinedInput, hiddenState, cellState)
		if err != nil {
			return nil, fmt.Errorf("decoder LSTM forward failed: %w", err)
		}

		// Hidden state to output logits
		outputLogits, err := d.OutputLayer.Forward(hiddenState)
		if err != nil {
			return nil, fmt.Errorf("decoder output layer forward failed: %w", err)
		}

		outputs = append(outputs, outputLogits)
	}

	d.InitialHiddenState = initialHidden // Store the initial hidden state for backward pass
	d.InitialCellState = cellState // Store the final cell state
	return outputs, nil
}

// Backward performs the backward pass of the RNNDecoder.
func (d *RNNDecoder) Backward(grads []*tn.Tensor) error {
	var lstmHiddenGrad *tn.Tensor
	var lstmCellGrad *tn.Tensor

	// Initialize hidden and cell gradients for the LSTM
	// These will accumulate gradients from all timesteps
	lstmHiddenGrad = tn.NewTensor(d.LSTM.Cells[0][0].PrevHidden.Shape, make([]float64, len(d.LSTM.Cells[0][0].PrevHidden.Data)), true)
	lstmCellGrad = tn.NewTensor(d.LSTM.Cells[0][0].PrevCell.Shape, make([]float64, len(d.LSTM.Cells[0][0].PrevCell.Data)), true)

	for t := len(grads) - 1; t >= 0; t-- {
		outputGrad := grads[t]

		// Backpropagate through the output layer
		err := d.OutputLayer.Backward(outputGrad)
		if err != nil {
			return fmt.Errorf("decoder output layer backward failed: %w", err)
		}

		// Accumulate gradients for the LSTM hidden state
		if d.OutputLayer.Input().Grad != nil {
			if lstmHiddenGrad == nil {
				lstmHiddenGrad = d.OutputLayer.Input().Grad
			} else {
				lstmHiddenGrad.Add(d.OutputLayer.Input().Grad)
			}
		}

		// Backpropagate through the LSTM
		err = d.LSTM.Backward(lstmHiddenGrad, lstmCellGrad)
		if err != nil {
			return fmt.Errorf("decoder LSTM backward failed: %w", err)
		}

		// Get gradients from the LSTMCell
		inputGrad := d.LSTM.Cells[0][0].InputTensor.Grad
		prevHiddenGrad := d.LSTM.Cells[0][0].PrevHidden.Grad
		prevCellGrad := d.LSTM.Cells[0][0].PrevCell.Grad

		// Split the gradient for the concatenated input (embeddedInput and attentionOutput)
		splitGrads, err := tn.Split(inputGrad, 1, []int{d.Embedding.DimModel, d.Attention.DimModel})
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

		// Update the hidden and cell gradients for the next timestep
		lstmHiddenGrad = prevHiddenGrad
		lstmCellGrad = prevCellGrad
	}

	// The final hidden gradient is the gradient for the context vector
	// This needs to be handled carefully as the context vector was used to initialize the LSTM hidden state.
	// For now, we'll just set the gradient of the initial hidden state.
	d.InitialHiddenState.Grad = lstmHiddenGrad

	return nil
}

// DecodeStep performs a single decoding step.
// It takes the current input token, the previous hidden and cell states, and the encoder's context vector.
// It returns the output logits for the current step, and the new hidden and cell states.
func (d *RNNDecoder) DecodeStep(inputToken *tn.Tensor, prevHiddenState, prevCellState, contextVector *tn.Tensor) (*tn.Tensor, *tn.Tensor, *tn.Tensor, error) {
	// Embed the decoder input
	embeddedInput, err := d.Embedding.Forward(inputToken)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("decoder embedding failed: %w", err)
	}

	// Apply attention
	attentionOutput, err := d.Attention.Forward(embeddedInput, contextVector, contextVector)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("decoder attention failed: %w", err)
	}

	// Concatenate embedded input and attention output
	combinedInput, err := tn.Concat([]*tn.Tensor{embeddedInput, attentionOutput}, 2)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("decoder concat failed: %w", err)
	}

	// Reshape combinedInput from [batchSize, 1, 2 * embeddingDim] to [batchSize, 2 * embeddingDim]
	batchSize := combinedInput.Shape[0]
	reshapedCombinedInput, err := combinedInput.Reshape([]int{batchSize, combinedInput.Shape[2]})
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to reshape combined input for LSTM: %w", err)
	}

	// Pass through LSTM
	hiddenState, cellState, err := d.LSTM.Forward(reshapedCombinedInput, prevHiddenState, prevCellState)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("decoder LSTM forward failed: %w", err)
	}

	// Hidden state to output logits
	outputLogits, err := d.OutputLayer.Forward(hiddenState)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("decoder output layer forward failed: %w", err)
	}

	return outputLogits, hiddenState, cellState, nil
}

// Parameters returns all learnable parameters of the RNNDecoder.
func (d *RNNDecoder) Parameters() []*tn.Tensor {
	params := []*tn.Tensor{}
	params = append(params, d.Embedding.Parameters()...)
	params = append(params, d.LSTM.Parameters()...)
	params = append(params, d.OutputLayer.Parameters()...)
	params = append(params, d.Attention.Parameters()...)
	return params
}
