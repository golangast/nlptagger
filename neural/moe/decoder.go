package moe

import (
	"encoding/gob"
	"fmt"
	"math/rand"
	"nlptagger/neural/nn"
	. "nlptagger/neural/tensor"
)

func init() {
	gob.Register(&RNNDecoder{})
}

// RNNDecoder is a simple RNN-based decoder for sequence generation.
type RNNDecoder struct {
	// LSTM layer for recurrent processing
	LSTM *nn.LSTM
	// Layer normalization after LSTM
	LayerNorm *nn.LayerNorm
	// Linear layer to project LSTM output to vocabulary size
	OutputLayer *nn.Linear
	// Output vocabulary size
	OutputVocabSize int
	// Embedding layer for the decoder input
	Embedding         *nn.Embedding
	MaxAttentionHeads int
	Attention         *nn.MultiHeadCrossAttention
	// Initial hidden and cell states for the LSTM
	InitialHiddenState *Tensor
	InitialCellState   *Tensor

	// Intermediate states for BPTT (not serialized)
	hiddenStates     []*Tensor // Hidden state at each timestep
	cellStates       []*Tensor // Cell state at each timestep
	embeddedInputs   []*Tensor // Embedded inputs at each timestep
	attentionOutputs []*Tensor // Attention outputs at each timestep
	combinedInputs   []*Tensor // Combined inputs to LSTM at each timestep
	decoderInputs    []*Tensor // Decoder inputs at each timestep
}

// NewRNNDecoder creates a new RNNDecoder.
// inputDim is the dimension of the context vector from the encoder.
// outputVocabSize is the size of the target vocabulary.
// hiddenSize is the hidden dimension of the LSTM.
// numLayers is the number of LSTM layers.
// dropoutRate is the dropout rate between LSTM layers.
func NewRNNDecoder(inputDim, outputVocabSize, hiddenSize, maxAttentionHeads, numLayers int, dropoutRate float64) (*RNNDecoder, error) {
	// LSTM input dimension will be embeddingDim + attentionOutputDim
	// Assuming attentionOutputDim is also inputDim for simplicity in this setup
	lstmInputDim := inputDim + inputDim // embeddedInput + attentionOutput

	// Create multi-layer LSTM with dropout
	lstm, err := nn.NewLSTM(lstmInputDim, hiddenSize, numLayers)
	if err != nil {
		return nil, fmt.Errorf("failed to create LSTM for decoder: %w", err)
	}
	lstm.DropoutRate = dropoutRate
	lstm.Training = true // Will be set to false during inference

	// Create layer normalization
	layerNorm := nn.NewLayerNorm(hiddenSize)

	outputLayer, err := nn.NewLinear(hiddenSize, outputVocabSize)
	if err != nil {
		return nil, fmt.Errorf("failed to create output linear layer for decoder: %w", err)
	}

	embedding := nn.NewEmbedding(outputVocabSize, inputDim)

	attention, err := nn.NewMultiHeadCrossAttention(inputDim, maxAttentionHeads, maxAttentionHeads)
	if err != nil {
		return nil, fmt.Errorf("failed to create multi-head attention for decoder: %w", err)
	}

	return &RNNDecoder{
			LSTM:              lstm,
			LayerNorm:         layerNorm,
			OutputLayer:       outputLayer,
			OutputVocabSize:   outputVocabSize,
			Embedding:         embedding,
			MaxAttentionHeads: maxAttentionHeads,
			Attention:         attention,
		},
		nil
}

// Forward performs the forward pass of the RNNDecoder.
// It takes the context vector from the encoder and the target sequence (for teacher forcing) and generates a sequence of tokens.
// scheduledSamplingProb: probability of using model's own prediction instead of ground truth (0.0 = pure teacher forcing, 1.0 = pure sampling)
func (d *RNNDecoder) Forward(contextVector, targetSequence *Tensor, scheduledSamplingProb float64) ([]*Tensor, error) {
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
			padding := NewTensor([]int{batchSize, hiddenSize - initialHidden.Shape[1]}, make([]float64, batchSize*(hiddenSize-initialHidden.Shape[1])), false)
			initialHidden, err = Concat([]*Tensor{initialHidden, padding}, 1)
			if err != nil {
				return nil, fmt.Errorf("failed to pad initial hidden state: %w", err)
			}
		}
	}

	hiddenState := initialHidden
	cellState := NewTensor([]int{batchSize, hiddenSize}, make([]float64, batchSize*hiddenSize), false) // Initialize cell state to zeros

	// Create a tensor to hold the decoder outputs
	var outputs []*Tensor

	// Initialize intermediate state storage for BPTT
	d.hiddenStates = make([]*Tensor, 0, maxSequenceLength-1)
	d.cellStates = make([]*Tensor, 0, maxSequenceLength-1)
	d.embeddedInputs = make([]*Tensor, 0, maxSequenceLength-1)
	d.attentionOutputs = make([]*Tensor, 0, maxSequenceLength-1)
	d.combinedInputs = make([]*Tensor, 0, maxSequenceLength-1)
	d.decoderInputs = make([]*Tensor, 0, maxSequenceLength-1)

	// Start with the first token of the target sequence (teacher forcing)
	// We assume targetSequence starts with SOS.
	decoderInput, err := targetSequence.Slice(1, 0, 1)
	if err != nil {
		return nil, fmt.Errorf("failed to slice initial decoder input: %w", err)
	}

	// Loop up to maxSequenceLength - 1 because we predict the next token
	for t := 0; t < maxSequenceLength-1; t++ {
		// Scheduled sampling: decide between ground truth and model prediction for NEXT iteration
		// This decision is made at the END of the current iteration
		// For now, just use the current decoderInput

		// Store decoder input for this timestep (BEFORE any modifications)
		d.decoderInputs = append(d.decoderInputs, decoderInput)

		// Embed the decoder input
		embeddedInput, err := d.Embedding.Forward(decoderInput)
		if err != nil {
			return nil, fmt.Errorf("decoder embedding failed: %w", err)
		}
		d.embeddedInputs = append(d.embeddedInputs, embeddedInput)

		// Apply attention
		attentionOutput, err := d.Attention.Forward(embeddedInput, contextVector, contextVector)
		if err != nil {
			return nil, fmt.Errorf("decoder attention failed: %w", err)
		}
		d.attentionOutputs = append(d.attentionOutputs, attentionOutput)

		// Concatenate embedded input and attention output
		combinedInput, err := Concat([]*Tensor{embeddedInput, attentionOutput}, 1)
		if err != nil {
			return nil, fmt.Errorf("decoder concat failed: %w", err)
		}

		// Reshape combinedInput from [batchSize, 2, embeddingDim] to [batchSize, 2 * embeddingDim]
		reshapedCombinedInput, err := combinedInput.Reshape([]int{batchSize, combinedInput.Shape[1] * combinedInput.Shape[2]})
		if err != nil {
			return nil, fmt.Errorf("failed to reshape combined input for LSTM: %w", err)
		}
		d.combinedInputs = append(d.combinedInputs, reshapedCombinedInput)

		// Pass through LSTM
		hiddenState, cellState, err = d.LSTM.Forward(reshapedCombinedInput, hiddenState, cellState)
		if err != nil {
			return nil, fmt.Errorf("decoder LSTM forward failed: %w", err)
		}

		// Store hidden and cell states for this timestep
		d.hiddenStates = append(d.hiddenStates, hiddenState)
		d.cellStates = append(d.cellStates, cellState)

		// Apply layer normalization (if available - for backward compatibility)
		var normalizedHidden *Tensor
		if d.LayerNorm != nil {
			var err error
			normalizedHidden, err = d.LayerNorm.Forward(hiddenState)
			if err != nil {
				return nil, fmt.Errorf("decoder layer norm forward failed: %w", err)
			}
		} else {
			// For backward compatibility with models saved before LayerNorm
			normalizedHidden = hiddenState
		}

		// Hidden state to output logits
		outputLogits, err := d.OutputLayer.Forward(normalizedHidden)
		if err != nil {
			return nil, fmt.Errorf("decoder output layer forward failed: %w", err)
		}

		outputs = append(outputs, outputLogits)

		// Prepare input for NEXT timestep (scheduled sampling decision)
		if t < maxSequenceLength-2 { // Don't need next input for last timestep
			useModelPrediction := rand.Float64() < scheduledSamplingProb

			if useModelPrediction {
				// Use model's own prediction (scheduled sampling)
				argmax, err := outputLogits.Argmax(1)
				if err != nil {
					return nil, fmt.Errorf("argmax failed during scheduled sampling: %w", err)
				}
				// Use the predicted token IDs as the next input
				decoderInput, err = argmax.Reshape([]int{batchSize, 1})
				if err != nil {
					return nil, fmt.Errorf("failed to reshape argmax: %w", err)
				}
			} else {
				// Use ground truth (teacher forcing)
				slicedTensor, err := targetSequence.Slice(1, t+1, t+2)
				if err != nil {
					return nil, fmt.Errorf("error slicing target sequence: %w", err)
				}
				decoderInput = slicedTensor
			}
		}
	}

	d.InitialHiddenState = initialHidden
	d.InitialCellState = cellState

	return outputs, nil
}

// Backward performs the backward pass of the RNNDecoder with proper BPTT.
func (d *RNNDecoder) Backward(grads []*Tensor) error {
	if len(grads) != len(d.hiddenStates) {
		return fmt.Errorf("gradient length (%d) doesn't match number of timesteps (%d)", len(grads), len(d.hiddenStates))
	}

	numTimesteps := len(grads)
	batchSize := grads[0].Shape[0]
	hiddenSize := d.LSTM.HiddenSize

	// Initialize gradients for hidden and cell states (will accumulate from future timesteps)
	nextHiddenGrad := NewTensor([]int{batchSize, hiddenSize}, make([]float64, batchSize*hiddenSize), false)
	nextCellGrad := NewTensor([]int{batchSize, hiddenSize}, make([]float64, batchSize*hiddenSize), false)

	// Backpropagate through time (from last timestep to first)
	for t := numTimesteps - 1; t >= 0; t-- {
		// 1. Backprop through output layer
		err := d.OutputLayer.Backward(grads[t])
		if err != nil {
			return fmt.Errorf("decoder output layer backward at t=%d failed: %w", t, err)
		}

		// Get gradient w.r.t. hidden state from output layer
		hiddenGrad := d.OutputLayer.Input().Grad

		// Add gradient from future timestep
		hiddenGrad.Add(nextHiddenGrad)

		// 2. Backprop through LSTM
		// We need to set up the LSTM cell state to match this timestep
		// The LSTM.Backward expects the current cell's state to be set
		if t > 0 {
			d.LSTM.Cells[0][0].PrevHidden = d.hiddenStates[t-1]
			d.LSTM.Cells[0][0].PrevCell = d.cellStates[t-1]
		} else {
			d.LSTM.Cells[0][0].PrevHidden = d.InitialHiddenState
			d.LSTM.Cells[0][0].PrevCell = d.InitialCellState
		}
		d.LSTM.Cells[0][0].InputTensor = d.combinedInputs[t]

		err = d.LSTM.Backward(hiddenGrad, nextCellGrad)
		if err != nil {
			return fmt.Errorf("decoder LSTM backward at t=%d failed: %w", t, err)
		}

		// Get gradients for next iteration
		inputGrad := d.LSTM.Cells[0][0].InputTensor.Grad
		if t > 0 {
			nextHiddenGrad = d.LSTM.Cells[0][0].PrevHidden.Grad
			nextCellGrad = d.LSTM.Cells[0][0].PrevCell.Grad
		}

		// 3. Backprop through reshape (gradient flows straight through)
		// inputGrad is already the right shape [batchSize, 2*embeddingDim]

		// 4. Backprop through concat - split the gradient
		embeddingDim := d.Embedding.DimModel
		splitGrads, err := Split(inputGrad, 1, []int{embeddingDim, embeddingDim})
		if err != nil {
			return fmt.Errorf("decoder split at t=%d failed: %w", t, err)
		}
		embeddedGrad := splitGrads[0]
		attentionGrad := splitGrads[1]

		// 5. Backprop through attention
		// Reshape attention gradient to 3D
		reshapedAttentionGrad, err := attentionGrad.Reshape([]int{batchSize, 1, embeddingDim})
		if err != nil {
			return fmt.Errorf("failed to reshape attention gradient at t=%d: %w", t, err)
		}

		err = d.Attention.Backward(reshapedAttentionGrad)
		if err != nil {
			return fmt.Errorf("decoder attention backward at t=%d failed: %w", t, err)
		}

		// 6. Backprop through embedding
		// Manually set the input for this timestep because Embedding only remembers the last one
		d.Embedding.SetInput(d.decoderInputs[t])
		err = d.Embedding.Backward(embeddedGrad)
		if err != nil {
			return fmt.Errorf("decoder embedding backward at t=%d failed: %w", t, err)
		}
	}

	// Store gradient for initial hidden state (gradient w.r.t. context vector)
	if d.InitialHiddenState != nil {
		d.InitialHiddenState.Grad = nextHiddenGrad
	}

	return nil
}

// DecodeStep performs a single decoding step.
// It takes the current input token, the previous hidden and cell states, and the encoder's context vector.
// It returns the output logits for the current step, and the new hidden and cell states.
func (d *RNNDecoder) DecodeStep(inputToken *Tensor, prevHiddenState, prevCellState, contextVector *Tensor) (*Tensor, *Tensor, *Tensor, error) {
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
	combinedInput, err := Concat([]*Tensor{embeddedInput, attentionOutput}, 1)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("decoder concat failed: %w", err)
	}

	// Reshape combinedInput from [batchSize, 2, embeddingDim] to [batchSize, 2 * embeddingDim]
	batchSize := combinedInput.Shape[0]
	reshapedCombinedInput, err := combinedInput.Reshape([]int{batchSize, combinedInput.Shape[1] * combinedInput.Shape[2]})
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
func (d *RNNDecoder) Parameters() []*Tensor {
	params := []*Tensor{}
	params = append(params, d.Embedding.Parameters()...)
	params = append(params, d.LSTM.Parameters()...)
	if d.LayerNorm != nil {
		params = append(params, d.LayerNorm.Parameters()...)
	}
	params = append(params, d.OutputLayer.Parameters()...)
	params = append(params, d.Attention.Parameters()...)
	return params
}
