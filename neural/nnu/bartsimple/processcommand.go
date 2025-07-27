package bartsimple

import (
	"errors"
	"fmt"
	"log"
	"strings"
)

// BartProcessCommand performs an autoregressive forward pass for inference to generate a summary.
func (m *SimplifiedBARTModel) BartProcessCommand(command string) (string, error) {
	// Check if tokenizer and vocabulary are loaded
	if m.Tokenizer == nil {
		return "", errors.New("tokenizer is not loaded in the model")
	}
	if m.Vocabulary == nil {
		return "", errors.New("vocabulary is not loaded in the model")
	}
	if m.Encoder == nil {
		return "", errors.New("encoder is not loaded in the model")
	}
	if m.Decoder == nil {
		return "", errors.New("decoder is not loaded in the model")
	}
	if m.OutputLinear == nil {
		return "", errors.New("output linear layer is not loaded in the model")
	}

	// Tokenize the input command using the model's tokenizer
	// Assuming m.Tokenizer has an Encode method that returns []int and error
	inputTokenIDs, err := m.Tokenizer.Encode(command)
	if err != nil {
		return "", fmt.Errorf("input command tokenization failed: %w", err)
	}

	if len(inputTokenIDs) == 0 {
		return "", errors.New("input command tokenization failed: no tokens generated")
	}
	// Limit input sequence length for simplicity
	if len(inputTokenIDs) > m.MaxSequenceLength {
		inputTokenIDs = inputTokenIDs[:m.MaxSequenceLength]
	}

	// Convert token IDs to a tensor
	inputTensorData := make([]float64, len(inputTokenIDs))
	for i, id := range inputTokenIDs {
		inputTensorData[i] = float64(id)
	}
	// Assuming NewTensor function is available
	inputTensor := NewTensor(inputTensorData, []int{1, len(inputTokenIDs)}, true) // Batch size 1

	// Embed the input tensor
	// Assuming TokenEmbedding.Forward exists and returns *Tensor
	encoderInputEmbeddings, err := m.TokenEmbedding.Forward(inputTensor)
	if err != nil {
		return "", fmt.Errorf("encoder token embedding failed: %w", err)
	}

	// Add positional embeddings
	// Assuming PositionalEmbedding.Forward exists and returns *Tensor
	encoderInputWithPos, err := m.PositionalEmbedding.Forward(encoderInputEmbeddings)
	if err != nil {
		return "", fmt.Errorf("encoder positional embedding failed: %w", err)
	}

	// Create a dummy encoder mask (for simplicity, assuming no padding mask needed for this basic test)
	// In a real scenario, you'd create a mask based on padded tokens.
	var encoderMask *Tensor = nil

	// Pass through the simplified encoder layer
	// Assuming Encoder.Layer.Forward exists and returns *Tensor
	encoderOutput, err := m.Encoder.Layer.Forward(encoderInputWithPos, encoderMask)
	if err != nil {
		return "", fmt.Errorf("simplified encoder layer forward pass failed: %w", err)
	}

	// --- Autoregressive Decoding Loop ---
	summaryTokens := []int{}
	// Start with the beginning-of-sentence token
	decoderInputIDs := []float64{float64(m.Tokenizer.BosID)}

	// Generate tokens one by one up to the max sequence length
	for i := 0; i < m.MaxSequenceLength; i++ {
		// Convert current decoder input IDs to a tensor
		decoderInputTensor := NewTensor(decoderInputIDs, []int{1, len(decoderInputIDs)}, true)

		// Embed the decoder input
		decoderInputEmbeddings, err := m.TokenEmbedding.Forward(decoderInputTensor)
		if err != nil {
			return "", fmt.Errorf("decoder token embedding failed: %w", err)
		}

		// Add positional embeddings
		decoderInputWithPos, err := m.PositionalEmbedding.Forward(decoderInputEmbeddings)
		if err != nil {
			return "", fmt.Errorf("decoder positional embedding failed: %w", err)
		}

		// Create a causal mask for the decoder's self-attention
		// This prevents the model from looking at future tokens during generation.
		selfAttentionMask := createCausalMask(1, len(decoderInputIDs))
		var crossAttentionMask *Tensor = nil // No cross-attention mask for simplicity

		// Pass through the decoder
		decoderOutput, err := m.Decoder.Layer.Forward(decoderInputWithPos, encoderOutput, selfAttentionMask, crossAttentionMask)
		if err != nil {
			return "", fmt.Errorf("decoder layer forward pass failed: %w", err)
		}

		// Pass through the final linear layer to get logits
		outputLogits, err := m.OutputLinear.Forward(decoderOutput)
		if err != nil {
			return "", fmt.Errorf("output linear layer failed: %w", err)
		}

		// --- Greedy Decoding for the *last* token ---
		// Get the logits for the last token in the sequence
		lastTokenLogitsIndex := (outputLogits.Shape[1] - 1) * outputLogits.Shape[2]
		lastTokenLogits := outputLogits.Data[lastTokenLogitsIndex:]

		// Apply Softmax to get probabilities
		probabilities := Softmax(lastTokenLogits)

		// Greedy decoding: select the token with the highest probability
		maxProb := -1.0
		predictedTokenID := -1
		for j, prob := range probabilities {
			if prob > maxProb {
				maxProb = prob
				predictedTokenID = j
			}
		}

		// Stop if we predict the end-of-sequence token
		if predictedTokenID == m.Tokenizer.EosID {
			break
		}

		// Add the predicted token to our summary and to the next decoder input
		summaryTokens = append(summaryTokens, predictedTokenID)
		decoderInputIDs = append(decoderInputIDs, float64(predictedTokenID))
	}

	// Convert token IDs to words and join into a string using the model's vocabulary
	summaryWords := []string{}
	// We already checked for m.Vocabulary != nil at the beginning of the function, but adding another check here for safety
	if m.Vocabulary == nil {
		log.Println("Warning: Vocabulary is unexpectedly nil during word conversion.")
		// Return token IDs as a fallback
		wordStrings := make([]string, len(summaryTokens))
		for i, tokenID := range summaryTokens {
			wordStrings[i] = fmt.Sprintf("%d", tokenID)
		}
		return strings.Join(wordStrings, " "), nil
	}

	for _, tokenID := range summaryTokens {
		// Assuming m.Vocabulary has a GetWordFromTokenID method
		word, found := m.Vocabulary.GetWordFromTokenID(tokenID)
		if !found {
			// Optional: Log a warning here if you want to know which token IDs were not found
			log.Printf("Warning: Token ID %d not found in vocabulary, mapped to [UNK].", tokenID)
		}
		summaryWords = append(summaryWords, word)

	}

	generatedSummary := strings.Join(summaryWords, " ") // Assuming "strings" package imported

	// Print or return the generated summary
	return generatedSummary, nil
}

// createCausalMask creates a causal mask for self-attention.
// The mask is a tensor where future positions are masked out with a large negative number.
func createCausalMask(batchSize, seqLength int) *Tensor {
	maskShape := []int{batchSize, 1, seqLength, seqLength} // Shape for broadcasting
	maskData := make([]float64, batchSize*seqLength*seqLength)
	for b := 0; b < batchSize; b++ {
		for i := 0; i < seqLength; i++ {
			for j := 0; j < seqLength; j++ {
				idx := b*seqLength*seqLength + i*seqLength + j
				if j > i {
					// Mask out future positions
					maskData[idx] = -1e9 // A large negative number
				} else {
					// Allow attention to current and past positions
					maskData[idx] = 0.0
				}
			}
		}
	}
	// The mask itself does not require gradients
	return NewTensor(maskData, maskShape, false)
}
