package bartsimple

import (
	"errors"
	"fmt"
	"log"     // Added log import for error reporting
	"strings" // Added strings import for joining words
	// You might need to import "math" if your Softmax implementation is here
)

// BartProcessCommand performs a simplified forward pass for inference and generates a summary.
func (m *SimplifiedBARTModel) BartProcessCommand(command string) (string, error) {
	// Simplified: Tokenize input and perform a single forward pass through the encoder and decoder.
	// No caching or autoregressive loop for simplicity in this version.

	// Check if tokenizer and vocabulary are loaded
	if m.Tokenizer == nil {
		return "", errors.New("tokenizer is not loaded in the model")
	}
	if m.Vocabulary == nil {
		return "", errors.New("vocabulary is not loaded in the model")
	}
	if m.Tokenizer == nil {
		return "", errors.New("tokenizer is not loaded in the model")
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

	// Assuming m.Tokenizer.BosID exists and returns int
	bosTokenID := m.Tokenizer.BosID
	// Let's assume the target sequence length is the same as the input sequence length for this simplification
	targetSeqLength := len(inputTokenIDs) // Simplified assumption
	decoderInputIDs := make([]float64, targetSeqLength)
	for i := range decoderInputIDs {
		decoderInputIDs[i] = float64(bosTokenID)
	}
	// Assuming NewTensor function is available
	decoderInputTensor := NewTensor(decoderInputIDs, []int{1, targetSeqLength}, true) // Batch size 1

	// Embed the decoder input
	// Assuming TokenEmbedding.Forward exists and returns *Tensor
	decoderInputEmbeddings, err := m.TokenEmbedding.Forward(decoderInputTensor)
	if err != nil {
		return "", fmt.Errorf("decoder token embedding failed: %w", err)
	}

	// Add positional embeddings to decoder input
	// Assuming PositionalEmbedding.Forward exists and returns *Tensor
	decoderInputWithPos, err := m.PositionalEmbedding.Forward(decoderInputEmbeddings)
	if err != nil {
		return "", fmt.Errorf("decoder positional embedding failed: %w", err)
	}

	// Create dummy masks for the decoder (simplified)
	var selfAttentionMask *Tensor = nil  // Causal mask would be needed in a real autoregressive decoder
	var crossAttentionMask *Tensor = nil // Cross-attention mask based on encoder padding

	// Pass through the simplified decoder layer
	// Assuming Decoder.Layer.Forward exists and returns *Tensor
	decoderOutput, err := m.Decoder.Layer.Forward(decoderInputWithPos, encoderOutput, selfAttentionMask, crossAttentionMask)
	if err != nil {
		return "", fmt.Errorf("simplified decoder layer forward pass failed: %w", err)
	}

	// Before the final linear layer
	// fmt.Printf("Before Final OutputLinear Layer: Shape: %v, First 10: %v\n", decoderOutput.Shape, decoderOutput.Data[:min(10, len(decoderOutput.Data))])

	// Pass decoder output through the final linear layer to get logits
	// Assuming OutputLinear.Forward exists and returns *Tensor
	outputLogits, err := m.OutputLinear.Forward(decoderOutput)
	if err != nil {
		return "", fmt.Errorf("output linear layer failed: %w", err)
	}

	// --- Decoding and Summary Generation Logic ---

	// Assuming outputLogits shape is [batch_size, sequence_length, vocab_size]
	if len(outputLogits.Shape) != 3 {
		return "", fmt.Errorf("expected output logits shape [batch_size, sequence_length, vocab_size], but got %v", outputLogits.Shape)
	}

	batchSize := outputLogits.Shape[0]
	seqLength := outputLogits.Shape[1]
	vocabSize := outputLogits.Shape[2]

	// For simplicity, process the first item in the batch
	if batchSize == 0 {
		return "", errors.New("empty batch in output logits")
	}

	summaryTokens := []int{}
	// Implement greedy decoding
	for s := 0; s < seqLength; s++ { // Iterate through sequence length

		// Get the logits for the current token position across the vocabulary
		tokenLogitsData := make([]float64, vocabSize)
		// Calculate the start index for the current token's logits in the flattened Data slice
		// Assuming outputLogits.Data is flattened in row-major order: batch, sequence, vocab
		startIndex := s * vocabSize // For the first item in the batch (b=0)
		if startIndex+vocabSize > len(outputLogits.Data) {
			log.Printf("Warning: Accessing out of bounds logits data at index %d\n", startIndex)
			break // Prevent panic
		}
		copy(tokenLogitsData, outputLogits.Data[startIndex:startIndex+vocabSize])

		// Apply Softmax to get probabilities (assuming Softmax function is available in bartsimple package)
		// This is the function you need to ensure is defined and exported in your bartsimple package
		probabilities := Softmax(tokenLogitsData) // Assuming Softmax is accessible in this package

		// Greedy decoding: select the token with the highest probability
		maxProb := -1.0
		predictedTokenID := -1
		for i, prob := range probabilities {
			if prob > maxProb {
				maxProb = prob
				predictedTokenID = i
			}
		}

		// Append the predicted token ID
		if predictedTokenID != -1 {
			summaryTokens = append(summaryTokens, predictedTokenID)
			// Optional: Stop decoding if an end-of-sequence token is predicted
			// Assuming m.Vocabulary.EndOfSequenceTokenID exists
			// if m.Vocabulary != nil && predictedTokenID == m.Vocabulary.EndOfSequenceTokenID {
			//     break // Stop decoding at EOS
			// }
		} else {
			log.Println("Warning: Predicted token ID is -1, skipping token.")
		}
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
