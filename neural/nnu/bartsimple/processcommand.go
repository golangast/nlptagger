package bartsimple

import (
	"errors"
	"fmt"
	"log"
	"sort"
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
	inputTensor := NewTensor(inputTensorData, []int{1, len(inputTokenIDs)}, true) // Batch size 1

	// Embed the input tensor
	encoderInputEmbeddings, err := m.TokenEmbedding.Forward(inputTensor)
	if err != nil {
		return "", fmt.Errorf("encoder token embedding failed: %w", err)
	}

	// Add positional embeddings
	encoderInputWithPos, err := m.PositionalEmbedding.Forward(encoderInputEmbeddings)
	if err != nil {
		return "", fmt.Errorf("encoder positional embedding failed: %w", err)
	}

	var encoderMask *Tensor = nil

	// Pass through the simplified encoder layer
	encoderOutput, err := m.Encoder.Layer.Forward(encoderInputWithPos, encoderMask)
	if err != nil {
		return "", fmt.Errorf("simplified encoder layer forward pass failed: %w", err)
	}

	// --- Beam Search Decoding Loop ---
	beamWidth := 5 // Example beam width
	beams := make([]struct {
		tokenIDs []int
		score    float64
	}, 1)
	beams[0] = struct {
		tokenIDs []int
		score    float64
	}{
		tokenIDs: []int{m.Tokenizer.BosID},
		score:    0.0,
	}

	for i := 0; i < m.MaxSequenceLength; i++ {
		allCandidates := make([]struct {
			tokenIDs []int
			score    float64
		}, 0)

		for _, beam := range beams {
			if beam.tokenIDs[len(beam.tokenIDs)-1] == m.Tokenizer.EosID {
				allCandidates = append(allCandidates, beam)
				continue
			}

			decoderInputIDs := make([]float64, len(beam.tokenIDs))
			for j, id := range beam.tokenIDs {
				decoderInputIDs[j] = float64(id)
			}
			decoderInputTensor := NewTensor(decoderInputIDs, []int{1, len(decoderInputIDs)}, true)

			decoderInputEmbeddings, err := m.TokenEmbedding.Forward(decoderInputTensor)
			if err != nil {
				return "", fmt.Errorf("decoder token embedding failed: %w", err)
			}

			decoderInputWithPos, err := m.PositionalEmbedding.Forward(decoderInputEmbeddings)
			if err != nil {
				return "", fmt.Errorf("decoder positional embedding failed: %w", err)
			}

			selfAttentionMask := createCausalMask(1, len(decoderInputIDs))
			var crossAttentionMask *Tensor = nil

			decoderOutput, err := m.Decoder.Layer.Forward(decoderInputWithPos, encoderOutput, selfAttentionMask, crossAttentionMask)
			if err != nil {
				return "", fmt.Errorf("decoder layer forward pass failed: %w", err)
			}

			outputLogits, err := m.OutputLinear.Forward(decoderOutput)
			if err != nil {
				return "", fmt.Errorf("output linear layer failed: %w", err)
			}

			lastTokenLogitsIndex := (outputLogits.Shape[1] - 1) * outputLogits.Shape[2]
			lastTokenLogitsData := outputLogits.Data[lastTokenLogitsIndex:]
			lastTokenLogitsTensor := NewTensor(lastTokenLogitsData, []int{1, 1, len(lastTokenLogitsData)}, false)

			probabilitiesTensor, err := lastTokenLogitsTensor.Softmax(2)
			if err != nil {
				return "", fmt.Errorf("softmax failed: %w", err)
			}

			// Find top-k tokens
			topK := findTopK(probabilitiesTensor.Data, beamWidth)

			for _, token := range topK {
				newCandidate := struct {
					tokenIDs []int
					score    float64
				}{
					tokenIDs: append(append([]int{}, beam.tokenIDs...), token.tokenID),
					score:    beam.score - token.probability, // Use log probability
				}
				allCandidates = append(allCandidates, newCandidate)
			}
		}

		// Sort candidates by score
		sort.Slice(allCandidates, func(i, j int) bool {
			return allCandidates[i].score > allCandidates[j].score
		})

		// Select top-k candidates for the next iteration
		if len(allCandidates) > beamWidth {
			beams = allCandidates[:beamWidth]
		} else {
			beams = allCandidates
		}

		// Check for termination
		allFinished := true
		for _, beam := range beams {
			if beam.tokenIDs[len(beam.tokenIDs)-1] != m.Tokenizer.EosID {
				allFinished = false
				break
			}
		}
		if allFinished {
			break
		}
	}

	// Select the best summary
	bestBeam := beams[0]
	summaryTokens := bestBeam.tokenIDs[1:] // Exclude BOS token
	if len(summaryTokens) > 0 && summaryTokens[len(summaryTokens)-1] == m.Tokenizer.EosID {
		summaryTokens = summaryTokens[:len(summaryTokens)-1] // Exclude EOS token
	}

	// Convert token IDs to words and join into a string using the model's vocabulary
	summaryWords := []string{}
	if m.Vocabulary == nil {
		log.Println("Warning: Vocabulary is unexpectedly nil during word conversion.")
		wordStrings := make([]string, len(summaryTokens))
		for i, tokenID := range summaryTokens {
			wordStrings[i] = fmt.Sprintf("%d", tokenID)
		}
		return strings.Join(wordStrings, " "), nil
	}

	for _, tokenID := range summaryTokens {
		word, found := m.Vocabulary.GetWordFromTokenID(tokenID)
		if !found {
			log.Printf("Warning: Token ID %d not found in vocabulary, mapped to [UNK].", tokenID)
		}
		summaryWords = append(summaryWords, word)

	}

	generatedSummary := strings.Join(summaryWords, " ")

	return generatedSummary, nil
}

// createCausalMask creates a causal mask for self-attention.
// The mask is a tensor where future positions are masked out with a large negative number.

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

// findTopK finds the top-k tokens with the highest probabilities.
func findTopK(probabilities []float64, k int) []struct {
	tokenID     int
	probability float64
} {
	type tokenProb struct {
		tokenID     int
		probability float64
	}

	probs := make([]tokenProb, len(probabilities))
	for i, p := range probabilities {
		probs[i] = tokenProb{tokenID: i, probability: p}
	}

	sort.Slice(probs, func(i, j int) bool {
		return probs[i].probability > probs[j].probability
	})

	if k > len(probs) {
		k = len(probs)
	}

	topK := make([]struct {
		tokenID     int
		probability float64
	}, k)
	for i := 0; i < k; i++ {
		topK[i] = struct {
			tokenID     int
			probability float64
		}{tokenID: probs[i].tokenID, probability: probs[i].probability}
	}

	return topK
}

