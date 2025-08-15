package bartsimple

import (
	"errors"
	"fmt"
	"log"
	"strings"
)

// Tokenizer represents a basic tokenizer for the bartsimple package.
// It should use the Vocabulary from the tagger/vocab package.
type Tokenizer struct {
	// Reference the Vocabulary type from the tagger/vocab package
	Vocabulary *Vocabulary
	BosID      int // Beginning of Sentence token ID
	EosID      int // End of Sentence token ID
	PadID      int // Padding token ID
	UnkID      int // Unknown token ID
}

// NewTokenizer creates a new Tokenizer for the bartsimple package,
// using a Vocabulary from the tagger/vocab package.
func NewTokenizer(vocabulary *Vocabulary, bosID, eosID, padID, unkID int) (*Tokenizer, error) {
	if vocabulary == nil {
		return nil, errors.New("vocabulary cannot be nil for bartsimple tokenizer")
	}
	return &Tokenizer{
		Vocabulary: vocabulary,
		BosID:      bosID,
		EosID:      eosID,
		PadID:      padID,
		UnkID:      unkID,
	}, nil
}

// Encode converts a string into a slice of token IDs using the tagger/vocab Vocabulary.
// This is a basic implementation using whitespace tokenization.
func (t *Tokenizer) Encode(text string) ([]int, error) {
	if t.Vocabulary == nil {
		return nil, errors.New("tokenizer's vocabulary is not set")
	}

	tokens := []int{}
	// Add BOS token at the beginning (optional, depending on your model)
	tokens = append(tokens, t.BosID) // Add BOS

	words := strings.Fields(text) // Simple whitespace tokenization
	for _, word := range words {
		// Apply the same cleaning as used for building the vocabulary
		cleanedWord := strings.ToLower(strings.Trim(word, ".,!?;:\"'()[]{}-_")) // Basic cleaning
		if cleanedWord != "" {
			// Call GetTokenID on the Vocabulary from tagger/vocab
			tokenID := t.Vocabulary.GetTokenID(cleanedWord) // This should now work if tagger/vocab.Vocabulary has GetTokenID
			tokens = append(tokens, tokenID)
		}
	}

	// ... rest of the method
	tokens = append(tokens, t.EosID) // Add EOS
	return tokens, nil
}

// Decode converts a slice of token IDs back into a string using the tagger/vocab Vocabulary.
func (t *Tokenizer) Decode(tokenIDs []int) (string, error) {
	if t.Vocabulary == nil {
		return "", errors.New("tokenizer's vocabulary is not set")
	}

	words := []string{}
	for _, id := range tokenIDs {
		// Explicitly handle special tokens first
		if id == t.BosID {
			continue // Skip BOS token
		} else if id == t.EosID {
			break // Stop at EOS token
		} else if id == t.PadID {
			continue // Skip padding tokens
		} else if id == t.UnkID {
			words = append(words, "[UNK]") // Add [UNK] for unknown tokens
		} else {
			// Only attempt to get word from vocabulary if it's not a special token
			word, found := t.Vocabulary.GetWordFromTokenID(id)
			if found {
				words = append(words, word)
			} else {
				words = append(words, fmt.Sprintf("[%d]", id)) // Fallback for unmapped IDs
			}
		}
	}
	return strings.Join(words, " "), nil
}

// TokenizeAndConvertToIDs converts a string of text into a slice of token IDs
// using the provided vocabulary and pads the sequence to a maximum length.
func TokenizeAndConvertToIDs(text string, vocab *Vocabulary, maxLen int) ([]int, error) {
	if vocab == nil {
		return nil, errors.New("vocabulary is nil")
	}

	// 1. Basic Tokenization (split by whitespace)
	// You will likely need a more sophisticated tokenizer for real-world applications
	words := strings.Fields(text)

	// 2. Convert words to token IDs using the vocabulary
	tokenIDs := []int{}

	// Add the Beginning of Sentence (BOS) token ID at the start
	// Ensure your Vocabulary has BOS_ID defined and correctly set.
	if vocab.BeginningOfSentenceID != -1 { // Check if BOS_ID is initialized
		tokenIDs = append(tokenIDs, vocab.BeginningOfSentenceID)
	} else {
		// Handle case where BOS_ID is not set (log a warning or return error)
		log.Println("Warning: BeginningOfSentenceID not set in vocabulary. Skipping BOS token.")
	}

	for _, word := range words {
		// Clean the word (basic cleaning, should match vocabulary building cleaning)
		cleanedWord := strings.ToLower(strings.Trim(word, ".,!?;:\"'()[]{}-_"))

		if cleanedWord != "" {
			// Get the token ID for the cleaned word.
			// Use GetTokenID which returns the UnknownTokenID if the word is not found.
			tokenID := vocab.GetTokenID(cleanedWord)
			tokenIDs = append(tokenIDs, tokenID)
		}
	}

	// Add the End of Sentence (EOS) token ID at the end
	// Ensure your Vocabulary has EOS_ID defined and correctly set.
	if vocab.EndOfSentenceID != -1 { // Check if EOS_ID is initialized
		tokenIDs = append(tokenIDs, vocab.EndOfSentenceID)
	} else {
		log.Println("Warning: EndOfSentenceID not set in vocabulary. Skipping EOS token.")
	}

	// 3. Padding or Truncating the sequence to maxLen
	if len(tokenIDs) > maxLen {
		// Truncate the sequence if it's longer than maxLen
		tokenIDs = tokenIDs[:maxLen]
	} else if len(tokenIDs) < maxLen {
		// Pad the sequence with PaddingTokenID if it's shorter than maxLen
		// Ensure your Vocabulary has PaddingTokenID defined and correctly set.
		if vocab.PaddingTokenID != -1 { // Check if PaddingTokenID is initialized
			paddingSize := maxLen - len(tokenIDs)
			padding := make([]int, paddingSize)
			for i := range padding {
				padding[i] = vocab.PaddingTokenID
			}
			tokenIDs = append(tokenIDs, padding...)
		} else {
			log.Println("Warning: PaddingTokenID not set in vocabulary. Skipping padding.")
		}
	}

	return tokenIDs, nil
}
