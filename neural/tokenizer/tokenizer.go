package tokenizer

import (
	"errors"
	"fmt"
	
	"strings"
	"unicode"

	vocab "github.com/zendrulat/nlptagger/neural/nnu/vocab"
)

// Tokenizer represents a basic tokenizer for the MoE package.
// It should use the Vocabulary from the tagger/vocab package.
type Tokenizer struct {
	// Reference the Vocabulary type from the tagger/vocab package
	Vocabulary *vocab.Vocabulary
}

// NewTokenizer creates a new Tokenizer for the MoE package,
// using a Vocabulary from the tagger/vocab package.
func NewTokenizer(vocabulary *vocab.Vocabulary) (*Tokenizer, error) {
	if vocabulary == nil {
		return nil, errors.New("vocabulary cannot be nil for MoE tokenizer")
	}
	return &Tokenizer{
			Vocabulary: vocabulary,
		},
		nil
}

// Encode converts a string into a slice of token IDs using the tagger/vocab Vocabulary.
// This is a basic implementation using whitespace tokenization.
func (t *Tokenizer) Encode(text string) ([]int, error) {
	if t.Vocabulary == nil {
		return nil, errors.New("tokenizer's vocabulary is not set")
	}

	// Use the more robust TokenizeAndConvertToIDs for JSON-aware tokenization
	tokenIDs, err := TokenizeAndConvertToIDs(text, t.Vocabulary, -1) // -1 for maxLen means no padding/truncation here
	if err != nil {
		return nil, err
	}

	// TokenizeAndConvertToIDs already handles BOS/EOS, so just return its result
	return tokenIDs, nil
}

// Decode converts a slice of token IDs back into a string using the tagger/vocab Vocabulary.
// This version is JSON-aware and does not add extra spaces or punctuation.
func (t *Tokenizer) Decode(tokenIDs []int) (string, error) {
	if t.Vocabulary == nil {
		return "", errors.New("tokenizer's vocabulary is not set")
	}

	var sb strings.Builder
	var lastToken string
	for _, id := range tokenIDs {
		// Assuming BOS/EOS/PAD are handled by the model or by convention
		// and not explicitly skipped here unless they are actual tokens to be ignored.
		// For now, we'll just decode all provided token IDs.

		word := t.Vocabulary.GetWord(id)
		if word == "" {
			word = fmt.Sprintf("[%d]", id)
		}

		// JSON-aware spacing
		if sb.Len() > 0 && !isJSONPunct(lastToken) && !isJSONPunct(word) {
			sb.WriteString(" ")
		}
		sb.WriteString(word)
		lastToken = word
	}

	return sb.String(), nil
}

func isJSONPunct(s string) bool {
	if len(s) != 1 {
		return false
	}
	return strings.ContainsRune(`{}[],:"`, []rune(s)[0])
}

// Tokenize splits a string into words based on whitespace and JSON punctuation.
func Tokenize(text string) []string {
	var tokens []string
	runes := []rune(text)
	i := 0
	for i < len(runes) {
		r := runes[i]

		// Skip whitespace
		if unicode.IsSpace(r) {
			i++
			continue
		}

		// Handle JSON structural characters as single tokens
		if strings.ContainsRune(`{}[],:"`, r) {
			tokens = append(tokens, string(r))
			i++
			continue
		}

		// Handle other tokens (words, numbers, etc.)
		start := i
		for i < len(runes) && !unicode.IsSpace(runes[i]) && !strings.ContainsRune(`{}[],:"`, runes[i]) {
			i++
		}
		token := string(runes[start:i])
		if token != "" {
			tokens = append(tokens, token)
		}
	}
	return tokens
}

// TokenizeAndConvertToIDs converts a string of text into a slice of token IDs
// using the provided vocabulary and pads the sequence to a maximum length.
// This version is more JSON-aware.
func TokenizeAndConvertToIDs(text string, vocab *vocab.Vocabulary, maxLen int) ([]int, error) {
	if vocab == nil {
		return nil, errors.New("vocabulary is nil")
	}

	tokenIDs := []int{}

	// Add the Beginning of Sentence (BOS) token ID at the start
	// Assuming BOS is handled by convention or added during training
	// For now, we'll just add a placeholder if needed.
	// if vocab.BeginningOfSentenceID != -1 {
	// 	tokenIDs = append(tokenIDs, vocab.BeginningOfSentenceID)
	// }

	runes := []rune(text)
	i := 0
	for i < len(runes) {
		r := runes[i]

		// Skip whitespace
		if unicode.IsSpace(r) {
			i++
			continue
		}

		// Handle JSON structural characters as single tokens
		if strings.ContainsRune(`{}[],:"`, r) {
			tokenIDs = append(tokenIDs, vocab.GetTokenID(string(r)))
			i++
			continue
		}

		// Handle other tokens (words, numbers, etc.)
		start := i
		for i < len(runes) && !unicode.IsSpace(runes[i]) && !strings.ContainsRune(`{}[],:"`, runes[i]) {
			i++
		}
		token := string(runes[start:i])
		if token != "" {
			tokenIDs = append(tokenIDs, vocab.GetTokenID(token))
		}

	} // This closes the 'for i < len(runes)' loop.

	// Add the End of Sentence (EOS) token ID at the end (moved outside the loop)
	// Assuming EOS is handled by convention or added during training
	// For now, we'll just add a placeholder if needed.
	// if vocab.EndOfSentenceID != -1 {
	// 	tokenIDs = append(tokenIDs, vocab.EndOfSentenceID)
	// }

	// Padding or Truncating the sequence to maxLen (moved outside the loop)
	if maxLen != -1 { // Only apply padding/truncation if maxLen is not -1
		if len(tokenIDs) > maxLen {
			tokenIDs = tokenIDs[:maxLen]
		} else if len(tokenIDs) < maxLen {
			// Assuming PaddingTokenID is handled by convention (e.g., vocab.GetTokenID("[PAD]"))
			// if vocab.PaddingTokenID != -1 {
				paddingSize := maxLen - len(tokenIDs)
				padding := make([]int, paddingSize)
				for i := range padding {
					padding[i] = vocab.GetTokenID("<pad>") // Use GetTokenID for <pad>
				}
				tokenIDs = append(tokenIDs, padding...)
			// } else {
			// 	
			// }
		}
	}

	return tokenIDs, nil
}
