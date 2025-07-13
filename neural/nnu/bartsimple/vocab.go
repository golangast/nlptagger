package bartsimple // Assuming your vocabulary package is named vocab

import (
	"encoding/gob" // Import errors for returning errors
	"errors"
	"fmt"
	"log" // Import log for warnings in GetTokenID
	"os"
	"strings"
)

// Vocabulary represents the mapping between words and token IDs.
// This is the authoritative definition for your project's vocabulary.
type Vocabulary struct {
	WordToToken map[string]int // Maps words (string) to token IDs (int)
	TokenToWord map[int]string // Maps token IDs (int) to words (string)
	Size        int            // The current size of the vocabulary (number of unique tokens)

	// Special tokens (adjust names and values as needed)
	UnknownTokenID        int
	PaddingTokenID        int
	BeginningOfSentenceID int
	EndOfSentenceID       int
}

// NewVocabulary creates an empty Vocabulary with initialized maps.
func NewVocabulary() *Vocabulary {
	return &Vocabulary{
		WordToToken: make(map[string]int),
		TokenToWord: make(map[int]string),
		Size:        0, // Start with size 0
		// Assign default values for special token IDs, or initialize them
		// when building from a corpus or loading from a file.
		// For now, let's use placeholder values or a convention.
		// It's better to assign these based on your vocabulary building process.
		UnknownTokenID:        0, // Placeholder, should be assigned a real ID
		PaddingTokenID:        0, // Placeholder
		BeginningOfSentenceID: 0, // Placeholder
		EndOfSentenceID:       0, // Placeholder
	}
}

// AddToken adds a word to the vocabulary and assigns a given token ID.
// It returns an error if the word already exists with a different token ID
// or if the token ID is already assigned to a different word.
func (v *Vocabulary) AddToken(word string, tokenID int) error {
	if existingID, exists := v.WordToToken[word]; exists {
		if existingID != tokenID {
			return fmt.Errorf("word '%s' already exists with token ID %d, cannot add with ID %d", word, existingID, tokenID)
		}
		return nil // Word already exists with the same ID
	}
	if existingWord, exists := v.TokenToWord[tokenID]; exists {
		return fmt.Errorf("token ID %d is already assigned to word '%s', cannot add '%s'", tokenID, existingWord, word)
	}
	v.WordToToken[word] = tokenID
	v.TokenToWord[tokenID] = word
	return nil
}

// // AddToken adds a word to the vocabulary and assigns a new token ID if it's not already present.
// func (v *Vocabulary) AddToken(word string) {
// 	if _, exists := v.WordToToken[word]; !exists {
// 		tokenID := v.Size // Assign the next available ID
// 		v.WordToToken[word] = tokenID
// 		v.TokenToWord[tokenID] = word
// 		v.Size++
// 	}
// }

// GetTokenID retrieves the token ID for a given word.
// Returns the UnknownTokenID if the word is not in the vocabulary.
func (v *Vocabulary) GetTokenID(word string) int {
	// Check if the vocabulary map is initialized

	if v == nil || v.WordToToken == nil {
		log.Println("Warning: Vocabulary or WordToToken map is not initialized. Returning UnknownTokenID.")
		// Return a default or a sentinel value if UnknownTokenID is not set
		if v != nil {
			return v.UnknownTokenID
		}
		return 0 // Fallback if v is nil
	}

	// Look up the word in the map
	if id, ok := v.WordToToken[word]; ok {
		return id // Return the token ID if found
	}
	return 102 // <--- Change this to a valid index (0-102)

}

// In your vocab.go file, replace the existing GetWordFromTokenID:

// GetWordFromTokenID retrieves the word for a given token ID.
// It returns the word string and a boolean indicating if the token was found.
// If the token ID is not found, it returns the [UNK] token string and false.
func (v *Vocabulary) GetWordFromTokenID(tokenID int) (string, bool) {
	// Check if the vocabulary map is initialized
	if v == nil || v.TokenToWord == nil {
		// Log a warning, but return a default placeholder since we can't even look up [UNK]
		log.Println("Warning: Vocabulary or TokenToWord map is not initialized in GetWordFromTokenID.")
		return "[UNK]", false // Return a hardcoded placeholder and false
	}

	// Attempt to find the word for the given tokenID
	if word, ok := v.TokenToWord[tokenID]; ok {
		return word, true // Token found, return the word and true
	}

	// Token ID was not found in the vocabulary.
	// We need to return the string for the [UNK] token.
	// First, try to get the [UNK] word using its dedicated ID from the Vocabulary struct.
	if unkWord, ok := v.TokenToWord[v.UnknownTokenID]; ok {
		// Return the actual string associated with the UnknownTokenID if it exists in the map
		return unkWord, false // Return the [UNK] word and false (because the original tokenID wasn't found)
	}

	// Fallback: If the UnknownTokenID itself is not found in the map (e.g., Vocabulary wasn't built correctly)
	// Return a hardcoded placeholder string.
	log.Printf("Warning: UnknownTokenID (%d) not found in vocabulary map. Using hardcoded [UNK] string.", v.UnknownTokenID)
	return "[UNK]", false // Return a hardcoded placeholder and false
}

// BuildFromCorpus builds the vocabulary from a corpus of text.
// This is a basic implementation; real-world tokenization and vocabulary
// building would be more complex (handling punctuation, case, etc.).
func (v *Vocabulary) BuildFromCorpus(corpus string) {
	// Reset vocabulary for building from scratch
	v.WordToToken = make(map[string]int)
	v.TokenToWord = make(map[int]string)
	v.Size = 0

	// Add special tokens first and assign their IDs
	// You might want to use constants for special token strings
	v.AddToken("[UNK]", len(v.TokenToWord)) // Unknown
	v.UnknownTokenID = v.WordToToken["[UNK]"]

	v.AddToken("[PAD]", len(v.TokenToWord)) // Padding
	v.PaddingTokenID = v.WordToToken["[PAD]"]

	v.AddToken("[BOS]", len(v.TokenToWord)) // Beginning of Sentence
	v.BeginningOfSentenceID = v.WordToToken["[BOS]"]

	v.AddToken("[EOS]", len(v.TokenToWord)) // End of Sentence
	v.EndOfSentenceID = v.WordToToken["[EOS]"]

	// Simple whitespace tokenization for building vocabulary
	// You'll likely need a more robust tokenizer here
	words := strings.Fields(corpus)
	for _, word := range words {
		// You might want to perform basic text cleaning here (lowercase, punctuation removal)
		cleanedWord := strings.ToLower(strings.Trim(word, ".,!?;:\"'()[]{}-_")) // Basic cleaning
		if cleanedWord != "" {
			v.AddToken(cleanedWord, len(v.TokenToWord)) // Add the cleaned word
		}
	}
}

// Save saves the vocabulary to a file in Gob format.
func (v *Vocabulary) Save(filePath string) error {

	// 1. Check if the file exists
	_, err := os.Stat(filePath)
	if err == nil {
		// File exists, proceed to delete
		err = os.Remove(filePath)
		if err != nil {
			log.Fatalf("Error deleting file '%s': %v\n", filePath, err)
		}
	} else if !errors.Is(err, os.ErrNotExist) {
		// Handle other potential errors during Stat
		log.Fatalf("Error checking file existence for '%s': %v\n", filePath, err)
	}

	// 2. Create a new file (or truncate and open if it existed)
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to create vocabulary file: %w", err)
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	if err := encoder.Encode(v); err != nil {
		return fmt.Errorf("failed to encode vocabulary: %w", err)
	}
	return nil
}

// LoadVocabulary loads the vocabulary from a file in Gob format.
func LoadVocabulary(filePath string) (*Vocabulary, error) {
	file, err := os.Open(filePath)
	if err != nil {
		// If the file doesn't exist, return a specific error indicating that
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("vocabulary file not found at %s", filePath)
		}
		return nil, fmt.Errorf("failed to open vocabulary file: %w", err)
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	var v Vocabulary
	if err := decoder.Decode(&v); err != nil {
		return nil, fmt.Errorf("failed to decode vocabulary: %w", err)
	}
	return &v, nil
}
