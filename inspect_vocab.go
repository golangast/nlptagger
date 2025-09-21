//go:build ignore

package main

import (
	"fmt"
	"log"

	mainvocab "github.com/golangast/nlptagger/neural/nnu/vocab"
)

func main() {
	// Define paths
	const vocabPath = "gob_models/vocabulary.gob"

	// Setup vocabulary
	vocabulary, err := mainvocab.LoadVocabulary(vocabPath)
	if err != nil {
		log.Fatalf("Failed to set up vocabulary: %v", err)
	}

	fmt.Println("Vocabulary:")
	for i, word := range vocabulary.TokenToWord {
		fmt.Printf("%d: %s\n", i, word)
	}
}
