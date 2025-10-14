package main

import (
	"fmt"
	"log"

	"nlptagger/neural/nnu/vocab"
)

func Vocabmain() {
	vocabPath := "gob_models/sentence_vocabulary.gob"
	vocabulary, err := vocab.LoadVocabulary(vocabPath)
	if err != nil {
		log.Fatalf("Failed to load vocabulary from %s: %v", vocabPath, err)
	}

	fmt.Printf("Vocabulary loaded from %s\n", vocabPath)
	fmt.Printf("TokenToWord mapping:\n")
	for i, word := range vocabulary.TokenToWord {
		fmt.Printf("  %d: %s\n", i, word)
	}
	fmt.Printf("PaddingTokenID: %d\n", vocabulary.PaddingTokenID)
	fmt.Printf("BosID: %d\n", vocabulary.BosID)
	fmt.Printf("EosID: %d\n", vocabulary.EosID)
	fmt.Printf("UnkID: %d\n", vocabulary.UnkID)
}
