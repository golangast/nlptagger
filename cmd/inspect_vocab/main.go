package main

import (
	"fmt"
	"log"

	"nlptagger/neural/nnu/vocab"
)

func main() {
	v, err := vocab.LoadVocabulary("gob_models/semantic_output_vocabulary.gob")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Semantic Output Vocabulary size: %d\n\n", len(v.WordToToken))
	fmt.Println("First 40 tokens:")
	for i := 0; i < 40 && i < len(v.TokenToWord); i++ {
		fmt.Printf("ID %2d: %q\n", i, v.TokenToWord[i])
	}
}
