package main

import (
	"encoding/json"
	"fmt"
	"log"

	"nlptagger/neural/nnu/vocab"
	"nlptagger/neural/semantic"
	"nlptagger/neural/tokenizer"
)

func main() {
	// Load semantic output vocabulary
	v, err := vocab.LoadVocabulary("gob_models/semantic_output_vocabulary.gob")
	if err != nil {
		log.Fatal(err)
	}

	// Create tokenizer
	tok, err := tokenizer.NewTokenizer(v)
	if err != nil {
		log.Fatal(err)
	}

	// Create an expected semantic output for "add folder kim with file main.go"
	expectedOutput := semantic.SemanticOutput{
		Operation: "Create",
		TargetResource: semantic.TargetResource{
			Type: "Filesystem::Folder",
			Name: "kim",
			Properties: map[string]interface{}{
				"path": "./",
			},
			Children: []semantic.TargetResource{
				{
					Type: "Filesystem::File",
					Name: "main.go",
				},
			},
		},
		Context: semantic.Context{
			UserRole: "admin",
		},
	}

	// Serialize to JSON
	jsonBytes, err := json.Marshal(expectedOutput)
	if err != nil {
		log.Fatal(err)
	}

	// Add BOS and EOS tokens like in training
	trainingString := "<s> " + string(jsonBytes) + " </s>"

	fmt.Printf("Expected semantic output string:\n%s\n\n", trainingString)

	// Tokenize it
	tokenIDs, err := tok.Encode(trainingString)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Expected token IDs: %v\n\n", tokenIDs)

	// Decode back to verify
	decoded, err := tok.Decode(tokenIDs)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Decoded back: %s\n\n", decoded)

	// Show what the model actually generated
	actualIDs := []int{9, 35, 2, 31, 18, 16, 4, 28, 2, 17, 37, 27, 31}
	actualDecoded, err := tok.Decode(actualIDs)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("What model actually generated:\nIDs: %v\nDecoded: %s\n", actualIDs, actualDecoded)
}
