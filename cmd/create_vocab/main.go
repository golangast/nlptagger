package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"strings"

	"nlptagger/neural/nnu/vocab"
)

type IntentTrainingExample struct {
	Query        string `json:"query"`
	ParentIntent string `json:"parent_intent"`
	ChildIntent  string `json:"child_intent"`
	Description  string `json:"description"`
	Sentence     string `json:"sentence"`
}

func main() {
	// Define paths
	const trainingDataPath = "trainingdata/intent_data.json"
	const vocabPath = "gob_models/query_vocabulary.gob"

	// Load training data
	file, err := ioutil.ReadFile(trainingDataPath)
	if err != nil {
		log.Fatalf("Failed to read training data: %v", err)
	}

	var intents []IntentTrainingExample
	if err := json.Unmarshal(file, &intents); err != nil {
		log.Fatalf("Failed to unmarshal training data: %v", err)
	}

	// Create and save vocabulary
	tokenVocab := vocab.NewVocabulary()
	for _, intent := range intents {
		tokens := strings.Fields(strings.ToLower(intent.Query))
		for _, token := range tokens {
			tokenVocab.AddToken(token)
		}
	}

	err = tokenVocab.Save(vocabPath)
	if err != nil {
		log.Fatalf("Failed to save vocabulary: %v", err)
	}

	fmt.Println("vocabs", tokenVocab.TokenToWord)
	fmt.Println("vocab size", tokenVocab.Size())

	fmt.Println("Vocabulary created and saved successfully.")
}
