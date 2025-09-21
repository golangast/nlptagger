package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"time"

	moemodel "nlptagger/neural/moe/model"
	"nlptagger/neural/tensor"
	"nlptagger/neural/tokenizer"
	vocabbert "nlptagger/neural/nnu/bert/vocab"
)

var (
	moeInferenceQuery = flag.String("moe_inference_query", "", "Natural language query for MoE inference")
)

func main() {
	// Seed the random number generator
	rand.Seed(time.Now().UnixNano())

	flag.Parse()

	// Define paths
	const vocabPath = "gob_models/vocabulary.gob"
	const moeModelPath = "gob_models/moe_model.gob"

	vocabulary, err := vocabbert.SetupVocabulary(vocabPath, []string{}) // No specific training data for vocab setup here
	if err != nil {
		log.Fatalf("Failed to set up vocabulary: %v", err)
	}

	// Load MoE model
	moeModel, err := moemodel.LoadMoEClassificationModelFromGOB(moeModelPath)
	if err != nil {
		log.Fatalf("Failed to load MoE model: %v", err)
	}

	// Create tokenizer
	tokenizer, err := tokenizer.NewTokenizer(vocabulary)
	if err != nil {
		log.Fatalf("Failed to create tokenizer: %v", err)
	}

	if *moeInferenceQuery != "" {
		log.Printf("Running MoE inference for query: \"%s\"", *moeInferenceQuery)

		// Tokenize the input query
		tokenIDs, err := tokenizer.Encode(*moeInferenceQuery)
		if err != nil {
			log.Fatalf("Failed to tokenize query: %v", err)
		}

		// Create input tensor for MoE model
		inputTensor := tensor.NewTensor([]int{1, len(tokenIDs)}, make([]float64, len(tokenIDs)), true)
		for i, id := range tokenIDs {
			inputTensor.Data[i] = float64(id)
		}

		// Run MoE model forward pass
		parentLogits, childLogits, err := moeModel.Forward(inputTensor)
		if err != nil {
			log.Fatalf("MoE model forward pass failed: %v", err)
		}

		// Process parent logits
		if parentLogits == nil || len(parentLogits.Data) == 0 {
			log.Fatalf("MoE model returned empty parent logits.")
		}
		parentIntentID := 0
		maxParentLogit := parentLogits.Data[0]
		for i := 1; i < len(parentLogits.Data); i++ {
			if parentLogits.Data[i] > maxParentLogit {
				maxParentLogit = parentLogits.Data[i]
				parentIntentID = i
			}
		}
		parentIntent := moeModel.ParentVocabulary.IDToWord[parentIntentID]

		// Process child logits
		if childLogits == nil || len(childLogits.Data) == 0 {
			log.Fatalf("MoE model returned empty child logits.")
		}
		childIntentID := 0
		maxChildLogit := childLogits.Data[0]
		for i := 1; i < len(childLogits.Data); i++ {
			if childLogits.Data[i] > maxChildLogit {
				maxChildLogit = childLogits.Data[i]
				childIntentID = i
			}
		}
		childIntent := moeModel.ChildVocabulary.IDToWord[childIntentID]

		fmt.Printf("\n--- MoE Inference Output ---\n")
		fmt.Printf("Parent Intent: %s\n", parentIntent)
		fmt.Printf("Child Intent: %s\n", childIntent)
		fmt.Println("-----------------------------\n")
		return
	}

	log.Println("No inference query provided. Use -moe_inference_query <query>.")
}
