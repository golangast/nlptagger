package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"

	"nlptagger/neural/moe"
	mainvocab "nlptagger/neural/nnu/vocab"
	"nlptagger/neural/tensor"
	"nlptagger/neural/tokenizer"
)

var (
	query        = flag.String("query", "", "Query for MoE inference")
	maxSeqLength = flag.Int("maxlen", 32, "Maximum sequence length")
)

func main() {
	rand.Seed(1) // Seed the random number generator for deterministic behavior
	flag.Parse()

	if *query == "" {
		log.Fatal("Please provide a query using the -query flag.")
	}

	// Define paths
	const vocabPath = "gob_models/query_vocabulary.gob"
	const moeModelPath = "gob_models/moe_classification_model.gob"
	const semanticOutputVocabPath = "gob_models/semantic_output_vocabulary.gob"

	// Load vocabularies
	vocabulary, err := mainvocab.LoadVocabulary(vocabPath)
	if err != nil {
		log.Fatalf("Failed to set up input vocabulary: %v", err)
	}

	semanticOutputVocabulary, err := mainvocab.LoadVocabulary(semanticOutputVocabPath)
	if err != nil {
		log.Fatalf("Failed to set up semantic output vocabulary: %v", err)
	}

	// Create tokenizer
	tok, err := tokenizer.NewTokenizer(vocabulary)
	if err != nil {
		log.Fatalf("Failed to create tokenizer: %v", err)
	}

	semanticOutputTokenizer, err := tokenizer.NewTokenizer(semanticOutputVocabulary)
	if err != nil {
		log.Fatalf("Failed to create semantic output tokenizer: %v", err)
	}

	// Load the trained MoEClassificationModel model
	model, err := moe.LoadIntentMoEModelFromGOB(moeModelPath)
	if err != nil {
		log.Fatalf("Failed to load MoE model: %v", err)
	}

	log.Printf("Running MoE inference for query: \"%s\"", *query)

	// Encode the query
	tokenIDs, err := tok.Encode(*query)
	if err != nil {
		log.Fatalf("Failed to encode query: %v", err)
	}

	// Pad or truncate the sequence to a fixed length
	if len(tokenIDs) > *maxSeqLength {
		tokenIDs = tokenIDs[:*maxSeqLength] // Truncate from the end
	} else {
		for len(tokenIDs) < *maxSeqLength {
			tokenIDs = append(tokenIDs, vocabulary.PaddingTokenID) // Appends padding
		}
	}
	inputData := make([]float64, len(tokenIDs))
	for i, id := range tokenIDs {
		inputData[i] = float64(id)
	}
	inputTensor := tensor.NewTensor([]int{1, len(inputData)}, inputData, false) // RequiresGrad=false for inference

	// Create a dummy target tensor for inference, as the Forward method expects two inputs.
	dummyTargetTokenIDs := make([]float64, *maxSeqLength)
	for i := 0; i < *maxSeqLength; i++ {
		dummyTargetTokenIDs[i] = float64(vocabulary.PaddingTokenID)
	}
	dummyTargetTensor := tensor.NewTensor([]int{1, *maxSeqLength}, dummyTargetTokenIDs, false)

	// Forward pass to get the context vector
	_, contextVector, err := model.Forward(inputTensor, dummyTargetTensor)
	if err != nil {
		log.Fatalf("MoE model forward pass failed: %v", err)
	}

	// Greedy search decode to get the predicted token IDs
	predictedIDs, err := model.GreedySearchDecode(contextVector, *maxSeqLength, semanticOutputVocabulary.GetTokenID("<s>"), semanticOutputVocabulary.GetTokenID("</s>"))
	if err != nil {
		log.Fatalf("Greedy search decode failed: %v", err)
	}

	// Decode the predicted IDs to a sentence
	predictedSentence, err := semanticOutputTokenizer.Decode(predictedIDs)
	if err != nil {
		log.Fatalf("Failed to decode predicted IDs: %v", err)
	}

	fmt.Println("--- Predicted Semantic Output ---")
	fmt.Println(predictedSentence)
	fmt.Println("---------------------------------")
}