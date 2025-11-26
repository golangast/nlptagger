package main

import (
	"bufio"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strings"

	"nlptagger/neural/moe"
	mainvocab "nlptagger/neural/nnu/vocab"
	"nlptagger/neural/tensor"
	"nlptagger/neural/tokenizer"
	"nlptagger/tagger/nertagger"
	"nlptagger/tagger/postagger"
	"nlptagger/tagger/tag"
)

func runLLM() {
	rand.Seed(1) // Seed the random number generator for deterministic behavior

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

	log.Println("NER Tagger and POS Tagger initialized.")

	reader := bufio.NewReader(os.Stdin)
	fmt.Println("LLM Interaction Mode. Type 'exit' to quit.")
	fmt.Println("-----------------------------------------")

	for {
		fmt.Print("> ")
		query, _ := reader.ReadString('\n')
		query = strings.TrimSpace(query)

		if query == "exit" {
			break
		}

		// --- Tagging ---
		words := strings.Fields(query)
		posTags := postagger.TagTokens(words)
		taggedData := nertagger.Nertagger(tag.Tag{Tokens: words, PosTag: posTags})

		fmt.Println("\n--- Part-of-Speech & Named Entity Recognition ---")
		for i := range taggedData.Tokens {
			nerTag := "O" // Default to Outside of any entity
			if i < len(taggedData.NerTag) && taggedData.NerTag[i] != "" {
				nerTag = taggedData.NerTag[i]
			}
			fmt.Printf("Word: %-15s POS: %-10s NER: %s\n", taggedData.Tokens[i], taggedData.PosTag[i], nerTag)
		}
		fmt.Println("-----------------------------------------------------")

		// Encode the query
		tokenIDs, err := tok.Encode(query)
		if err != nil {
			log.Printf("Failed to encode query: %v", err)
			continue
		}

		// Pad or truncate the sequence to a fixed length
		maxSeqLength := 32 // Assuming a max length
		if len(tokenIDs) > maxSeqLength {
			tokenIDs = tokenIDs[:maxSeqLength] // Truncate from the end
		} else {
			for len(tokenIDs) < maxSeqLength {
				tokenIDs = append(tokenIDs, vocabulary.PaddingTokenID) // Appends padding
			}
		}
		inputData := make([]float64, len(tokenIDs))
		for i, id := range tokenIDs {
			inputData[i] = float64(id)
		}
		inputTensor := tensor.NewTensor([]int{1, len(inputData)}, inputData, false)

		// Create a dummy target tensor for inference
		dummyTargetTokenIDs := make([]float64, maxSeqLength)
		for i := 0; i < maxSeqLength; i++ {
			dummyTargetTokenIDs[i] = float64(vocabulary.PaddingTokenID)
		}
		dummyTargetTensor := tensor.NewTensor([]int{1, maxSeqLength}, dummyTargetTokenIDs, false)

		// Forward pass to get the context vector
		_, contextVector, err := model.Forward(0.0, inputTensor, dummyTargetTensor)
		if err != nil {
			log.Printf("MoE model forward pass failed: %v", err)
			continue
		}

		// Greedy search decode to get the predicted token IDs
		predictedIDs, err := model.GreedySearchDecode(contextVector, maxSeqLength, semanticOutputVocabulary.GetTokenID("<s>"), semanticOutputVocabulary.GetTokenID("</s>"), 1.0)
		if err != nil {
			log.Printf("Greedy search decode failed: %v", err)
			continue
		}

		// Decode the predicted IDs to a sentence
		predictedSentence, err := semanticOutputTokenizer.Decode(predictedIDs)
		if err != nil {
			log.Printf("Failed to decode predicted IDs: %v", err)
			continue
		}

		fmt.Println("---")
		fmt.Println(predictedSentence)
		fmt.Println("---------------------------------")
	}
}
