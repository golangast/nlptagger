package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"os/exec"
	"strings"

	"github.com/zendrulat/nlptagger/neural/moe"
	mainvocab "github.com/zendrulat/nlptagger/neural/nnu/vocab"
	"github.com/zendrulat/nlptagger/neural/tensor"
	"github.com/zendrulat/nlptagger/neural/tokenizer"
	"github.com/zendrulat/nlptagger/tagger/nertagger"
	"github.com/zendrulat/nlptagger/tagger/postagger"
	"github.com/zendrulat/nlptagger/tagger/tag"
)

func main() {
	trainWord2Vec := flag.Bool("train-word2vec", false, "Train the Word2Vec model")
	trainMoE := flag.Bool("train-moe", false, "Train the MoE model")
	trainIntentClassifier := flag.Bool("train-intent-classifier", false, "Train the intent classification model")
	moeInferenceQuery := flag.String("moe_inference", "", "Run MoE inference with the given query")
	runLLMFlag := flag.Bool("llm", false, "Run in interactive LLM mode")

	flag.Parse()

	if *runLLMFlag {
		runLLM()
	} else if *trainWord2Vec {
		runModule("cmd/train_word2vec")
	} else if *trainMoE {
		runModule("cmd/train_moe")
	} else if *trainIntentClassifier {
		runModule("cmd/train_intent_classifier")
	} else if *moeInferenceQuery != "" {
		runMoeInference(*moeInferenceQuery)
	} else {
		log.Println("No action specified. Use -train-word2vec, -train-moe, -train-intent-classifier, -moe_inference <query>, or -llm.")
	}
}

func runMoeInference(query string) {
	cmd := exec.Command("go", "run", "./cmd/moe_inference", "-query", query)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err := cmd.Run()
	if err != nil {
		log.Fatalf("Failed to run MoE inference: %v", err)
	}
}

func runModule(path string) {
	cmd := exec.Command("go", "run", "./"+path)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err := cmd.Run()
	if err != nil {
		log.Fatalf("Failed to run module %s: %v", path, err)
	}
}

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