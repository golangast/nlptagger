package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math" // Keep math for softmax
	"math/rand"
	"os"
	"sort"

	"nlptagger/neural/moe"
	mainvocab "nlptagger/neural/nnu/vocab"
	"nlptagger/neural/tensor"
	"nlptagger/neural/tokenizer"
)

// IntentTrainingExample represents a single training example for intent classification.
type IntentTrainingExample struct {
	Query        string `json:"query"`
	ParentIntent string `json:"parent_intent"`
	ChildIntent  string `json:"child_intent"`
	Description  string `json:"description"`
	Sentence     string `json:"sentence"`
}

// IntentTrainingData represents the structure of the intent training data JSON.
type IntentTrainingData []IntentTrainingExample

// LoadIntentTrainingData loads the intent training data from a JSON file.
func LoadIntentTrainingData(filePath string) (*IntentTrainingData, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open training data file %s: %w", filePath, err)
	}
	defer file.Close()

	bytes, err := io.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("failed to read training data file %s: %w", filePath, err)
	}

	var data IntentTrainingData
	err = json.Unmarshal(bytes, &data)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal training data JSON from %s: %w", filePath, err)
	}

	return &data, nil
}

type Prediction struct {
	TokenID    int
	Word       string
	Confidence float64
}

func getTopNPredictions(probabilities []float64, vocab []string, n int) []Prediction {
	predictions := make([]Prediction, 0, len(probabilities))
	for i, p := range probabilities {
		if i < 2 { // Skip <pad> and UNK
			continue
		}
		if i < len(vocab) {
			word := vocab[i]
			predictions = append(predictions, Prediction{
				TokenID:    i,
				Word:       word,
				Confidence: p * 100.0,
			})
		}
	}

	// Sort predictions by confidence
	sort.Slice(predictions, func(i, j int) bool {
		return predictions[i].Confidence > predictions[j].Confidence
	})

	if len(predictions) < n {
		return predictions
	}
	return predictions[:n]
}

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
	const parentIntentVocabPath = "gob_models/parent_intent_vocabulary.gob"
	const childIntentVocabPath = "gob_models/child_intent_vocabulary.gob"
	const intentTrainingDataPath = "trainingdata/intent_data.json"

	// Load vocabularies
	vocabulary, err := mainvocab.LoadVocabulary(vocabPath)
	if err != nil {
		log.Fatalf("Failed to set up input vocabulary: %v", err)
	}

	// Setup parent intent vocabulary
	parentIntentVocabulary, err := mainvocab.LoadVocabulary(parentIntentVocabPath)
	if err != nil {
		log.Fatalf("Failed to set up parent intent vocabulary: %v", err)
	}

	// Setup child intent vocabulary
	childIntentVocabulary, err := mainvocab.LoadVocabulary(childIntentVocabPath)
	if err != nil {
		log.Fatalf("Failed to set up child intent vocabulary: %v", err)
	}

	// Create tokenizer
	tokenizer, err := tokenizer.NewTokenizer(vocabulary)
	if err != nil {
		log.Fatalf("Failed to create tokenizer: %w", err)
	}

	// Load the trained IntentMoE model
	model, err := moe.LoadIntentMoEModelFromGOB(moeModelPath)
	if err != nil {
		log.Fatalf("Failed to load MoE model: %v", err)
	}

	// Load intent training data
	intentTrainingData, err := LoadIntentTrainingData(intentTrainingDataPath)
	if err != nil {
		log.Fatalf("Failed to load intent training data: %v", err)
	}

	log.Printf("---\n--- DEBUG: Parent Intent Vocabulary (TokenToWord): %v ---", parentIntentVocabulary.TokenToWord)
	log.Printf("---\n--- DEBUG: Child Intent Vocabulary (TokenToWord): %v ---", childIntentVocabulary.TokenToWord)

	log.Printf("Running MoE inference for query: \"%s\"", *query)

	// Encode the query

	tokenIDs, err := tokenizer.Encode(*query)
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

	// Forward pass
	parentLogits, childLogits, err := model.Forward(inputTensor)
	if err != nil {
		log.Fatalf("MoE model forward pass failed: %v", err)
	}

	// Interpret parent intent output
	parentProbabilities := softmax(parentLogits.Data)
	topParentPredictions := getTopNPredictions(parentProbabilities, parentIntentVocabulary.TokenToWord, 3)

	fmt.Println("--- Top 3 Parent Intent Predictions ---")
	for _, p := range topParentPredictions {
		importance := ""
		if p.Confidence > 50.0 {
			importance = " (Important)"
		}
		fmt.Printf("  - Word: %-20s (Confidence: %.2f%%)%s\n", p.Word, p.Confidence, importance)
	}
	fmt.Println("------------------------------------")

	// Interpret child intent output
	childProbabilities := softmax(childLogits.Data)
	topChildPredictions := getTopNPredictions(childProbabilities, childIntentVocabulary.TokenToWord, 3)

	fmt.Println("--- Top 3 Child Intent Predictions ---")
	for _, p := range topChildPredictions {
		importance := ""
		if p.Confidence > 50.0 {
			importance = " (Important)"
		}
		fmt.Printf("  - Word: %-20s (Confidence: %.2f%%)%s\n", p.Word, p.Confidence, importance)
	}
	fmt.Println("-----------------------------------")

	if len(topParentPredictions) > 0 && len(topChildPredictions) > 0 {
		predictedParentWord := topParentPredictions[0].Word
		predictedChildWord := topChildPredictions[0].Word
		fmt.Printf("\nDescription: The model's top prediction is an action related to %s, specifically to %s.\n", predictedParentWord, predictedChildWord)

		// Find and print the intent sentence
		foundSentence := ""
		for _, example := range *intentTrainingData {
			if example.ParentIntent == predictedParentWord && example.ChildIntent == predictedChildWord {
				foundSentence = example.Sentence
				break
			}
		}

		if foundSentence != "" {
			fmt.Printf("Intent Sentence: %s\n", foundSentence)
		} else {
			fmt.Println("Intent Sentence: Not found in training data.")
		}
	}
}

// softmax applies the softmax function to a slice of float64.
func softmax(logits []float64) []float64 {
	if len(logits) == 0 {
		return []float64{}
	}
	maxLogit := logits[0]
	for _, logit := range logits {
		if logit > maxLogit {
			maxLogit = logit
		}
	}
	expSum := 0.0
	for _, logit := range logits {
		expSum += math.Exp(logit - maxLogit)
	}

	probabilities := make([]float64, len(logits))
	for i, logit := range logits {
		probabilities[i] = math.Exp(logit-maxLogit) / expSum
	}
	return probabilities
}
