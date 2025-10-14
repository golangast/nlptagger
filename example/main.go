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
	"os/exec"
	"sort"

	"nlptagger/neural/moe"
	mainvocab "nlptagger/neural/nnu/vocab"
	"nlptagger/neural/tensor"
	"nlptagger/neural/tokenizer"
	"nlptagger/tagger/nertagger"
	"nlptagger/tagger/postagger"
	"nlptagger/tagger/tag"
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
	tok, err := tokenizer.NewTokenizer(vocabulary)
	if err != nil {
		log.Fatalf("Failed to create tokenizer: %w", err)
	}

	// Load the trained MoEClassificationModel model
	model, err := moe.LoadIntentMoEModelFromGOB(moeModelPath)
	if err != nil {
		log.Fatalf("Failed to load MoE model: %v", err)
	}

	// Load intent training data
	intentTrainingData, err := LoadIntentTrainingData(intentTrainingDataPath)
	if err != nil {
		log.Fatalf("Failed to load intent training data: %v", err)
	}

	log.Printf("--- DEBUG: Parent Intent Vocabulary (TokenToWord): %v ---", parentIntentVocabulary.TokenToWord)
	log.Printf("--- DEBUG: Child Intent Vocabulary (TokenToWord): %v ---", childIntentVocabulary.TokenToWord)

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
	// The actual content of this tensor won't be used for parent/child intent classification.
	dummyTargetTokenIDs := make([]float64, *maxSeqLength)
	for i := 0; i < *maxSeqLength; i++ {
		dummyTargetTokenIDs[i] = float64(vocabulary.PaddingTokenID)
	}
	dummyTargetTensor := tensor.NewTensor([]int{1, *maxSeqLength}, dummyTargetTokenIDs, false)

	// Forward pass
	parentLogits, childLogits, _, _, err := model.Forward(inputTensor, dummyTargetTensor)
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

	// Perform POS tagging
	posResult := postagger.Postagger(*query)
	fmt.Println("\n--- POS Tagging Results ---")
	fmt.Printf("Tokens: %v\n", posResult.Tokens)
	fmt.Printf("POS Tags: %v\n", posResult.PosTag)

	// Perform NER tagging
	nerResult := nertagger.Nertagger(posResult)
	fmt.Println("\n--- NER Tagging Results ---")
	fmt.Printf("Tokens: %v\n", nerResult.Tokens)
	fmt.Printf("NER Tags: %v\n", nerResult.NerTag)

	// Generate and execute command based on NER/POS tags and intent predictions
	fmt.Println("\n--- Generating Command ---")
	command := generateCommand("file_system", topChildPredictions[0].Word, nerResult)
	if command != "" {
		fmt.Printf("Generated Command: %s\n", command)
		// Execute the command
		cmd := exec.Command("bash", "-c", command)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		err := cmd.Run()
		if err != nil {
			log.Printf("Error executing command: %v", err)
		}
	} else {
		fmt.Println("Could not generate a command.")
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

func generateCommand(parentIntent, childIntent string, nerResult tag.Tag) string {
	switch parentIntent {
	case "file_system":
		switch childIntent {
		case "create":
			var fileName, folderName string
			for i, tag := range nerResult.NerTag {
				if tag == "OBJECT_TYPE" && nerResult.Tokens[i] == "file" {
					if i+2 < len(nerResult.Tokens) && nerResult.NerTag[i+1] == "NAME_PREFIX" && nerResult.NerTag[i+2] == "NAME" {
						fileName = nerResult.Tokens[i+2]
					}
				} else if tag == "OBJECT_TYPE" && nerResult.Tokens[i] == "folder" {
					if i+2 < len(nerResult.Tokens) && nerResult.NerTag[i+1] == "NAME_PREFIX" && nerResult.NerTag[i+2] == "NAME" {
						folderName = nerResult.Tokens[i+2]
					}
				}
			}
			if fileName != "" && folderName != "" {
				return fmt.Sprintf("mkdir -p %s && touch %s/%s", folderName, folderName, fileName)
			} else if fileName != "" {
				return fmt.Sprintf("touch %s", fileName)
			}
		}
		// Add other file_system child intents here (e.g., "delete", "read")
	}
	// Add other parent intents here

	return ""
}