package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"os/signal"
	"runtime/pprof" // Added for profiling
	"strings"
	"syscall"

	"nlptagger/neural/moe"
	. "nlptagger/neural/nn"
	mainvocab "nlptagger/neural/nnu/vocab"
	. "nlptagger/neural/tensor"
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

// TokenizeAndConvertToIDs tokenizes a text and converts tokens to their corresponding IDs, handling padding/truncation.
func TokenizeAndConvertToIDs(text string, tokenizer *tokenizer.Tokenizer, vocabulary *mainvocab.Vocabulary, maxLen int) ([]int, error) {

	tokenIDs, err := tokenizer.Encode(text)
	if err != nil {
		return nil, fmt.Errorf("failed to encode text: %w", err)
	}

	// Handle empty input strings by returning a slice with only the PaddingTokenID
	if len(tokenIDs) == 0 && maxLen > 0 {
		tokenIDs = make([]int, maxLen)
		for i := range tokenIDs {
			tokenIDs[i] = vocabulary.PaddingTokenID
		}
		return tokenIDs, nil
	} else if len(tokenIDs) == 0 {
		return []int{vocabulary.PaddingTokenID}, nil
	}

	if maxLen > 0 {
		if len(tokenIDs) > maxLen {
			tokenIDs = tokenIDs[:maxLen]
		} else if len(tokenIDs) < maxLen {
			paddingSize := maxLen - len(tokenIDs)
			padding := make([]int, paddingSize)
			for i := range padding {
				padding[i] = vocabulary.PaddingTokenID
			}
			tokenIDs = append(tokenIDs, padding...)
		}
	}
	return tokenIDs, nil
}

// TrainIntentMoEModel trains the MoEClassificationModel.
func TrainIntentMoEModel(model *moe.IntentMoE, data *IntentTrainingData, epochs int, learningRate float64, batchSize int, queryVocab, parentIntentVocab, childIntentVocab *mainvocab.Vocabulary, queryTokenizer *tokenizer.Tokenizer, maxSequenceLength int, profileFile *os.File) error {

	if model == nil {
		return errors.New("cannot train a nil model")
	}
	if data == nil || len(*data) == 0 {
		return errors.New("no training data provided")
	}

	optimizer := NewOptimizer(model.Parameters(), learningRate, 1.0) // Using a clip value of 1.0

	for epoch := 0; epoch < epochs; epoch++ {
		log.Printf("Epoch %d/%d", epoch+1, epochs)
		totalLoss := 0.0
		numBatches := 0
		// Create batches for training
		for i := 0; i < len(*data); i += batchSize {
			end := i + batchSize
			if end > len(*data) {
				end = len(*data)
			}
			batch := (*data)[i:end]

			loss, err := trainIntentMoEBatch(model, optimizer, batch, queryVocab, parentIntentVocab, childIntentVocab, queryTokenizer, maxSequenceLength)
			if err != nil {
				log.Printf("Error training batch: %v", err)
				continue // Or handle error more strictly
			}
			totalLoss += loss
			numBatches++
		}
		if numBatches > 0 {
			log.Printf("Epoch %d, Average Loss: %f", epoch+1, totalLoss/float64(numBatches))
		}
	}

	return nil
}

// trainIntentMoEBatch performs a single training step on a batch of data.
func trainIntentMoEBatch(intentMoEModel *moe.IntentMoE, optimizer Optimizer, batch IntentTrainingData, queryVocab, parentIntentVocab, childIntentVocab *mainvocab.Vocabulary, queryTokenizer *tokenizer.Tokenizer, maxSequenceLength int) (float64, error) {
	optimizer.ZeroGrad()

	batchSize := len(batch)

	inputIDsBatch := make([]int, batchSize*maxSequenceLength)
	parentIntentIDsBatch := make([]int, batchSize)
	childIntentIDsBatch := make([]int, batchSize)

	for i, example := range batch {
		// Tokenize and pad query
		queryTokens, err := TokenizeAndConvertToIDs(example.Query, queryTokenizer, queryVocab, maxSequenceLength)
		if err != nil {
			return 0, fmt.Errorf("query tokenization failed for item %d: %w", i, err)
		}
		copy(inputIDsBatch[i*maxSequenceLength:(i+1)*maxSequenceLength], queryTokens)

		// Get parent and child intent IDs
		parentIntentIDsBatch[i] = parentIntentVocab.GetTokenID(example.ParentIntent)
		childIntentIDsBatch[i] = childIntentVocab.GetTokenID(example.ChildIntent)
	}

	// Convert input IDs to a Tensor (embeddings will be handled by the model)
	inputTensor := NewTensor([]int{batchSize, maxSequenceLength}, convertIntsToFloat64s(inputIDsBatch), false)

	// Forward pass through the IntentMoE model
	parentLogits, childLogits, err := intentMoEModel.Forward(inputTensor)
	if err != nil {
		return 0, fmt.Errorf("IntentMoE model forward pass failed: %w", err)
	}

	// Calculate loss for the parent and child intents
	parentLoss, parentGrad := CrossEntropyLoss(parentLogits, parentIntentIDsBatch, parentIntentVocab.PaddingTokenID)
	childLoss, childGrad := CrossEntropyLoss(childLogits, childIntentIDsBatch, childIntentVocab.PaddingTokenID)

	totalLoss := parentLoss + childLoss
	allGrads := []*Tensor{parentGrad, childGrad}

	// Backward pass
	err = intentMoEModel.Backward(allGrads...)
	if err != nil {
		return 0, fmt.Errorf("IntentMoE model backward pass failed: %w", err)
	}

	optimizer.Step()

	return totalLoss, nil
}

func convertIntsToFloat64s(input []int) []float64 {
	output := make([]float64, len(input))
	for i, v := range input {
		output[i] = float64(v)
	}
	return output
}

func BuildVocabularies(dataPath string) (*mainvocab.Vocabulary, *mainvocab.Vocabulary, *mainvocab.Vocabulary, error) {
	queryVocabulary := mainvocab.NewVocabulary()
	parentIntentVocabulary := mainvocab.NewVocabulary()
	childIntentVocabulary := mainvocab.NewVocabulary()

	intentTrainingData, err := LoadIntentTrainingData(dataPath)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to load intent training data from %s: %w", dataPath, err)
	}

	for _, pair := range *intentTrainingData {
		// Use the same tokenizer logic as during inference to build the vocabulary
		tokenizedQuery := tokenizer.Tokenize(strings.ToLower(pair.Query))
		for _, word := range tokenizedQuery {
			queryVocabulary.AddToken(word)
		}
		parentIntentVocabulary.AddToken(pair.ParentIntent)
		childIntentVocabulary.AddToken(pair.ChildIntent)
	}

	return queryVocabulary, parentIntentVocabulary, childIntentVocabulary, nil
}

func main() {
	const intentTrainingDataPath = "./trainingdata/intent_data.json"
	// Start CPU profiling
	f, err := os.Create("cpu.prof")
	if err != nil {
		log.Fatal("could not create CPU profile: ", err)
	}
	if err := pprof.StartCPUProfile(f); err != nil {
		log.Fatal("could not start CPU profile: ", err)
	}

	// Set up a channel to listen for interrupt signals
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Goroutine to handle graceful shutdown on signal
	go func() {
		<-sigChan // Block until a signal is received
		log.Println("Received interrupt signal. Stopping CPU profile and closing file.")
		pprof.StopCPUProfile()
		f.Close()
		os.Exit(0) // Exit gracefully
	}()

	// Define training parameters
	epochs := 500
	learningRate := 0.001
	batchSize := 64
	queryVocabularySavePath := "gob_models/query_vocabulary.gob"
	parentIntentVocabularySavePath := "gob_models/parent_intent_vocabulary.gob"
	childIntentVocabularySavePath := "gob_models/child_intent_vocabulary.gob"

	// Try to load vocabularies first
	queryVocabulary, err := mainvocab.LoadVocabulary(queryVocabularySavePath)
	if err != nil {
		log.Println("Failed to load query vocabulary, creating a new one.")
	}
	parentIntentVocabulary, err := mainvocab.LoadVocabulary(parentIntentVocabularySavePath)
	if err != nil {
		log.Println("Failed to load parent intent vocabulary, creating a new one.")
	}
	childIntentVocabulary, err := mainvocab.LoadVocabulary(childIntentVocabularySavePath)
	if err != nil {
		log.Println("Failed to load child intent vocabulary, creating a new one.")
	}

	if queryVocabulary == nil || parentIntentVocabulary == nil || childIntentVocabulary == nil {
		log.Println("Building vocabularies from scratch...")
		queryVocabulary, parentIntentVocabulary, childIntentVocabulary, err = BuildVocabularies(intentTrainingDataPath)
		if err != nil {
			log.Fatalf("Failed to build vocabularies: %v", err)
		}
	}

	log.Printf("Query Vocabulary (after load/create): Size=%d", len(queryVocabulary.TokenToWord))
	log.Printf("Parent Intent Vocabulary (after load/create): Size=%d", len(parentIntentVocabulary.TokenToWord))
	log.Printf("Child Intent Vocabulary (after load/create): Size=%d", len(childIntentVocabulary.TokenToWord))

	// Load Intent training data
	intentTrainingData, err := LoadIntentTrainingData(intentTrainingDataPath)
	if err != nil {
		log.Fatalf("Failed to load intent training data from %s: %v", intentTrainingDataPath, err)
	}
	log.Printf("Loaded %d training examples.", len(*intentTrainingData))

	// After vocabularies are fully populated, determine vocab sizes and create/load model
	inputVocabSize := len(queryVocabulary.TokenToWord)
	parentVocabSize := len(parentIntentVocabulary.TokenToWord)
	childVocabSize := len(childIntentVocabulary.TokenToWord)
	embeddingDim := 128
	numExperts := 2
	maxSequenceLength := 32 // Max length for input query and output description

	log.Printf("Query Vocabulary Size: %d", inputVocabSize)
	log.Printf("Parent Intent Vocabulary Size: %d", parentVocabSize)
	log.Printf("Child Intent Vocabulary Size: %d", childVocabSize)

	var intentMoEModel *moe.IntentMoE // Declare intentMoEModel here

	modelSavePath := "gob_models/moe_classification_model.gob"

	// Try to load IntentMoE model
	intentMoEModel, err = moe.LoadIntentMoEModelFromGOB(modelSavePath)
	if err != nil {
		log.Printf("Failed to load IntentMoE model, creating a new one: %v", err)
		intentMoEModel, err = moe.NewIntentMoE(inputVocabSize, embeddingDim, numExperts, parentVocabSize, childVocabSize)
		if err != nil {
			log.Fatalf("Failed to create new IntentMoE model: %v", err)
		}
	} else {
		log.Printf("Loaded IntentMoE model from %s", modelSavePath)
	}

	// Create tokenizers once after vocabularies are loaded/created
	queryTokenizer, err := tokenizer.NewTokenizer(queryVocabulary)
	if err != nil {
		log.Fatalf("Failed to create query tokenizer: %v", err)
	}

	// Train the model
	err = TrainIntentMoEModel(intentMoEModel, intentTrainingData, epochs, learningRate, batchSize, queryVocabulary, parentIntentVocabulary, childIntentVocabulary, queryTokenizer, maxSequenceLength, f)
	if err != nil {
		log.Fatalf("Failed to train IntentMoE model: %v", err)
	}

	// Save the trained model
	fmt.Printf("Saving IntentMoE model to %s", modelSavePath)
	err = moe.SaveIntentMoEModelToGOB(intentMoEModel, modelSavePath)
	if err != nil {
		log.Fatalf("Failed to save IntentMoE model: %v", err)
	}

	// Save the vocabularies
	err = queryVocabulary.Save(queryVocabularySavePath)
	if err != nil {
		log.Fatalf("Failed to save query vocabulary: %v", err)
	}
	err = parentIntentVocabulary.Save(parentIntentVocabularySavePath)
	if err != nil {
		log.Fatalf("Failed to save parent intent vocabulary: %v", err)
	}
	err = childIntentVocabulary.Save(childIntentVocabularySavePath)
	if err != nil {
		log.Fatalf("Failed to save child intent vocabulary: %v", err)
	}
}
