package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"strings"

	moemodel "nlptagger/neural/moe/model"
	. "nlptagger/neural/nn"
	. "nlptagger/neural/tensor"
	"nlptagger/neural/tokenizer"
	mainvocab "nlptagger/neural/nnu/vocab"
)

// IntentTrainingExample represents a single training example with a query and its intents.
type IntentTrainingExample struct {
	Query        string `json:"query"`
	ParentIntent string `json:"parent_intent"`
	ChildIntent  string `json:"child_intent"`
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

func main() {
	// Define paths
	const trainingDataPath = "trainingdata/intent_data.json"
	const queryVocabPath = "gob_models/query_vocabulary.gob"
	const parentIntentVocabPath = "gob_models/parent_intent_vocabulary.gob"
	const childIntentVocabPath = "gob_models/child_intent_vocabulary.gob"
	const modelSavePath = "gob_models/moe_classification_model.gob"

	// Load training data
	trainingData, err := LoadIntentTrainingData(trainingDataPath)
	if err != nil {
		log.Fatalf("Failed to load intent training data: %v", err)
	}

	// Create vocabularies
	queryVocab := mainvocab.NewVocabulary()
	parentIntentVocab := mainvocab.NewVocabulary()
	childIntentVocab := mainvocab.NewVocabulary()

	

	 	for _, example := range *trainingData {
		words := strings.Fields(strings.ToLower(example.Query))
		for _, word := range words {
			queryVocab.AddToken(word)
		}
		parentIntentVocab.AddToken(example.ParentIntent)
		childIntentVocab.AddToken(example.ChildIntent)
	}

	// Save vocabularies
	queryVocab.Save(queryVocabPath)
	parentIntentVocab.Save(parentIntentVocabPath)
	childIntentVocab.Save(childIntentVocabPath)

	log.Printf("Query vocabulary size: %d", queryVocab.Size())
	log.Printf("Parent intent vocabulary size: %d", parentIntentVocab.Size())
	log.Printf("Child intent vocabulary size: %d", childIntentVocab.Size())

	// Create model
	model, err := moemodel.NewMoEClassificationModel(
		queryVocab.Size(),
		64, // embeddingDim
		parentIntentVocab.Size(),
		childIntentVocab.Size(),
		2, // numExperts
		1, // k
		32, // maxSeqLength
	)
	if err != nil {
		log.Fatalf("Failed to create new MoE model: %v", err)
	}

	// Train the model
	TrainIntentModel(model, trainingData, queryVocab, parentIntentVocab, childIntentVocab, 10, 0.001, 32, 32)

	// Save the trained model
	log.Printf("Saving MoE Classification model to %s", modelSavePath)
	err = moemodel.SaveMoEClassificationModelToGOB(model, modelSavePath)
	if err != nil {
		log.Fatalf("Failed to save MoE model: %v", err)
	}

	log.Println("Training complete.")
}

// TrainIntentModel trains the MoEClassificationModel for intent classification.
func TrainIntentModel(model *moemodel.MoEClassificationModel, data *IntentTrainingData, queryVocab, parentIntentVocab, childIntentVocab *mainvocab.Vocabulary, epochs int, learningRate float64, batchSize int, maxSeqLength int) {
	optimizer := NewOptimizer(model.Parameters(), learningRate, 5.0)

	for epoch := 0; epoch < epochs; epoch++ {
		log.Printf("Epoch %d/%d", epoch+1, epochs)
		totalLoss := 0.0
		numBatches := 0

		for i := 0; i < len(*data); i += batchSize {
			end := i + batchSize
			if end > len(*data) {
				end = len(*data)
			}
			batch := (*data)[i:end]

			loss, err := trainIntentModelBatch(model, optimizer, batch, queryVocab, parentIntentVocab, childIntentVocab, maxSeqLength)
			if err != nil {
				log.Printf("Error training batch: %v", err)
				continue
			}
			totalLoss += loss
			numBatches++
		}
		if numBatches > 0 {
			log.Printf("Epoch %d, Average Loss: %f", epoch+1, totalLoss/float64(numBatches))
		}
	}
}

// trainIntentModelBatch performs a single training step on a batch of intent data.
func trainIntentModelBatch(model *moemodel.MoEClassificationModel, optimizer Optimizer, batch IntentTrainingData, queryVocab, parentIntentVocab, childIntentVocab *mainvocab.Vocabulary, maxSeqLength int) (float64, error) {
	optimizer.ZeroGrad()

	batchSize := len(batch)

	inputIDsBatch := make([]int, batchSize*maxSeqLength)
	parentIntentIDs := make([]int, batchSize)
	childIntentIDs := make([]int, batchSize)

	tokenizer, err := tokenizer.NewTokenizer(queryVocab)
	if err != nil {
		return 0, fmt.Errorf("failed to create tokenizer: %w", err)
	}

	for i, example := range batch {
		tokenIDs, err := tokenizer.Encode(example.Query)
		if err != nil {
			return 0, fmt.Errorf("query tokenization failed for item %d: %w", i, err)
		}

		if len(tokenIDs) > maxSeqLength {
			tokenIDs = tokenIDs[:maxSeqLength]
		} else {
			padding := make([]int, maxSeqLength-len(tokenIDs))
			for j := range padding {
				padding[j] = queryVocab.PaddingTokenID
			}
			tokenIDs = append(tokenIDs, padding...)
		}
		copy(inputIDsBatch[i*maxSeqLength:(i+1)*maxSeqLength], tokenIDs)

		parentIntentIDs[i] = parentIntentVocab.GetTokenID(example.ParentIntent)
		childIntentIDs[i] = childIntentVocab.GetTokenID(example.ChildIntent)
	}

	inputTensor := NewTensor([]int{batchSize, maxSeqLength}, convertIntsToFloat64s(inputIDsBatch), false)

	parentLogits, childLogits, err := model.Forward(inputTensor, nil, nil, nil)
	if err != nil {
		return 0, fmt.Errorf("model forward pass failed: %w", err)
	}

	parentLoss, parentGrad := CrossEntropyLoss(parentLogits, parentIntentIDs, -1)
	childLoss, childGrad := CrossEntropyLoss(childLogits, childIntentIDs, -1)

	totalLoss := parentLoss + childLoss

	// Backward pass
	if parentGrad == nil || childGrad == nil {
		log.Printf("Skipping backward pass due to nil gradient")
		return totalLoss, nil
	}
	err = model.Backward(parentGrad, childGrad)
	if err != nil {
		return 0, fmt.Errorf("model backward pass failed: %w", err)
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
