package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"strings"

	moemodel "nlptagger/neural/moe/model"
	"nlptagger/neural/nn"
	mainvocab "nlptagger/neural/nnu/vocab"
	"nlptagger/neural/tensor"
	"nlptagger/neural/tokenizer"
)

// TrainingExample represents a single training example with input text and parent/child intents.
type TrainingExample struct {
	InputText    string `json:"query"`
	ParentIntent string `json:"parent_intent"`
	ChildIntent  string `json:"child_intent"`
}

// TrainingData represents the structure of the training data JSON.
type TrainingData []TrainingExample

// LoadTrainingData loads the training data from a JSON file.
func LoadTrainingData(filePath string) (*TrainingData, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open training data file %s: %w", filePath, err)
	}
	defer file.Close()

	bytes, err := io.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("failed to read training data file %s: %w", filePath, err)
	}

	var data TrainingData
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
	trainingData, err := LoadTrainingData(trainingDataPath)
	if err != nil {
		log.Fatalf("Failed to load training data: %v", err)
	}

	// Create vocabularies
	queryVocabulary := mainvocab.NewVocabulary()
	queryVocabulary.AddToken("<s>")
	queryVocabulary.AddToken("</s>")
	parentIntentVocabulary := mainvocab.NewVocabulary()
	childIntentVocabulary := mainvocab.NewVocabulary()

	for _, example := range *trainingData {
		words := strings.Fields(example.InputText)
		for _, word := range words {
			queryVocabulary.AddToken(word)
		}
		parentIntentVocabulary.AddToken(example.ParentIntent)
		childIntentVocabulary.AddToken(example.ChildIntent)
	}

	// Save vocabularies
	if err := queryVocabulary.Save(queryVocabPath); err != nil {
		log.Fatalf("Failed to save query vocabulary: %v", err)
	}
	if err := parentIntentVocabulary.Save(parentIntentVocabPath); err != nil {
		log.Fatalf("Failed to save parent intent vocabulary: %v", err)
	}
	if err := childIntentVocabulary.Save(childIntentVocabPath); err != nil {
		log.Fatalf("Failed to save child intent vocabulary: %v", err)
	}

	// Create the model
	model, err := moemodel.NewMoEClassificationModel(
		queryVocabulary.Size(),
		128, // embeddingDim (reduced from 256)
		parentIntentVocabulary.Size(),
		childIntentVocabulary.Size(),
		4,  // numExperts (reduced from 8)
		1,  // k (reduced from 2)
		32, // maxSeqLength (reduced from 64)
	)
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	// Train the model
	optimizer := nn.NewOptimizer(model.Parameters(), 0.001, 5.0)
	tokenizer, _ := tokenizer.NewTokenizer(queryVocabulary) // Initialize tokenizer once

	const batchSize = 32 // Define batch size

	for epoch := 0; epoch < 3; epoch++ { // Reduced epochs from 10 to 3
		// Shuffle training data for each epoch (optional but good practice)
		// rand.Shuffle(len(*trainingData), func(i, j int) {
		// 	(*trainingData)[i], (*trainingData)[j] = (*trainingData)[j], (*trainingData)[i]
		// })

		for i := 0; i < len(*trainingData); i += batchSize {
			batchEnd := i + batchSize
			if batchEnd > len(*trainingData) {
				batchEnd = len(*trainingData)
			}
			batch := (*trainingData)[i:batchEnd]

			// Prepare batch inputs and targets
			var batchInputIDs [][]float64
			var batchParentTargetIDs []int
			var batchChildTargetIDs []int
			currentBatchSize := len(batch)

			for _, example := range batch {
				tokenIDs, _ := tokenizer.Encode(example.InputText)
				inputData := make([]float64, len(tokenIDs))
				for i, id := range tokenIDs {
					inputData[i] = float64(id)
				}
				batchInputIDs = append(batchInputIDs, inputData)
				batchParentTargetIDs = append(batchParentTargetIDs, parentIntentVocabulary.WordToToken[example.ParentIntent])
				batchChildTargetIDs = append(batchChildTargetIDs, childIntentVocabulary.WordToToken[example.ChildIntent])
			}

			// Pad sequences to maxSeqLength (64)
			paddedInputData := make([]float64, currentBatchSize*model.BertConfig.MaxPositionEmbeddings)
			for rowIdx, seq := range batchInputIDs {
				copy(paddedInputData[rowIdx*model.BertConfig.MaxPositionEmbeddings:], seq)
				// Padding with 0s (assuming 0 is a valid padding token ID)
				for k := len(seq); k < model.BertConfig.MaxPositionEmbeddings; k++ {
					paddedInputData[rowIdx*model.BertConfig.MaxPositionEmbeddings+k] = float64(queryVocabulary.PaddingTokenID) // Use actual padding token ID
				}
			}

			inputTensor := tensor.NewTensor([]int{currentBatchSize, model.BertConfig.MaxPositionEmbeddings}, paddedInputData, false)

			optimizer.ZeroGrad()

			// Forward pass
			parentLogits, childLogits, err := model.Forward(inputTensor)
			if err != nil {
				log.Printf("Forward pass failed: %v", err)
				continue
			}

			// Calculate loss
			parentLoss, parentGrad := tensor.CrossEntropyLoss(parentLogits, batchParentTargetIDs, parentIntentVocabulary.PaddingTokenID)
			childLoss, childGrad := tensor.CrossEntropyLoss(childLogits, batchChildTargetIDs, childIntentVocabulary.PaddingTokenID)

			totalLoss := parentLoss + childLoss

			// Backward pass
			if err := model.Backward(parentGrad, childGrad); err != nil {
				log.Printf("Backward pass failed: %v", err)
				continue
			}

			optimizer.Step()
			log.Printf("Epoch: %d, Batch: %d, Loss: %f", epoch, i/batchSize, totalLoss)
		}
	}

	// Save the model
	if err := moemodel.SaveMoEClassificationModelToGOB(model, modelSavePath); err != nil {
		log.Fatalf("Failed to save model: %v", err)
	}

	fmt.Println("Training complete.")
}
