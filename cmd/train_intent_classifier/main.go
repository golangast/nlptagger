package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
	"strings"

	moemodel "github.com/zendrulat/nlptagger/neural/moe/model"
	"github.com/zendrulat/nlptagger/neural/nn"
	mainvocab "github.com/zendrulat/nlptagger/neural/nnu/vocab"
	"github.com/zendrulat/nlptagger/neural/tensor"
	"github.com/zendrulat/nlptagger/neural/tokenizer"
)

// TrainingExample represents a single training example with input text and parent/child intents.
type TrainingExample struct {
	Query        string `json:"query"`
	ParentIntent string `json:"parent_intent"`
	ChildIntent  string `json:"child_intent"`
	Description  string `json:"description"`
	Sentence     string `json:"sentence"`
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
	rand.Seed(1) // For reproducibility
	log.Printf("Random number after seeding: %f", rand.Float64())
	// Define paths
	const trainingDataPath = "trainingdata/intent_data.json"
	const queryVocabPath = "gob_models/query_vocabulary.gob"
	const parentIntentVocabPath = "gob_models/parent_intent_vocabulary.gob"
	const childIntentVocabPath = "gob_models/child_intent_vocabulary.gob"
	const sentenceVocabPath = "gob_models/sentence_vocabulary.gob"
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
	sentenceVocabulary := mainvocab.NewVocabulary()

	for _, example := range *trainingData {
		words := strings.Fields(example.Sentence)
		for _, word := range words {
			queryVocabulary.AddToken(word)
			sentenceVocabulary.AddToken(word) // Also add to sentence vocab
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
	if err := sentenceVocabulary.Save(sentenceVocabPath); err != nil {
		log.Fatalf("Failed to save sentence vocabulary: %v", err)
	}

	// Create the model
	model, err := moemodel.NewMoEClassificationModel(
		queryVocabulary.Size(),
		256, // embeddingDim (increased from 128)
		parentIntentVocabulary.Size(),
		childIntentVocabulary.Size(),
		sentenceVocabulary.Size(),
		4,  // numExperts (increased from 2)
		2,  // k (increased from 1)
		32, // maxSeqLength (increased to match inference)
	)
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	// Train the model
	optimizer := nn.NewOptimizer(model.Parameters(), 0.001, 5.0)
	queryTokenizer, _ := tokenizer.NewTokenizer(queryVocabulary) // Initialize tokenizer once
	sentenceTokenizer, _ := tokenizer.NewTokenizer(sentenceVocabulary)

	log.Printf("Sentence Vocabulary PaddingTokenID: %d", sentenceVocabulary.PaddingTokenID)
	log.Printf("Sentence Vocabulary Size: %d", sentenceVocabulary.Size())

	const batchSize = 32 // Define batch size

	for epoch := 0; epoch < 100; epoch++ { // Increased epochs for better learning
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
			var batchSentenceTargetIDs [][]int
			currentBatchSize := len(batch)

			for _, example := range batch {
				tokenIDs, _ := queryTokenizer.Encode(example.Sentence)
				inputData := make([]float64, len(tokenIDs))
				for i, id := range tokenIDs {
					inputData[i] = float64(id)
				}
				batchInputIDs = append(batchInputIDs, inputData)
				batchParentTargetIDs = append(batchParentTargetIDs, parentIntentVocabulary.WordToToken[example.ParentIntent])
				batchChildTargetIDs = append(batchChildTargetIDs, childIntentVocabulary.WordToToken[example.ChildIntent])

				sentenceTokenIDs, _ := sentenceTokenizer.Encode(example.Sentence)
				batchSentenceTargetIDs = append(batchSentenceTargetIDs, sentenceTokenIDs)
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

			paddedSentenceData := make([]float64, currentBatchSize*model.BertConfig.MaxPositionEmbeddings)
			for rowIdx, seq := range batchSentenceTargetIDs {
				floatSeq := make([]float64, len(seq))
				for i, v := range seq {
					floatSeq[i] = float64(v)
				}
				copy(paddedSentenceData[rowIdx*model.BertConfig.MaxPositionEmbeddings:], floatSeq)
				for k := len(seq); k < model.BertConfig.MaxPositionEmbeddings; k++ {
					paddedSentenceData[rowIdx*model.BertConfig.MaxPositionEmbeddings+k] = float64(sentenceVocabulary.PaddingTokenID)
				}
			}

			inputTensor := tensor.NewTensor([]int{currentBatchSize, model.BertConfig.MaxPositionEmbeddings}, paddedInputData, false)
			targetSentenceTensor := tensor.NewTensor([]int{currentBatchSize, model.BertConfig.MaxPositionEmbeddings}, paddedSentenceData, false)

			optimizer.ZeroGrad()

			// Forward pass
			parentLogits, childLogits, sentenceOutputs, err := model.Forward(inputTensor, targetSentenceTensor)
			if err != nil {
				log.Printf("Forward pass failed: %+v", err)
				continue
			}

			// Calculate loss
			parentLoss, parentGrad := tensor.CrossEntropyLoss(parentLogits, batchParentTargetIDs, parentIntentVocabulary.PaddingTokenID, 0.0)
			childLoss, childGrad := tensor.CrossEntropyLoss(childLogits, batchChildTargetIDs, childIntentVocabulary.PaddingTokenID, 0.0)

			flatTargets := make([]int, 0, currentBatchSize*model.BertConfig.MaxPositionEmbeddings)
			for _, t := range batchSentenceTargetIDs {
				flatTargets = append(flatTargets, t...)
				// Pad the individual sentence targets to maxSeqLength
				for k := len(t); k < model.BertConfig.MaxPositionEmbeddings; k++ {
					flatTargets = append(flatTargets, sentenceVocabulary.PaddingTokenID)
				}
			}
			sentenceLoss, sentenceGrad := tensor.CrossEntropyLoss(sentenceOutputs, flatTargets, sentenceVocabulary.PaddingTokenID, 0.0)

			totalLoss := parentLoss + childLoss + sentenceLoss

			// Backward pass
			if err := model.Backward(parentGrad, childGrad, sentenceGrad); err != nil {
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
