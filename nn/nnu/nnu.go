package nnu

import (
	"encoding/gob"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/golangast/nlptagger/nn/simplenn"
	"github.com/golangast/nlptagger/tagger/tag"
)

// Structure to represent training data in JSON
type TrainingDataJSON struct {
	Sentences []tag.Tag `json:"sentences"`
}

// Function to load training data from a JSON file
func LoadTrainingDataFromJSON(filePath string) (*TrainingDataJSON, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	data, err := ioutil.ReadAll(file)
	if err != nil {
		return nil, err
	}

	var trainingData TrainingDataJSON
	err = json.Unmarshal(data, &trainingData)
	if err != nil {
		return nil, err
	}

	return &trainingData, nil
}
func MaxIndex(values []float64) int {
	maxIndex := 0
	maxValue := values[0]
	for i, val := range values {
		if val > maxValue {
			maxValue = val
			maxIndex = i
		}
	}
	return maxIndex
}
func IndexToPosTag(posTagVocab map[string]int, predictedTagIndex int) (string, bool) {
	for tag, index := range posTagVocab {
		if index == predictedTagIndex {
			return tag, true
		}
	}
	return "", false
}

func CreateTokenVocab(trainingData []tag.Tag) map[string]int {
	tokenVocab := make(map[string]int)
	tokenVocab["UNK"] = 0 // Add "UNK" token initially
	index := 1
	for _, sentence := range trainingData { // Iterate through tag.Tag slice
		for _, token := range sentence.Tokens {
			if _, ok := tokenVocab[token]; !ok {
				tokenVocab[token] = index
				index++
			}
		}
	}

	// If index exceeded vocabulary size
	if index > len(tokenVocab)-1 { // Dynamically determine vocabulary size
		// Handle unknown tokens
		tokenVocab["UNK"] = len(tokenVocab) // Add "UNK" token
		index = len(tokenVocab)             // Update index to reflect new vocabulary size

		// Optionally, print a warning message
		//fmt.Println("Warning: Vocabulary size exceeded. Adding 'UNK' token.")
	}

	return tokenVocab
}

// Function to calculate accuracy
func calculateAccuracy(nn *simplenn.SimpleNN, trainingData []tag.Tag, tokenVocab map[string]int, posTagVocab map[string]int) float64 {
	correctPredictions := 0
	totalPredictions := 0

	for _, taggedSentence := range trainingData {
		for i := range taggedSentence.Tokens {
			inputs := make([]float64, nn.InputSize)
			tokenIndex, ok := tokenVocab[taggedSentence.Tokens[i]]
			if ok {
				inputs[tokenIndex] = 1
			} else {
				inputs[tokenVocab["UNK"]] = 1 // Handle unknown tokens
			}

			predictedTag := predict(nn, inputs, posTagVocab)

			if predictedTag == taggedSentence.PosTag[i] {
				correctPredictions++
			}
			totalPredictions++
		}
	}
	return float64(correctPredictions) / float64(totalPredictions)
}

// Training function
func Train(trainingData []tag.Tag, epochs int, learningRate float64, nn *simplenn.SimpleNN) float64 {
	var accuracy float64
	for epoch := 0; epoch < epochs; epoch++ {
		for _, taggedSentence := range trainingData {
			for i := range taggedSentence.Tokens {
				token := taggedSentence.Tokens[i]
				targetTag := taggedSentence.PosTag[i]

				// Convert token to one-hot encoded input
				inputs := make([]float64, nn.InputSize)
				tokenVocab := CreateTokenVocab(trainingData)
				tokenIndex, ok := tokenVocab[token]
				if ok {
					if _, ok := tokenVocab[token]; !ok {
						fmt.Printf("Token '%s' not found in vocabulary!\n", token)
					}
					inputs[tokenIndex] = 1
				}

				// Forward pass
				outputs := nn.ForwardPass(inputs)

				errors, posTagVocab := nn.CalculateError(targetTag, outputs, trainingData)

				nn.Backpropagate(errors, outputs, learningRate, inputs)

				// Calculate accuracy for this epoch
				accuracy = calculateAccuracy(nn, trainingData, tokenVocab, posTagVocab)
				//fmt.Printf("Epoch %d: Accuracy = %.2f%%\n", epoch+1, accuracy*100)
			}

		}

	}
	return accuracy
}

// Function to make a prediction
func predict(nn *simplenn.SimpleNN, inputs []float64, posTagVocab map[string]int) string {
	predictedOutput := nn.ForwardPass(inputs)
	predictedTagIndex := MaxIndex(predictedOutput)
	predictedTag, _ := IndexToPosTag(posTagVocab, predictedTagIndex)
	return predictedTag
}

func SaveModelToGOB(model *simplenn.SimpleNN, filePath string) error {
	file, err := os.Create(filePath)

	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	err = encoder.Encode(model)
	if err != nil {
		return err
	}

	return nil
}
func LoadModelFromGOB(filePath string) (*simplenn.SimpleNN, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	var model simplenn.SimpleNN
	err = decoder.Decode(&model)
	if err != nil {
		return nil, err
	}

	return &model, nil
}
