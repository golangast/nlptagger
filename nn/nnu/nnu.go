package nnu

import (
	"encoding/gob"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"os"

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

// Training function
func (nn *SimpleNN) Train(trainingData []tag.Tag, epochs int, learningRate float64) float64 {
	var accuracy float64
	for epoch := 0; epoch < epochs; epoch++ {
		for _, taggedSentence := range trainingData {
			for i := range taggedSentence.Tokens {
				token := taggedSentence.Tokens[i]
				targetTag := taggedSentence.PosTag[i]
				correctPredictions := 0
				totalPredictions := 0

				// Convert token to one-hot encoded input
				inputs := make([]float64, nn.InputSize)
				tokenVocab := CreateTokenVocab(trainingData)
				tokenIndex, ok := tokenVocab[token]
				if ok {
					//fmt.Printf("Token: %s, Token Index: %d, Input Size: %d\n", token, tokenIndex, len(inputs))
					if _, ok := tokenVocab[token]; !ok {
						fmt.Printf("Token '%s' not found in vocabulary!\n", token)
					}
					inputs[tokenIndex] = 1
				}

				// Forward pass
				outputs := nn.Predict(inputs)

				// Calculate error
				targetOutput := make([]float64, nn.OutputSize)
				posTagVocab := CreatePosTagVocab(trainingData)
				targetTagIndex, ok := posTagVocab[targetTag]
				if ok {
					targetOutput[targetTagIndex] = 1
				}
				errors := make([]float64, nn.OutputSize)
				for i := range errors {
					errors[i] = targetOutput[i] - outputs[i]
				}

				// Backpropagation
				// 1. Output layer weight updates
				for i := 0; i < nn.OutputSize; i++ {
					gradient := errors[i] * outputs[i] * (1 - outputs[i]) // Sigmoid derivative
					for j := 0; j < nn.HiddenSize; j++ {
						nn.WeightsHO[i][j] += learningRate * gradient * outputs[j] //Hidden Layer is input to Output Layer
					}
				}

				// 2. Hidden layer weight updates
				hiddenErrors := make([]float64, nn.HiddenSize)
				for i := 0; i < nn.HiddenSize; i++ {
					errorSum := 0.0
					for j := 0; j < nn.OutputSize; j++ {
						errorSum += errors[j] * nn.WeightsHO[j][i]
					}
					hiddenErrors[i] = errorSum * outputs[i] * (1 - outputs[i]) // Sigmoid derivative
				}
				for i := 0; i < nn.HiddenSize; i++ {
					for j := 0; j < nn.InputSize; j++ {
						nn.WeightsIH[i][j] += learningRate * hiddenErrors[i] * inputs[j] //Input Layer is input to Hidden Layer
					}
				}
				for _, taggedSentence := range trainingData {
					for i := range taggedSentence.Tokens {
						// ... (your existing code for forward pass, error calculation, backpropagation)

						// Accuracy Calculation
						predictedOutput := nn.Predict(inputs)
						predictedTagIndex := MaxIndex(predictedOutput)
						predictedTag, _ := IndexToPosTag(posTagVocab, predictedTagIndex) // Assuming indexToPosTag handles unknown tags

						if predictedTag == taggedSentence.PosTag[i] {
							correctPredictions++
						}
						totalPredictions++
					}
				}

				accuracy = float64(correctPredictions) / float64(totalPredictions)
				//fmt.Printf("Epoch %d: Accuracy = %.2f%%\n", epoch+1, accuracy*100)
			}

		}

	}
	return accuracy
}

func CreatePosTagVocab(trainingData []tag.Tag) map[string]int {
	posTagVocab := make(map[string]int)
	index := 0

	for _, taggedSentence := range trainingData {
		for _, posTag := range taggedSentence.PosTag {
			if _, ok := posTagVocab[posTag]; !ok { // Check if POS tag is already in the vocabulary
				posTagVocab[posTag] = index
				index++
			}
		}
	}

	return posTagVocab
}

// Forward pass
func (nn *SimpleNN) Predict(inputs []float64) []float64 {
	hidden := make([]float64, nn.HiddenSize)
	for i := range hidden {
		sum := 0.0
		for j := range inputs {
			sum += nn.WeightsIH[i][j] * inputs[j]
		}
		hidden[i] = sigmoid(sum)
	}

	output := make([]float64, nn.OutputSize)
	for i := range output {
		sum := 0.0
		for j := range hidden {
			sum += nn.WeightsHO[i][j] * hidden[j]
		}
		output[i] = sigmoid(sum)
	}

	return output
}

// Activation function (sigmoid)
func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

type SimpleNN struct {
	InputSize  int
	HiddenSize int
	OutputSize int
	WeightsIH  [][]float64 // Input to hidden weights
	WeightsHO  [][]float64 // Hidden to output weights
}

func NewSimpleNN(inputSize, hiddenSize, outputSize int) *SimpleNN {
	nn := &SimpleNN{
		InputSize:  inputSize,
		HiddenSize: hiddenSize,
		OutputSize: outputSize,
	}
	nn.WeightsIH = make([][]float64, hiddenSize)
	for i := range nn.WeightsIH {
		nn.WeightsIH[i] = make([]float64, inputSize)
		for j := range nn.WeightsIH[i] {
			nn.WeightsIH[i][j] = rand.Float64()*2 - 1 // Initialize with random weights
		}
	}
	nn.WeightsHO = make([][]float64, outputSize)
	for i := range nn.WeightsHO {
		nn.WeightsHO[i] = make([]float64, hiddenSize)
		for j := range nn.WeightsHO[i] {
			nn.WeightsHO[i][j] = rand.Float64()*2 - 1
		}
	}
	return nn
}
func SaveModelToGOB(model *SimpleNN, filePath string) error {
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
func LoadModelFromGOB(filePath string) (*SimpleNN, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	var model SimpleNN
	err = decoder.Decode(&model)
	if err != nil {
		return nil, err
	}

	return &model, nil
}
