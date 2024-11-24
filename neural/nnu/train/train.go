package train

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/golangast/nlptagger/neural/nn/dr"
	"github.com/golangast/nlptagger/neural/nn/ner"
	"github.com/golangast/nlptagger/neural/nn/phrase"
	"github.com/golangast/nlptagger/neural/nn/pos"
	"github.com/golangast/nlptagger/neural/nnu"
	"github.com/golangast/nlptagger/neural/nnu/calc"
	"github.com/golangast/nlptagger/neural/nnu/gobs"
	"github.com/golangast/nlptagger/tagger/tag"
)

// Structure to represent training data in JSON
type TrainingDataJSON struct {
	Sentences []tag.Tag `json:"sentences"`
}

// Training function
func Train(trainingData []tag.Tag, epochs int, learningRate float64, nn *nnu.SimpleNN) (float64, float64, float64, float64) {
	var posaccuracy float64
	var neraccuracy float64
	var phraseaccuracy float64
	var draccuracy float64
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
				outputs := pos.ForwardPassPos(nn, inputs)

				errors, posTagVocab, nerTagVocab, phraseTagVocab, drTagVocab := calc.CalculateError(targetTag, outputs, trainingData, nn)

				nn.Backpropagate(errors, outputs, learningRate, inputs)

				// Calculate accuracy for this epoch
				posaccuracy, neraccuracy, phraseaccuracy, draccuracy = calc.CalculateAccuracy(nn, trainingData, tokenVocab, posTagVocab, nerTagVocab, phraseTagVocab, drTagVocab)
				// fmt.Printf("Epoch %d: POS Accuracy = %.2f%%\n", epoch+1, posaccuracy*100)
				// fmt.Printf("Epoch %d: NER Accuracy = %.2f%%\n", epoch+1, neraccuracy*100)
				// fmt.Printf("Epoch %d: PHRASE Accuracy = %.2f%%\n", epoch+1, phraseaccuracy*100)
				//fmt.Printf("Epoch %d: Dependency Accuracy = %.2f%%\n", epoch+1, draccuracy*100)
			}

		}

	}
	return posaccuracy, neraccuracy, phraseaccuracy, draccuracy
}

// trainAndSaveModel trains a new neural network model, or loads an existing one if available.
// It then saves the trained model to a file.
func TrainAndSaveModel(trainingData *TrainingDataJSON, modeldirectory string) (*nnu.SimpleNN, error) {
	// Delete existing model file if it exists.
	if _, err := os.Stat("trained_model.gob"); err == nil {
		// If the file exists, remove it.
		if err := os.Remove("trained_model.gob"); err != nil {
			// If there's an error during removal, return an error.
			return nil, fmt.Errorf("error deleting model file: %w", err)
		}
	}

	// Load or train the neural network model.
	nn, err := LoadModelOrTrainNew(trainingData, modeldirectory)
	if err != nil {
		// If there's an error during loading or training, return an error.
		return nil, fmt.Errorf("error loading or training model: %w", err)
	}

	// Further training (if needed).
	epochs := 100       // Adjust the number of epochs as needed.
	learningRate := 0.1 // Adjust the learning rate as needed.
	// Train the model using the training data.
	Train(trainingData.Sentences, epochs, learningRate, nn)

	// Save the trained model to a file.
	if err := gobs.SaveModelToGOB(nn, "trained_model.gob"); err != nil {
		// If there's an error during saving, return an error.
		return nil, fmt.Errorf("error saving model: %w", err)
	}
	// Print a message indicating that the model has been saved.
	fmt.Println("Model saved to trained_model.gob")
	// Return the trained neural network model and nil error.
	return nn, nil
}

func LoadModelOrTrainNew(trainingData *TrainingDataJSON, modeldirectory string) (*nnu.SimpleNN, error) {
	tokenVocab, posTagVocab, _, _, _, _ := CreateVocab()
	nn, err := gobs.LoadModelFromGOB("trained_model.gob")
	if err != nil {
		fmt.Println("Error loading model, creating a new one:", err)
		inputSize := len(tokenVocab)
		hiddenSize := 5 // Adjust as needed
		outputSize := len(posTagVocab)
		nn = nn.NewSimpleNN(inputSize, hiddenSize, outputSize)
		// Load training data
		trainingData, err := LoadTrainingDataFromJSON(modeldirectory)
		if err != nil {
			fmt.Println("Error loading training data:", err)
		}
		// Train the network
		epochs := 100       // Adjust as needed
		learningRate := 0.1 // Adjust as needed
		posaccuracy, neraccuracy, phraseaccuracy, draccuracy := Train(trainingData.Sentences, epochs, learningRate, nn)
		fmt.Printf("Final POS Accuracy: %.2f%%\n", posaccuracy*100)
		fmt.Printf("Final NER Accuracy: %.2f%%\n", neraccuracy*100)
		fmt.Printf("Final Phrase Accuracy: %.2f%%\n", phraseaccuracy*100)
		fmt.Printf("Final Dependency relation Accuracy: %.2f%%\n", draccuracy*100)
		// Save the newly trained model
		err = gobs.SaveModelToGOB(nn, "trained_model.gob")
		if err != nil {
			return nil, fmt.Errorf("error saving model: %w", err)
		}
		fmt.Println("New model saved to trained_model.gob")
	} else {
		fmt.Println("Loaded model from trained_model.gob")
	}
	return nn, nil
}

// Function to load training data from a JSON file
func LoadTrainingDataFromJSON(filePath string) (*TrainingDataJSON, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}

	data, err := ioutil.ReadAll(file)
	if err != nil {
		return nil, err
	}

	var trainingData TrainingDataJSON
	err = json.Unmarshal(data, &trainingData)
	if err != nil {
		return nil, err
	}
	file.Close()

	return &trainingData, nil
}

func CreateVocab() (map[string]int, map[string]int, map[string]int, map[string]int, map[string]int, *TrainingDataJSON) {
	trainingData, err := LoadTrainingDataFromJSON("data/training_data.json")
	if err != nil {
		fmt.Println("error loading training data: %w", err)
	}
	// Create vocabularies
	tokenVocab := CreateTokenVocab(trainingData.Sentences)
	posTagVocab := pos.CreatePosTagVocab(trainingData.Sentences)
	nerTagVocab := ner.CreateTagVocabNer(trainingData.Sentences)
	phraseTagVocab := phrase.CreatePhraseTagVocab(trainingData.Sentences)
	drTagVocab := dr.CreateDRTagVocab(trainingData.Sentences)

	return tokenVocab, posTagVocab, nerTagVocab, phraseTagVocab, drTagVocab, trainingData
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
