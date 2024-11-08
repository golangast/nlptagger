package nn

import (
	"fmt"
	"os"
	"strings"

	"github.com/golangast/nlptagger/nn/nnu"
)

// Global variables for vocabularies
var tokenVocab map[string]int
var posTagVocab map[string]int

func LoadModelOrTrainNew(trainingData *nnu.TrainingDataJSON) (*nnu.SimpleNN, error) {
	nn, err := nnu.LoadModelFromGOB("trained_model.gob")
	if err != nil {
		fmt.Println("Error loading model, creating a new one:", err)
		inputSize := len(tokenVocab)
		hiddenSize := 5 // Adjust as needed
		outputSize := len(posTagVocab)
		nn = nnu.NewSimpleNN(inputSize, hiddenSize, outputSize)
		// Load training data
		trainingData, err := nnu.LoadTrainingDataFromJSON("data/training_data.json")
		if err != nil {
			fmt.Println("Error loading training data:", err)
		}
		// Train the network
		epochs := 100       // Adjust as needed
		learningRate := 0.1 // Adjust as needed
		accuracy := nn.Train(trainingData.Sentences, epochs, learningRate)
		fmt.Printf("Final Accuracy: %.2f%%\n", accuracy*100)
		// Save the newly trained model
		err = nnu.SaveModelToGOB(nn, "trained_model.gob")
		if err != nil {
			return nil, fmt.Errorf("error saving model: %w", err)
		}
		fmt.Println("New model saved to trained_model.gob")
	} else {
		fmt.Println("Loaded model from trained_model.gob")
	}
	return nn, nil
}
func PredictTags(nn *nnu.SimpleNN, sentence string) []string {
	// Tokenize the sentence into individual words.
	tokens := strings.Fields(sentence)
	// Create a slice to store the predicted POS tags.
	var predictedTags []string
	// Iterate over each token in the sentence.
	for _, token := range tokens {
		// Get the index of the token in the vocabulary.
		tokenIndex, ok := tokenVocab[token]
		// If the token is not in the vocabulary...
		if !ok {
			// Print a message indicating the token was not found.
			fmt.Printf("Token '%s' not found in vocabulary\n", token)
			// Assign "UNK" (unknown) as the POS tag for this token.
			predictedTags = append(predictedTags, "UNK")
			// Move on to the next token.
			continue
		}

		// Create the input vector for the neural network.
		inputs := make([]float64, nn.InputSize)
		inputs[tokenIndex] = 1 // Set the element corresponding to the token index to 1.

		// Get the predicted output from the neural network.
		predictedOutput := nn.Predict(inputs)
		// Get the index of the predicted POS tag.
		predictedTagIndex := nnu.MaxIndex(predictedOutput)
		// Get the actual POS tag string using the predicted index.
		predictedTag, ok := nnu.IndexToPosTag(posTagVocab, predictedTagIndex)
		// If the predicted tag index is not found in the vocabulary...
		if !ok {
			// Print an error message.
			fmt.Printf("Tag index %d not found in vocabulary\n", predictedTagIndex)
			// Append "UNK" to the predicted tags.
			predictedTags = append(predictedTags, "UNK")
			// Continue to the next token.
			continue
		}
		// Append the predicted tag to the list of predicted tags.
		predictedTags = append(predictedTags, predictedTag)
	}
	// Return the list of predicted POS tags.
	return predictedTags
}
func NN() *nnu.SimpleNN {
	// Delete existing model file
	if _, err := os.Stat("trained_model.gob"); err == nil {
		err = os.Remove("trained_model.gob")
		if err != nil {
			fmt.Println("Error deleting model file:", err)
		}
	}

	// Load training data
	trainingData, err := nnu.LoadTrainingDataFromJSON("data/training_data.json")
	if err != nil {
		fmt.Println("Error loading training data:", err)
	}

	// Create vocabularies
	tokenVocab = nnu.CreateTokenVocab(trainingData.Sentences)
	posTagVocab = nnu.CreatePosTagVocab(trainingData.Sentences)

	// Load or train the model
	nn, err := LoadModelOrTrainNew(trainingData)
	if err != nil {
		fmt.Println("Error:", err)
	}

	// Further training (if needed)
	epochs := 100       // Adjust as needed
	learningRate := 0.1 // Adjust as needed
	nn.Train(trainingData.Sentences, epochs, learningRate)

	// Save model
	err = nnu.SaveModelToGOB(nn, "trained_model.gob")
	if err != nil {
		fmt.Println("Error saving model:", err)
	}
	fmt.Println("Model saved to trained_model.gob")
	return nn
}
