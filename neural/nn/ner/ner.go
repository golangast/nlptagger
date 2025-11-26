package ner

// Package ner implements a basic neural network for Named Entity Recognition.

import (
	"fmt"

	"github.com/zendrulat/nlptagger/neural/nnu"
	"github.com/zendrulat/nlptagger/tagger/tag"
)

// Forward pass https://wiki.pathmind.com/neural-network
func ForwardPassNer(n *nnu.SimpleNN, inputs []float64) []float64 {
	// Create a slice to store the activations of the hidden layer.
	hidden := make([]float64, n.HiddenSize)
	// Iterate over each neuron in the hidden layer.
	for i := range hidden {
		// Initialize the sum of weighted inputs for the current neuron.
		sum := 0.0
		// Iterate over each input neuron.
		for j := range inputs {
			// Calculate the weighted input and add it to the sum.
			sum += n.WeightsIH[i][j] * inputs[j]
		}
		// Apply the sigmoid activation function to the sum and store the result in the hidden layer.
		hidden[i] = n.Sigmoid(sum)
	}

	// Create a slice to store the activations of the output layer.
	output := make([]float64, n.OutputSize)
	// Iterate over each neuron in the output layer.
	for i := range output {
		// Initialize the sum of weighted inputs for the current neuron.
		sum := 0.0
		// Iterate over each neuron in the hidden layer.
		for j := range hidden {
			// Calculate the weighted input and add it to the sum.
			sum += n.WeightsHO[i][j] * hidden[j]
		}
		// Apply the sigmoid activation function to the sum and store the result in the output layer.
		output[i] = n.Sigmoid(sum)
	}

	// Return the activations of the output layer.
	return output
}

func CalculateErrorNer(nn *nnu.SimpleNN, targetTag string, outputs []float64, trainingData []tag.Tag) ([]float64, map[string]int) {
	// Create a slice to store the target output values.
	targetOutput := make([]float64, nn.OutputSize)
	nerTagVocab := CreateTagVocabNer(trainingData) // Use NER vocab
	targetNerTagIndex, ok := nerTagVocab[targetTag]
	// If the target POS tag is found in the vocabulary...
	if ok {
		// Set the corresponding element in the target output to 1.
		targetOutput[targetNerTagIndex] = 1
	}

	// Create a slice to store the errors for each output neuron.
	errors := make([]float64, nn.OutputSize)
	// Calculate the error for each output neuron.
	for i := range errors {
		errors[i] = targetOutput[i] - outputs[i]
	}
	return errors, nerTagVocab
}
func CreateTagVocabNer(trainingData []tag.Tag) map[string]int {
	nerTagVocab := make(map[string]int)
	index := 0

	for _, taggedSentence := range trainingData {
		for _, nerTag := range taggedSentence.NerTag {
			if _, ok := nerTagVocab[nerTag]; !ok { // Check if POS tag is already in the vocabulary
				nerTagVocab[nerTag] = index
				index++
			}
		}
	}

	return nerTagVocab
}

func IndexToNerTag(nerTagVocab map[string]int, predictedTagIndex int) (string, bool) {
	for tag, index := range nerTagVocab {
		if index == predictedTagIndex {
			return tag, true
		}
	}
	return "", false
}

func PredictNerTags(nn *nnu.SimpleNN, inputs []float64, nerTagVocab map[string]int, predictedNerTags []string, predictedTagIndex int) []string {
	// Get the actual NER tag string using the predicted index.
	predictedTagNer, ok := IndexToNerTag(nerTagVocab, predictedTagIndex)
	// If the predicted tag index is not found in the vocabulary...
	if !ok {
		// Print an error message.
		fmt.Printf("NER tag index %d not found in vocabulary\n", predictedTagIndex)
		// Append "O" (outside of any named entity) to the predicted NER tags.
		predictedNerTags = append(predictedNerTags, "O")
	}
	// Append the predicted tag to the list of predicted NER tags.
	predictedNerTags = append(predictedNerTags, predictedTagNer)
	return predictedNerTags
}