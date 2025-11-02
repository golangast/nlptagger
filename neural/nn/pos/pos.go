// Package pos provides functions for Part-of-Speech tagging using a neural network.
package pos

import (
	"fmt"

	"github.com/zendrulat/nlptagger/neural/nnu"
	"github.com/zendrulat/nlptagger/tagger/tag"
)

// Forward pass https://wiki.pathmind.com/neural-network
func ForwardPassPos(n *nnu.SimpleNN, inputs []float64) []float64 {
	// Create a slice to store the activations of the hidden layer.
	if n.WeightsIH == nil || len(n.WeightsIH) == 0 || len(inputs) == 0 {
		fmt.Printf("n.WeightsIH is nil or empty. len(n.WeightsIH): %d, n.InputSize: %d\n", len(n.WeightsIH), n.InputSize)

		return []float64{}

	}

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

// CreatePosTagVocab now returns both the original and reverse vocabularies
func CreatePosTagVocab(trainingData []tag.Tag) map[string]int {
	posTagVocab := make(map[string]int)
	index := 0

	for _, taggedSentence := range trainingData {
		for _, posTag := range taggedSentence.PosTag {
			if _, ok := posTagVocab[posTag]; !ok {
				posTagVocab[posTag] = index
				index++
			}
		}
	}

	return posTagVocab
}
