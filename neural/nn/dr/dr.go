// Package dr implements a neural network for dependency relation tagging.

package dr

import (
	"fmt"

	"github.com/zendrulat/nlptagger/neural/nn/ner"
	"github.com/zendrulat/nlptagger/neural/nnu"
	"github.com/zendrulat/nlptagger/tagger/tag"
)

type SimpleNNDR struct {
	InputSize      int
	HiddenSize     int
	OutputSize     int
	WeightsIH      [][]float64
	WeightsHO      [][]float64
	TokenVocab     map[string]int
	PosTagVocab    map[string]int
	NerTagVocab    map[string]int
	PhraseTagVocab map[string]int
	DependencyTag  map[string]int
}

func CreateDRTagVocab(tags []tag.Tag) map[string]int {
	drTagVocab := make(map[string]int)
	index := 0

	for _, tag := range tags {
		//fmt.Printf("Sentence %d: Dependencies: %+v\n", i, tag.DepRelationsTag) // Print the Dependencies field

		for _, drt := range tag.Dependencies {
			//fmt.Println(j, drt.Head, drt.Dep) // Print relation details
			if _, ok := drTagVocab[drt.Dep]; !ok {
				drTagVocab[drt.Dep] = index
				index++
			}
		}
	}
	return drTagVocab
}

func IndexToDRTag(drTagVocab map[string]int, predictedTagIndex int) (string, bool) {
	for tag, index := range drTagVocab {
		if index == predictedTagIndex {
			return tag, true
		}
	}
	return "", false
}

// Forward pass https://wiki.pathmind.com/neural-network
func ForwardPassDR(n *nnu.SimpleNN, inputs []float64) []float64 {
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
func PredictDRTags(nn *nnu.SimpleNN, inputs []float64, drTagVocab map[string]int, predictedDRTags []string, predictedTagIndex int) []string {

	if predictedTagIndex != -1 {
		predictedTagDR, ok := ner.IndexToNerTag(drTagVocab, predictedTagIndex)

		if !ok {
			fmt.Printf("DR tag index %d not found in vocabulary\n", predictedTagIndex)
			predictedDRTags = append(predictedDRTags, "O")
		}
		predictedDRTags = append(predictedDRTags, predictedTagDR)
	} else {
		predictedDRTags = append(predictedDRTags, "O") // Default to "O" if no valid index
		fmt.Printf("predictedOutput is empty or length 0\n")
	}
	return predictedDRTags
}