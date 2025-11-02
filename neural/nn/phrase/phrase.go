// Package phrase provides a simple neural network for phrase tagging.

package phrase

import (
	"fmt"

	"github.com/zendrulat/nlptagger/neural/nnu"
	"github.com/zendrulat/nlptagger/tagger/tag"
)

type SimpleNNPhrase struct {
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

func CreatePhraseTagVocab(trainingData []tag.Tag) map[string]int {
	phraseTagVocab := make(map[string]int)
	index := 0

	for _, taggedSentence := range trainingData {
		for _, phraseTag := range taggedSentence.PhraseTag {
			if _, ok := phraseTagVocab[phraseTag]; !ok { // Check if POS tag is already in the vocabulary
				phraseTagVocab[phraseTag] = index
				index++
			}
		}
	}

	return phraseTagVocab
}

func IndexToPhraseTag(phraseTagVocab map[string]int, predictedTagIndex int) (string, bool) {
	for tag, index := range phraseTagVocab {
		if index == predictedTagIndex {
			return tag, true
		}
	}
	return "", false
}

// Forward pass https://wiki.pathmind.com/neural-network
func ForwardPassPhrase(n *nnu.SimpleNN, inputs []float64) []float64 {
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
func PredictPhraseTags(nn *nnu.SimpleNN, inputs []float64, phraseTagVocab map[string]int, predictedPhraseTags []string, predictedTagIndex int) []string {

	if predictedTagIndex != -1 { //check for valid index
		predictedTagPhrase, ok := IndexToPhraseTag(phraseTagVocab, predictedTagIndex)
		if !ok {
			fmt.Printf("Phrase tag index %d not found in vocabulary\n", predictedTagIndex)
			predictedPhraseTags = append(predictedPhraseTags, "O")
		}
		predictedPhraseTags = append(predictedPhraseTags, predictedTagPhrase)

	} else { //handle invalid index
		predictedPhraseTags = append(predictedPhraseTags, "O") // Default to "O" if no valid index
		fmt.Printf("predictedOutput is empty or length 0\n")
	}

	return predictedPhraseTags
}
