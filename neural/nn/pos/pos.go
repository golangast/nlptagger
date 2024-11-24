package pos

import (
	"fmt"

	"github.com/golangast/nlptagger/neural/nnu"
	"github.com/golangast/nlptagger/tagger/tag"
)

// Forward pass https://wiki.pathmind.com/neural-network
func ForwardPassPos(n *nnu.SimpleNN, inputs []float64) []float64 {
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
func IndexToPosTag(posTagVocab map[string]int, predictedTagIndex int) (string, bool) {
	for tag, index := range posTagVocab {
		if index == predictedTagIndex {
			return tag, true
		}
	}
	return "", false
}

func PredictPosTags(nn *nnu.SimpleNN, inputs []float64, posTagVocab map[string]int, predictedPosTags []string, predictedTagIndex int) []string {

	// Get the actual POS tag string using the predicted index.
	predictedTag, ok := IndexToPosTag(posTagVocab, predictedTagIndex)
	if !ok {
		// Print an error message.
		fmt.Printf("Tag index %d not found in vocabulary\n", predictedTagIndex)
		// Append "UNK" to the predicted tags.
		predictedPosTags = append(predictedPosTags, "UNK")
		// Continue to the next token.
	}
	// Append the predicted tag to the list of predicted tags.
	predictedPosTags = append(predictedPosTags, predictedTag)

	return predictedPosTags

}
