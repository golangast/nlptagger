package simplenn

import (
	"math"
	"math/rand/v2"

	"github.com/golangast/nlptagger/tagger/tag"
)

type SimpleNN struct {
	InputSize  int
	HiddenSize int
	OutputSize int
	WeightsIH  [][]float64
	WeightsHO  [][]float64
}

// Activation function (sigmoid)
func (nn SimpleNN) Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// Forward pass
func (nn *SimpleNN) ForwardPass(inputs []float64) []float64 {
	hidden := make([]float64, nn.HiddenSize)
	for i := range hidden {
		sum := 0.0
		for j := range inputs {
			sum += nn.WeightsIH[i][j] * inputs[j]
		}
		hidden[i] = nn.Sigmoid(sum)
	}

	output := make([]float64, nn.OutputSize)
	for i := range output {
		sum := 0.0
		for j := range hidden {
			sum += nn.WeightsHO[i][j] * hidden[j]
		}
		output[i] = nn.Sigmoid(sum)
	}

	return output
}
func (nn *SimpleNN) CalculateError(targetTag string, outputs []float64, trainingData []tag.Tag) ([]float64, map[string]int) {
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
	return errors, posTagVocab
}

func (nn *SimpleNN) Backpropagate(errors []float64, outputs []float64, learningRate float64, inputs []float64) {
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

func (nn *SimpleNN) NewSimpleNN(inputSize, hiddenSize, outputSize int) *SimpleNN {
	nnn := &SimpleNN{
		InputSize:  inputSize,
		HiddenSize: hiddenSize,
		OutputSize: outputSize,
	}
	nnn.WeightsIH = make([][]float64, hiddenSize)
	for i := range nnn.WeightsIH {
		nnn.WeightsIH[i] = make([]float64, inputSize)
		for j := range nnn.WeightsIH[i] {
			nnn.WeightsIH[i][j] = rand.Float64()*2 - 1 // Initialize with random weights
		}
	}
	nnn.WeightsHO = make([][]float64, outputSize)
	for i := range nnn.WeightsHO {
		nnn.WeightsHO[i] = make([]float64, hiddenSize)
		for j := range nnn.WeightsHO[i] {
			nnn.WeightsHO[i][j] = rand.Float64()*2 - 1
		}
	}
	return nnn
}
