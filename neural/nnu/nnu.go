package nnu

import (
	"math"
	"math/rand/v2"
)

type SimpleNN struct {
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

func (n *SimpleNN) Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
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

// Function to make a prediction

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
