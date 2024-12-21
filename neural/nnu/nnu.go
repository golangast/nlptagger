package nnu

import (
	"fmt"
	"math"
	"math/rand"
)

type SimpleNN struct {
	InputSize      int
	HiddenSize     int
	OutputSize     int
	OutputWeights  [][]float64
	HiddenWeights  [][]float64
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

func (nn *SimpleNN) NewSimpleNN(inputSize, hiddenSize, outputSize int, outputWeights [][]float64) *SimpleNN {
	nnn := &SimpleNN{
		InputSize:     inputSize,
		HiddenSize:    hiddenSize,
		OutputSize:    outputSize,
		OutputWeights: outputWeights,
	}
	nnn.WeightsIH = make([][]float64, hiddenSize)
	for i := range nnn.WeightsIH {
		nnn.WeightsIH[i] = make([]float64, inputSize)
		for j := range nnn.WeightsIH[i] {
			nnn.WeightsIH[i][j] = rand.Float64()*2 - 1 // Initialize with random weights between -1 and 1
		}
	}

	nnn.WeightsHO = make([][]float64, outputSize)
	for i := range nnn.WeightsHO {
		nnn.WeightsHO[i] = make([]float64, hiddenSize)
		for j := range nnn.WeightsHO[i] {
			nnn.WeightsHO[i][j] = rand.Float64()*2 - 1 // Initialize with random weights between -1 and 1
		}
	}

	nnn.HiddenWeights = make([][]float64, hiddenSize)
	for i := range nnn.HiddenWeights {
		nnn.HiddenWeights[i] = make([]float64, inputSize) // Assuming HiddenWeights should have same dimensions as WeightsIH
		// You can adjust this if HiddenWeights should have different dimensions.
	}

	nnn.OutputWeights = make([][]float64, nnn.HiddenSize) // Use nnn.HiddenSize
	for i := range nnn.OutputWeights {
		nnn.OutputWeights[i] = make([]float64, nnn.OutputSize) // Use nnn.OutputSize
		// ... initialization code ...
	}

	return nnn
}

// ForwardPass performs a forward pass through the neural network.
// It takes an input vector and returns the output vector.
func ForwardPass(nn *SimpleNN, inputs []float64) []float64 {
	// Calculate hidden layer activations
	hiddenOutputs := make([]float64, nn.HiddenSize)
	for i := 0; i < nn.HiddenSize; i++ {
		sum := 0.0
		for j := 0; j < nn.InputSize; j++ {
			sum += nn.WeightsIH[i][j] * inputs[j]
		}
		hiddenOutputs[i] = nn.Sigmoid(sum)
	}

	// Calculate output layer activations
	outputs := make([]float64, nn.OutputSize)
	for i := 0; i < nn.OutputSize; i++ {
		sum := 0.0
		for j := 0; j < nn.HiddenSize; j++ {
			sum += nn.WeightsHO[i][j] * hiddenOutputs[j]
		}
		outputs[i] = nn.Sigmoid(sum)
	}

	return outputs
}

func ForwardPassMLM(nn *SimpleNN, inputs []float64) []float64 {
	if len(inputs) != nn.InputSize {
		fmt.Printf("Input size in prepareMLMInput: %d\n", nn.InputSize)                   // Check the input size
		fmt.Printf("Length of inputs slice: %d\n", len(inputs))                           // Length of input vector
		fmt.Printf("Content of inputs slice: %v\n", inputs)                               // Check the input vector values
		fmt.Printf("Input size mismatch. Expected %d, got %d", nn.InputSize, len(inputs)) // Crucial error check
	}
	// Calculate hidden layer activations
	hiddenOutputs := make([]float64, nn.HiddenSize)
	for i := 0; i < nn.HiddenSize; i++ {
		sum := 0.0
		for j := 0; j < nn.InputSize; j++ { // Use nn.InputSize
			if i >= len(nn.WeightsIH) || j >= len(nn.WeightsIH[i]) {
				panic(fmt.Sprintf("WeightsIH index out of bounds i: %d j: %d", i, j))
			}
			sum += nn.WeightsIH[i][j] * inputs[j]
		}
		hiddenOutputs[i] = nn.Sigmoid(sum)
	}

	// Calculate output layer activations
	outputs := make([]float64, nn.OutputSize)

	// Add bias to each neuron in the output layer
	biasHO := make([]float64, nn.OutputSize)
	for i := range biasHO {
		biasHO[i] = 0.1 // Or another small positive value, or initialize differently if needed
	}

	for i := 0; i < nn.OutputSize; i++ {
		sum := biasHO[i] // Add the bias term
		for j := 0; j < nn.HiddenSize; j++ {
			if i >= len(nn.WeightsHO) || j >= len(nn.WeightsHO[i]) {
				panic(fmt.Sprintf("WeightsHO index out of bounds i: %d j: %d", i, j))
			}
			sum += nn.WeightsHO[i][j] * hiddenOutputs[j]
		}
		outputs[i] = nn.Sigmoid(sum)
		//fmt.Printf("Output %d: sum = %f, sigmoid(sum) = %f\n", i, sum, outputs[i])
	}

	return outputs
}
