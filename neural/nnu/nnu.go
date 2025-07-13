// Package nnu implements simple neural network structures and operations.
package nnu

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"os"
	"strings"
	"time"
)

type Sentence struct {
	Text string `json:"text"`
}

// SimpleNN represents a simple feedforward neural network.
type SimpleNN struct {
	// InputSize is the number of input neurons.
	InputSize int
	// HiddenSize is the number of hidden neurons.
	HiddenSize int
	// OutputSize is the number of output neurons.
	OutputSize int
	// Add these fields to store intermediate values for backpropagation
	HiddenOutputs []float64 // Store the outputs of the hidden layer

	// OutputWeights is a matrix of weights from hidden layer to output layer.
	OutputWeights [][]float64
	// HiddenWeights is a matrix of weights from input layer to hidden layer.
	HiddenWeights [][]float64

	WeightsIH [][]float64
	// WeightsHO is the matrix of weights connecting hidden layer to output layer.
	WeightsHO [][]float64
	// HiddenBiases is the biases for each neuron in the hidden layer.
	HiddenBiases []float64
	// OutputBiases is the biases for each neuron in the output layer.
	OutputBiases []float64
	// Inputs holds the current input data.
	Inputs  []float64
	Outputs []float64
	// MaskedInputs holds a subset of the input data, selected by a mask.
	Sentences    []Sentence
	MaskedInputs []float64
	// Targets holds the target output data.
	Targets []float64
	// MaskedIndices holds the indices of the input data that are masked.
	MaskedIndices []int
	// TokenVocab is a vocabulary of tokens mapped to integers.
	TokenVocab map[string]int
	// PosTagVocab is a vocabulary of part-of-speech tags mapped to integers.
	PosTagVocab map[string]int
	// NerTagVocab is a vocabulary of named entity recognition tags mapped to integers.
	NerTagVocab map[string]int
	// PhraseTagVocab is a vocabulary of phrase tags mapped to integers.
	PhraseTagVocab map[string]int
	// DependencyTag is a vocabulary of dependency tags mapped to integers.
	DependencyTag map[string]int

	PPosTag          []string
	PNerTag          []string
	PPhraseTag       []string
	PTokens          []string
	PDepRelationsTag []string
	PFeatures        []PredictedFeatures
	PDependencies    []PredictedDependency `json:"dependencies"` // Or "dependencyTag" if that's what the key is in your json
	PEpoch           int
	PIsName          bool
	PToken           string
	HiddenErrors     []float64
	OutputErrors     []float64
	LearningRate     float64
	PSentence        string
	PSentences       []string
	SimpleNN         []SimpleNN
}

type PredictedFeatures struct {
	WebServerKeyword  float64
	PreviousWord      float64
	NextWord          float64
	PreviousArticle   float64
	NextPreposition   float64
	NextOfIn          float64
	SpecialCharacters float64
	NameSuffix        float64
	PreviousTag       float64
	NextTag           float64
	FollowedByNumber  float64
	IsNoun            float64
}

type PredictedDependency struct {
	Dependent int    `json:"dependent"`
	Relation  string `json:"relation"`
	Head      int    `json:"head"`
	Dep       string `json:"dep"` // or Dependent if that's what your JSON has

}

// Sigmoid is a utility function that applies the sigmoid activation function to a single float64 value.
func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// Sigmoid applies the sigmoid activation function to a float64 value.
// It is a method attached to the SimpleNN struct, so it can be used like nn.Sigmoid(x).
func (n *SimpleNN) Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// MaxIndex returns the index of the maximum value in a slice of float64.
// It iterates through the slice and keeps track of the maximum value and its index.
func MaxIndex(values []float64) int {
	maxIndex := 0
	maxValue := values[0]
	for i, val := range values {
		if val > maxValue {
			maxIndex = i //Update maxIndex to the current index i
		}
	}
	return maxIndex
}

// Backpropagate performs the backpropagation algorithm to update the neural network's weights and biases.
// It calculates the error gradient and adjusts the weights and biases accordingly to minimize the error.
// `
// Parameters:
//   - errors: A slice of float64 representing the errors calculated at the output layer.
//   - outputs: A slice of float64 representing the outputs from both hidden and output layers.
//   - learningRate: A float64 representing the learning rate for weight and bias updates.
//   - inputs: A slice of float64 representing the input data to the network.
func (nn *SimpleNN) Backpropagate(totalloss, learningRate float64) {
	nn.LearningRate = learningRate
	fmt.Printf("Backpropagate - Learning Rate: %f, Total Loss: %f\n", nn.LearningRate, totalloss)
	if nn.WeightsIH == nil || len(nn.WeightsIH) == 0 {
		log.Println("Backpropagate - WeightsIH are nil or empty.")
	}
	hiddenOutputs := nn.calculateHiddenLayerOutputs()
	nn.OutputErrors = make([]float64, nn.OutputSize)
	nn.HiddenErrors = make([]float64, nn.HiddenSize)

	// Calculate output errors
	for i := 0; i < nn.OutputSize; i++ {
		nn.OutputErrors[i] = (nn.Targets[i] - nn.Outputs[i]) * nn.Outputs[i] * (1 - nn.Outputs[i])
	}

	// Calculate hidden errors
	for i := 0; i < nn.HiddenSize; i++ {
		var sum float64
		for j := 0; j < nn.OutputSize; j++ {
			sum += nn.WeightsHO[j][i] * nn.OutputErrors[j]
		}
		nn.HiddenErrors[i] = sum * hiddenOutputs[i] * (1 - hiddenOutputs[i])
	}
	nn.UpdateWeights()
}

func (nn *SimpleNN) UpdateWeights() {
	// Check for nil or empty weights
	if nn.WeightsHO == nil || len(nn.WeightsHO) == 0 {
		log.Println("UpdateWeights - WeightsHO are nil or empty.")
		return
	}
	if nn.WeightsIH == nil || len(nn.WeightsIH) == 0 {
		log.Println("UpdateWeights - WeightsIH are nil or empty.")
		return
	}
	// Get the hidden layer outputs
	hiddenOutputs := nn.calculateHiddenLayerOutputs()

	// Update output layer weights and biases
	for i := 0; i < nn.OutputSize; i++ {
		for j := 0; j < nn.HiddenSize; j++ {
			nn.WeightsHO[i][j] += nn.LearningRate * nn.OutputErrors[i] * hiddenOutputs[j]
		}
		nn.OutputBiases[i] += nn.LearningRate * nn.OutputErrors[i]
	}

	// Update hidden layer weights and biases
	for i := 0; i < nn.HiddenSize; i++ {
		for j := 0; j < nn.InputSize; j++ {
			nn.WeightsIH[i][j] += nn.LearningRate * nn.HiddenErrors[i] * nn.Inputs[j]
		}
		nn.HiddenBiases[i] += nn.LearningRate * nn.HiddenErrors[i]
	}
}

// NewSimpleNN initializes a new SimpleNN with the specified input, hidden, and output sizes.
// It allocates and initializes the weights and biases of the network.
func NewSimpleNN(filePath string) *SimpleNN {
	trainingData, err := LoadTrainingDataJSON(filePath)
	if err != nil {
		fmt.Println("error loading training data: %w", err)
	}
	nnn := SimpleNN{}
	// Only take the first sentence to set the sizes, all sentences should be the same length
	for _, sentence := range trainingData.Sentences {
		nnn.InputSize = 200 // Assuming we are just going to use this for a single sentence at a time.
		//nnn.InputSize = len(sentence.Tokens)
		nnn.Inputs = make([]float64, nnn.InputSize)
		// Assuming we are just going to use this for a single sentence at a time.
		nnn.WeightsIH = NewMatrix(nnn.HiddenSize, nnn.InputSize)
		nnn.WeightsHO = NewMatrix(nnn.OutputSize, nnn.HiddenSize)
		nnn.Targets = make([]float64, len(sentence.Tokens)) // These may not be appropriate here.
		nnn.Outputs = make([]float64, len(sentence.Tokens)) // These may not be appropriate here.
		nnn.PPosTag = sentence.PosTag                       // These may not be appropriate here.
		nnn.PNerTag = sentence.NerTag                       // These may not be appropriate here.
		nnn.PPhraseTag = sentence.PhraseTag                 // These may not be appropriate here.
		nnn.PTokens = sentence.Tokens                       // These may not be appropriate here.
		nnn.PSentence = strings.Join(sentence.Tokens, " ")
		nnn.PSentences = append(nnn.PSentences, nnn.PSentence)
		nnn.PDependencies = sentence.Dependencies // These may not be appropriate here.
	}

	if nnn.WeightsIH == nil {
		for i := range nnn.WeightsIH {
			nnn.WeightsIH[i] = make([]float64, nnn.InputSize)
		}
	}

	nnn.WeightsIH = NewMatrix(nnn.HiddenSize, nnn.InputSize)

	nnn.WeightsHO = make([][]float64, nnn.OutputSize)
	for i := range nnn.WeightsHO {
		nnn.WeightsHO[i] = make([]float64, nnn.HiddenSize)
	}
	nnn.initWeights()
	nnn.OutputBiases = make([]float64, nnn.OutputSize)
	rand.Seed(time.Now().UnixNano())
	for i := range nnn.OutputBiases {
		nnn.OutputBiases[i] = (rand.Float64() * 2) - 1
	}

	return &nnn
}

func (nn *SimpleNN) initWeights() {
	rand.Seed(time.Now().UnixNano())
	nn.WeightsHO = make([][]float64, nn.OutputSize)
	for i := range nn.WeightsHO {
		nn.WeightsHO[i] = make([]float64, nn.HiddenSize)
		for j := range nn.WeightsHO[i] {
			nn.WeightsHO[i][j] = (rand.Float64() * 2) - 1 // Assign random values between -1 and 1
		}
	}

}

func (nn *SimpleNN) Randomize() *SimpleNN {

	nn.HiddenBiases = make([]float64, nn.HiddenSize)
	nn.OutputBiases = make([]float64, nn.OutputSize)

	nn.WeightsIH = make([][]float64, nn.HiddenSize)
	for i := range nn.WeightsIH {
		nn.WeightsIH[i] = make([]float64, nn.InputSize)
	}
	nn.WeightsHO = make([][]float64, nn.OutputSize)
	for i := range nn.WeightsHO {
		nn.WeightsHO[i] = make([]float64, nn.HiddenSize)
	}
	for i := range nn.WeightsIH {
		//Randomize the values of WeightsIH
		for j := range nn.WeightsIH[i] {
			nn.WeightsIH[i][j] = rand.Float64()*0.02 - 0.01
		}
	}
	//Randomize the values of WeightsHO
	for i := range nn.WeightsHO {
		for j := range nn.WeightsHO[i] {
			nn.WeightsHO[i][j] = rand.Float64()*0.02 - 0.01
		}
	}
	//Randomize the values of HiddenBiases
	for i := range nn.HiddenBiases {
		nn.HiddenBiases[i] = rand.Float64()*0.02 - 0.01
	}
	//Randomize the values of OutputBiases
	for i := range nn.OutputBiases {
		nn.OutputBiases[i] = rand.Float64()*0.02 - 0.01
	}
	return nn
}

// ForwardPass performs a forward pass through the neural network with the given input data.
// It calculates the output of the network based on the current weights and biases.
//
// Parameters:
//   - inputs: A slice of float64 representing the input data.
//
// Returns:
//   - A slice of float64 representing the output values from the output layer.
func (nn *SimpleNN) ForwardPass(inputs []float64) []float64 {
	if len(inputs) != nn.InputSize {
		fmt.Printf("Warning: ForwardPass received input slice of length %d, expected %d.", len(inputs), nn.InputSize)
	}

	// Error handling for empty input slice
	if len(inputs) == 0 {
		fmt.Println("Warning: ForwardPass received an empty input slice.")
		return nil
	}

	// Use masked inputs if available, otherwise use regular inputs
	var effectiveInputs []float64
	if len(nn.MaskedInputs) > 0 {
		effectiveInputs = nn.MaskedInputs
	} else {
		effectiveInputs = inputs
	}

	// Error handling if effectiveInputs is still empty after checking MaskedInputs
	if len(effectiveInputs) == 0 {
		fmt.Println("Warning: No valid input data found for ForwardPass.")
		return nil
	}

	// Check if nn.HiddenOutputs is empty and log an error if so. Added to prevent out of index error in ForwardPass method

	hiddenOutputs := make([]float64, nn.HiddenSize)
	outputs := make([]float64, nn.OutputSize)

	for i := 0; i < nn.HiddenSize; i++ {
		var sum float64
		if i < len(nn.HiddenBiases) {
			sum = nn.HiddenBiases[i]
		}

		//check if nn.WeightsIH has i rows
		if i >= len(nn.WeightsIH) {
			fmt.Printf("Warning: i is out of range for nn.WeightsIH. i=%d, len(nn.WeightsIH)=%d. Skipping.\n", i, len(nn.WeightsIH))
			continue
		}
		for j := 0; j < len(nn.WeightsIH[i]); j++ {
			//add a check for the len of effective inputs
			if j >= len(effectiveInputs) {
				fmt.Printf("Warning: j is out of range for effectiveInputs. j=%d, len(effectiveInputs)=%d. Skipping.\n", j, len(effectiveInputs))
				continue
			}
			sum += nn.WeightsIH[i][j] * effectiveInputs[j]
		}
		hiddenOutputs[i] = nn.Sigmoid(sum)
	}
	for i := 0; i < nn.OutputSize; i++ {
		//Check if i is a valid row for nn.WeightsHO
		if i >= len(nn.WeightsHO) {
			log.Printf("Warning: i is out of range for nn.WeightsHO. i=%d, len(nn.WeightsHO)=%d. Skipping.\n", i, len(nn.WeightsHO))
			continue
		}

		sum := nn.OutputBiases[i]
		for j := range hiddenOutputs {
			//Check if j is a valid column for nn.WeightsHO[i]
			if j >= len(nn.WeightsHO[i]) {
				fmt.Printf("Warning: j is out of range for nn.WeightsHO[i]. j=%d, len(nn.WeightsHO[i])=%d. Skipping.\n", j, len(nn.WeightsHO[i]))
				continue
			}
			//Check if j is a valid index for hiddenOutputs
			if j >= len(hiddenOutputs) {
				fmt.Printf("Warning: j is out of range for hiddenOutputs. j=%d, len(hiddenOutputs)=%d. Skipping.\n", j, len(hiddenOutputs))
				continue
			}
			sum += nn.WeightsHO[i][j] * hiddenOutputs[j]
		}
		outputs[i] = nn.Sigmoid(sum)
	}
	nn.Outputs = outputs
	return outputs
}

// calculateHiddenLayerOutputs calculates the outputs of the hidden layer neurons.
//
// Parameters:
//   - nn: A pointer to a SimpleNN struct representing the neural network.
//
// Returns:
//   - A slice of float64 representing the output values from the hidden layer.
func (nn *SimpleNN) calculateHiddenLayerOutputs() []float64 {
	hiddenOutputs := make([]float64, nn.HiddenSize)

	var sum float64
	for i := 0; i < nn.HiddenSize; i++ {
		if len(nn.MaskedInputs) == 0 {
			return []float64{}
		}

		if nn.WeightsIH == nil || len(nn.WeightsIH) == 0 {
			return []float64{}
		}

		if len(nn.HiddenBiases) > i {
			sum = nn.HiddenBiases[i]
		} else {
			sum = 0.0
		}

		for j := 0; j < len(nn.MaskedInputs); j++ {
			// check for out of bounds
			if i >= len(nn.WeightsIH) || i < 0 || j >= len(nn.WeightsIH[i]) || j < 0 {
				fmt.Printf("Warning: Index i or j out of range for nn.WeightsIH. i=%d, j=%d, len(nn.WeightsIH)=%d, len(nn.WeightsIH[i])=%d, i < 0=%t, j < 0=%t. Skipping.\n", i, j, len(nn.WeightsIH), len(nn.WeightsIH[i]), i < 0, j < 0)
				continue // Skip this iteration
			}
			if j >= len(nn.MaskedInputs) {
				fmt.Println("Warning: j is out of range for nn.MaskedInputs.")
				continue
			}
			sum += nn.WeightsIH[i][j] * nn.MaskedInputs[j]
		}
		hiddenOutputs[i] = nn.Sigmoid(sum)

	}
	return hiddenOutputs
}

// calculateOutputLayerOutputs calculates the outputs of the output layer neurons.
//
// Parameters:
//   - nn: A pointer to a SimpleNN struct representing the neural network.
//   - hiddenOutputs: A slice of float64 representing the outputs from the hidden layer.
//
// Returns:
//   - A slice of float64 representing the output values from the output layer.
func (nn *SimpleNN) calculateOutputLayerOutputs(hiddenOutputs []float64) []float64 {
	outputs := make([]float64, nn.OutputSize) // Initialize the output slice with the correct size.
	if len(hiddenOutputs) == 0 || len(nn.WeightsHO) == 0 {
		// Return an empty slice if hiddenOutputs is empty.
		return []float64{}
	}
	for i := 0; i < nn.OutputSize; i++ {
		sum := nn.OutputBiases[i] // Initialize the sum with the bias of the current output neuron.
		for j := 0; j < nn.HiddenSize; j++ {
			sum += nn.WeightsHO[i][j] * hiddenOutputs[j] // Accumulate the weighted sum from hidden layer outputs.
		}
		outputs[i] = nn.Sigmoid(sum) // Apply the sigmoid activation function to the sum.
	}
	return outputs
}

// ForwardPassMasked performs a forward pass through the network using masked inputs.
// It's similar to ForwardPass but operates on a subset of the input data, `nn.MaskedInputs`.
//
// Returns:
//   - A slice of float64 representing the output values from the output layer.
//   - An error if `nn.MaskedInputs` is empty.
//
// This method is associated with the SimpleNN struct.
func (nn *SimpleNN) ForwardPassMasked() ([]float64, error) {
	if len(nn.MaskedInputs) == 0 {
		return nil, errors.New("Error: Masked inputs are empty.")
	}
	if len(nn.WeightsHO) == 0 {
		return []float64{}, nil
	}

	hiddenOutputs := nn.calculateHiddenLayerOutputs()
	outputs := nn.calculateOutputLayerOutputs(hiddenOutputs)
	nn.Outputs = []float64{}
	return outputs, nil
}

// ForwardPassMasked performs a forward pass through the network using masked inputs. It
// calculates the output of the neural network based on the masked input data.
func ForwardPassMasked(nn *SimpleNN) []float64 {
	// Check for empty masked inputs and handle the error

	hiddenOutputs := nn.calculateHiddenLayerOutputs()
	outputs := nn.calculateOutputLayerOutputs(hiddenOutputs)

	return outputs
}
func LoadTrainingDataJSON(filePath string) (*TrainingDataJSON, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Read all data from the file
	data, err := ioutil.ReadAll(file)
	if err != nil {
		return nil, err
	}

	// Define a struct to match the JSON structure with the top-level "sentences" key
	var jsonData struct {
		Sentences []SentenceData `json:"sentences"`
	}

	// Unmarshal the JSON data into the jsonData struct
	err = json.Unmarshal(data, &jsonData)
	if err != nil {
		return nil, fmt.Errorf("error unmarshaling JSON: %w", err)
	}

	// Create a TrainingDataJSON and copy the data
	trainingData := TrainingDataJSON{
		Sentences: jsonData.Sentences,
	}

	return &trainingData, nil
}

type TrainingDataJSON struct {
	Sentences []SentenceData `json:"sentences"`
}

type SentenceData struct {
	Tokens       []string              `json:"tokens"`
	PosTag       []string              `json:"posTag"`
	NerTag       []string              `json:"nerTag"`
	PhraseTag    []string              `json:"phraseTag"`
	Dependencies []PredictedDependency `json:"dependencies"`
	Sentence     string                `json:"sentence"`
}

// NewMatrix initializes a new matrix (2D slice) of the specified dimensions
// and fills it with small random values.
func NewMatrix(rows, cols int) [][]float64 {
	matrix := make([][]float64, rows)
	// Ensure rand.Seed is called once in your program, e.g., in main.go
	rand.Seed(time.Now().UnixNano()) // Consider moving this to main()

	for i := range matrix {
		matrix[i] = make([]float64, cols)
		for j := range matrix[i] {
			// Initialize with small random values, e.g., between -0.01 and 0.01
			randVal := rand.Float64()*0.02 - 0.01
			matrix[i][j] = randVal
		}
	}
	return matrix
}

// NewRagSimpleNN initializes a new SimpleNN specifically for the RAG task.
// It allocates and initializes the weights and biases of the network with random values.
func NewRagSimpleNN(hiddenSize, outputSize int) *SimpleNN {
	nnn := SimpleNN{}
	nnn.InputSize = 200         // Fixed input size for RAG (query + pooled doc embedding)
	nnn.HiddenSize = hiddenSize // Parameterized hidden layer size
	nnn.OutputSize = outputSize // Parameterized output layer size

	// Initialize weights and biases using the Randomize method
	// This ensures both WeightsIH and WeightsHO are initialized with random values
	nnn.Randomize()

	// Note: The fields related to training data (Sentences, Targets, etc.)
	// are likely not needed in the SimpleNN struct itself for the RAG forward pass.
	// They are used during training data loading and preparation *before* the forward pass.
	// Consider if these fields truly belong in the SimpleNN struct or if they should be
	// part of a separate training data structure used during training.

	return &nnn
}

type HiddenWeightIterator struct {
	nn *SimpleNN
	i  int // current row (hidden neuron index)
	j  int // current column (input neuron index)
}

// NewHiddenWeightIterator creates a new iterator for the hidden layer weights.
func (nn *SimpleNN) NewHiddenWeightIterator() *HiddenWeightIterator {
	return &HiddenWeightIterator{
		nn: nn,
		i:  0,
		j:  0,
	}
}

// Next moves the iterator to the next weight.
// It returns true if there are more weights, false otherwise.
func (it *HiddenWeightIterator) Next() bool {
	it.j++ // Move to the next column (input neuron)

	// If we reached the end of the current row, move to the next row
	if it.j >= it.nn.InputSize {
		it.j = 0 // Reset column to the beginning
		it.i++   // Move to the next row (hidden neuron)
	}

	// Continue as long as the current row index is within bounds
	return it.i < it.nn.HiddenSize
}

// Current returns the current indices (i, j) and the weight value.
func (it *HiddenWeightIterator) Current() (int, int, float64) {
	return it.i, it.j, it.nn.WeightsIH[it.i][it.j]
}

// Update sets the value of the current weight.
func (it *HiddenWeightIterator) Update(value float64) {
	it.nn.WeightsIH[it.i][it.j] = value
}

// OutputWeightIterator iterates over the weights from the hidden to the output layer.
type OutputWeightIterator struct {
	nn *SimpleNN
	i  int // current row (output neuron index)
	j  int // current column (hidden neuron index)
}

// NewOutputWeightIterator creates a new iterator for the output layer weights.
func (nn *SimpleNN) NewOutputWeightIterator() *OutputWeightIterator {
	return &OutputWeightIterator{
		nn: nn,
		i:  0,
		j:  0,
	}
}

// Next moves the iterator to the next weight.
// It returns true if there are more weights, false otherwise.
func (it *OutputWeightIterator) Next() bool {
	if it.i < it.nn.OutputSize && it.j < it.nn.HiddenSize {
		it.j++
		if it.j >= it.nn.HiddenSize {
			it.j = 0
			it.i++
		}
	}
	return false
}

// Current returns the current indices (i, j) and the weight value.
func (it *OutputWeightIterator) Current() (int, int, float64) {
	return it.i, it.j, it.nn.WeightsHO[it.i][it.j]
}

// Update sets the value of the current weight.
func (it *OutputWeightIterator) Update(value float64) {
	it.nn.WeightsHO[it.i][it.j] = value
}

// HiddenBiasIterator iterates over the hidden layer biases.
type HiddenBiasIterator struct {
	nn *SimpleNN
	i  int // current index (hidden neuron index)
}

// NewHiddenBiasIterator creates a new iterator for the hidden layer biases.
func (nn *SimpleNN) NewHiddenBiasIterator() *HiddenBiasIterator {
	return &HiddenBiasIterator{
		nn: nn,
		i:  0,
	}
}

// Next moves the iterator to the next bias.
// It returns true if there are more biases, false otherwise.
func (it *HiddenBiasIterator) Next() bool {
	it.i++
	// The check if there are more biases should be after the increment
	// to correctly handle the last element.
	return it.i < it.nn.HiddenSize
}

// Current returns the current index (i) and the bias value.
func (it *HiddenBiasIterator) Current() (int, float64) {
	return it.i, it.nn.HiddenBiases[it.i]
}

// Update sets the value of the current bias.
func (it *HiddenBiasIterator) Update(value float64) {
	it.nn.HiddenBiases[it.i] = value
}

// OutputBiasIterator iterates over the output layer biases.
type OutputBiasIterator struct {
	nn *SimpleNN
	i  int // current index (output neuron index)
}

// NewOutputBiasIterator creates a new iterator for the output layer biases.
func (nn *SimpleNN) NewOutputBiasIterator() *OutputBiasIterator {
	return &OutputBiasIterator{
		nn: nn,
		i:  0, // Start at index 0
	}
}

// Next moves the iterator to the next bias.
// It returns true if there are more biases, false otherwise.
func (it *OutputBiasIterator) Next() bool {
	// Check if the current index is less than the number of output biases
	if it.i < it.nn.OutputSize {
		// Increment the index for the next iteration *after* checking
		it.i++
		return true // There is a next bias at the new index
	}
	return false // No more biases
}

// Current returns the current index (i) and the bias value.
func (it *OutputBiasIterator) Current() (int, float64) {
	// The index `it.i` was incremented in the last call to `Next()`.
	// The current element is at the *previous* index.
	currentIndex := it.i - 1
	return currentIndex, it.nn.OutputBiases[currentIndex]
}

// Update sets the value of the current bias.
func (it *OutputBiasIterator) Update(value float64) {
	// The index `it.i` was incremented in the last call to `Next()`.
	// The current element is at the *previous* index.
	currentIndex := it.i - 1
	it.nn.OutputBiases[currentIndex] = value
}
