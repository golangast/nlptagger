package word2vec

import (
	"bufio" // Add this import
	"encoding/gob"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"sort"
	"strings"

	// Package word2vec implements a basic Word2Vec model for creating word embeddings.

	"github.com/zendrulat/nlptagger/neural/nn/g"
)

const (
	PaddingToken = "[PAD]"
	UNKToken     = "[UNK]"
)

// SimpleWord2Vec is a basic Word2Vec implementation in Go.
// does NOT include negative sampling or other important optimizations.
// WordVectors represents the word embeddings
type WordVectors map[int][]float64

// Vector represents a word vector
type Vector []float64

type SimpleWord2Vec struct {
	// Core word representation
	Vocabulary   map[string]int
	WordVectors  map[int][]float64
	VectorSize   int
	NgramVectors map[string][]float64
	VocabSize    int

	// Context and semantic information
	ContextEmbeddings map[string][]float64
	ContextLabels     map[string]string
	SentenceTags      map[string][]string

	// Hyperparameters and network configuration
	LearningRate        float64
	MaxGrad             float64
	HiddenSize          int
	SimilarityThreshold float64
	Window              int
	Epochs              int
	NegativeSamples     int
	UseCBOW             bool
	NgramSize           int

	// Weights and biases (neural network parameters)
	Weights [][]float64
	Biases  [][]float64
	Ann     *g.ANN

	//MinWord_frequency
	MinWordFrequency int

	// IndexToWord maps an index back to its corresponding word.
	IndexToWord map[int]string

	// Consider if any new fields should be grouped here based on how they are used
}

// convertToMap converts WordVectors to a map[string][]float64.
func ConvertToMap(wv WordVectors, vocab map[string]int) map[string][]float64 {
	result := make(map[string][]float64)
	for word, index := range vocab {
		result[word] = wv[index]
	}
	return result
}

// ReadTrainingData reads the training data from a plain text file, one sentence per line.
func ReadTrainingData(filePath string) ([]string, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("error opening training data file: %w", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	var sentences []string
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if len(line) > 0 {
			sentences = append(sentences, line)
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading training data file: %w", err)
	}

	return sentences, nil
}

func (sw2v *SimpleWord2Vec) TrainSentenceContext(sentences []string) map[string][]float64 {
	for _, sentence := range sentences {
		words := strings.Fields(sentence)
		context := make([]float64, sw2v.VectorSize)

		for _, word := range words {
			wordIndex, ok := sw2v.Vocabulary[word]
			if !ok {
				// Handle unknown words, e.g., using UNK token
				wordIndex, ok = sw2v.Vocabulary[UNKToken]
				if !ok {
					fmt.Printf("Warning: Unknown word '%s' and no UNK token found\n", word)
					continue
				}
			}
			for i := 0; i < sw2v.VectorSize; i++ {
				context[i] += sw2v.WordVectors[wordIndex][i]
			}
		}
		sw2v.ContextEmbeddings[sentence] = context
	}

	return sw2v.ContextEmbeddings
}

// InitializeWeights initializes the weights and biases of the RNN.
func (sw2v *SimpleWord2Vec) InitializeWeights() {
	sw2v.Weights = make([][]float64, sw2v.HiddenSize)
	sw2v.Biases = make([][]float64, sw2v.HiddenSize)
	for i := 0; i < sw2v.HiddenSize; i++ {
		sw2v.Weights[i] = make([]float64, sw2v.VectorSize)
		sw2v.Biases[i] = make([]float64, 1) // Initialize biases here
		// Initialize weights with random values or other initialization strategy.
		for j := 0; j < sw2v.VectorSize; j++ {
			sw2v.Weights[i][j] = rand.Float64() //Example random initialization
		}
		// Initialize biases with random values or other initialization strategy
		for j := 0; j < 1; j++ {
			sw2v.Biases[i][j] = rand.Float64()
		}
	}
}

// forwardPass performs the forward pass of the RNN.
func (sw2v *SimpleWord2Vec) ForwardPass(words []string) []float64 {
	hiddenState := make([]float64, sw2v.VectorSize) // Initialize hidden state

	for _, word := range words {
		wordIndex, ok := sw2v.Vocabulary[word]
		if !ok {
			wordIndex, ok = sw2v.Vocabulary[UNKToken]
			if !ok {
				fmt.Printf("Warning: Unknown word '%s' and no UNK token found\n", word)
				continue
			}
		}

		inputVector := sw2v.WordVectors[wordIndex]
		if inputVector == nil {
			fmt.Printf("Error: Word index %d not found in WordVectors for word '%s'\n", wordIndex, word)
			continue
		}
		// Check if Weights and Biases are initialized correctly
		if sw2v.Weights == nil || len(sw2v.Weights) != sw2v.HiddenSize {
			fmt.Println("Error: sw2v.Weights is not initialized correctly")
			return nil
		}
		for i := 0; i < sw2v.HiddenSize; i++ {
			if len(sw2v.Weights[i]) != len(inputVector) {
				fmt.Printf("Error: Mismatch in Weights dimension at index: %v\n", i)
				return nil
			}
		}

		if sw2v.Biases == nil || len(sw2v.Biases) != sw2v.HiddenSize {
			fmt.Println("Error: sw2v.Biases is not initialized correctly")
			return nil
		}
		for i := 0; i < sw2v.HiddenSize; i++ {
			if len(sw2v.Biases[i]) != 1 {
				fmt.Printf("Error: Bias dimension mismatch at index: %v\n", i)
				return nil
			}
		}

		if sw2v.HiddenSize <= 0 || sw2v.VectorSize <= 0 {
			fmt.Println("Error: HiddenSize and VectorSize must be greater than zero.")
			return nil
		}

		newHiddenState := make([]float64, sw2v.VectorSize)
		for i := 0; i < sw2v.VectorSize; i++ {
			weightedSum := 0.0
			for j := 0; j < len(inputVector); j++ {
				if i%sw2v.HiddenSize == j%sw2v.HiddenSize {
					weightedSum += sw2v.Weights[i%sw2v.HiddenSize][j] * inputVector[j]
				}
			}
			weightedSum += sw2v.Biases[i%sw2v.HiddenSize][0]
			weightedSum += hiddenState[i]

			newHiddenState[i] = math.Tanh(weightedSum)
		}
		hiddenState = newHiddenState
	}

	output := make([]float64, sw2v.VectorSize)
	copy(output, hiddenState)
	return output
}

// calculateLoss calculates the Mean Squared Error (MSE) loss.
func (sw2v *SimpleWord2Vec) CalculateLoss(output, context []float64) float64 {
	loss := 0.0
	for i := 0; i < sw2v.VectorSize; i++ {
		diff := output[i] - context[i]
		loss += diff * diff
	}
	loss /= float64(sw2v.VectorSize)
	return loss
}

// backpropagate performs backpropagation and updates weights and biases.
func (sw2v *SimpleWord2Vec) Backpropagate(output, context []float64, learningRate float64) {
	// Calculate gradients for weights and biases (replace with your gradient calculation)
	weightGradients := make([][]float64, sw2v.HiddenSize)
	biasGradients := make([][]float64, sw2v.HiddenSize)
	for i := 0; i < sw2v.HiddenSize; i++ {
		weightGradients[i] = make([]float64, sw2v.VectorSize)
		biasGradients[i] = make([]float64, 1)
		for j := 0; j < sw2v.VectorSize; j++ {
			weightGradients[i][j] = (output[i] - context[i]) // Example gradient
		}
		biasGradients[i][0] = (output[i] - context[i]) // Example gradient
	}

	// Clip gradients to prevent exploding gradients
	maxGrad := 1.0 // Maximum gradient norm
	for i := 0; i < sw2v.HiddenSize; i++ {
		weightNorm := math.Sqrt(sumOfSquares(weightGradients[i]))
		if weightNorm > maxGrad {
			scale := maxGrad / weightNorm
			for j := 0; j < sw2v.VectorSize; j++ {
				weightGradients[i][j] *= scale
			}
		}
		sw2v.Weights[i][0] -= learningRate * weightGradients[i][0]
	}
	for i := 0; i < sw2v.HiddenSize; i++ {
		sw2v.Biases[i][0] -= learningRate * biasGradients[i][0]
	}
}

// contains checks if a string slice contains a specific string.
func contains(slice []string, str string) bool {
	for _, s := range slice {
		if s == str {
			return true
		}
	}
	return false
}

// Helper function to calculate the sum of squares of a slice
func sumOfSquares(slice []float64) float64 {
	sum := 0.0
	for _, val := range slice {
		sum += val * val
	}
	return sum
}

// getWordByIndex retrieves the word associated with the given index.
func (sw2v *SimpleWord2Vec) getWordByIndex(index int) string {
	return sw2v.IndexToWord[index]
}

// SaveModel saves the trained model to a GOB file.
func (sw2v *SimpleWord2Vec) SaveModel(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("error creating model file: %w", err)
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	err = encoder.Encode(sw2v)
	if err != nil {
		return fmt.Errorf("error encoding model: %w", err)
	}

	return nil
}

// LoadModel loads a trained model from a GOB file.
func LoadModel(filename string) (*SimpleWord2Vec, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("error opening model file: %w", err)
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	var sw2v SimpleWord2Vec
	err = decoder.Decode(&sw2v)
	if err != nil {
		return nil, fmt.Errorf("error decoding model: %w", err)
	}

	return &sw2v, nil
}

// Train implements a simplified Skip-gram model for training word embeddings.
func (sw2v *SimpleWord2Vec) Train(trainingData []string) {
	// Initialize default hyperparameters if not set
	if sw2v.Epochs == 0 {
		sw2v.Epochs = 10 // Reduced for demonstration
	}
	if sw2v.LearningRate == 0.0 {
		sw2v.LearningRate = 0.01
	}
	if sw2v.Window == 0 {
		sw2v.Window = 2
	}
	if sw2v.NegativeSamples == 0 {
		sw2v.NegativeSamples = 5
	}
	if sw2v.MinWordFrequency == 0 {
	}

	// Build vocabulary and initialize word vectors if they are nil
	if sw2v.Vocabulary == nil {
		sw2v.Vocabulary = make(map[string]int)
		wordCounts := make(map[string]int)
		for _, sentence := range trainingData {
			words := strings.Fields(sentence)
			for _, word := range words {
				wordCounts[word]++
			}
		}

		// Filter out words below the minimum frequency threshold
		filteredWords := make([]string, 0)
		for word, count := range wordCounts {
			if count >= sw2v.MinWordFrequency {
				filteredWords = append(filteredWords, word)
			}
		}

		// Sort the words for deterministic vocabulary assignment
		sort.Strings(filteredWords)

		// Add the filtered words to the vocabulary
		sw2v.Vocabulary[UNKToken] = 0 // Reserve index 0 for UNK token
		sw2v.IndexToWord = make(map[int]string)
		sw2v.IndexToWord[0] = UNKToken
		wordCount := 1
		for _, word := range filteredWords {
			sw2v.Vocabulary[word] = wordCount
			sw2v.IndexToWord[wordCount] = word
			wordCount++
		}
		sw2v.VocabSize = wordCount
	}

	if sw2v.WordVectors == nil {
		sw2v.WordVectors = make(WordVectors)
		for _, index := range sw2v.Vocabulary {
			sw2v.WordVectors[index] = make([]float64, sw2v.VectorSize)
			for i := 0; i < sw2v.VectorSize; i++ {
				sw2v.WordVectors[index][i] = rand.Float64()
			}
			// Update the vocabulary size after initialization
			if index >= sw2v.VocabSize {
				sw2v.VocabSize = index + 1
			}
		}
	}

	if sw2v.NgramVectors == nil {
		sw2v.NgramVectors = make(map[string][]float64)
	}
	if sw2v.NgramSize == 0 {
		sw2v.NgramSize = 3
	}
	sw2v.InitializeWeights()
	// Create a list of word indices for negative sampling
	wordIndices := make([]int, sw2v.VocabSize)
	for i := range wordIndices {
		wordIndices[i] = i
	}

	//var totalLoss float64
	//var iterationCount int

	// Main training loop
	tokenizedTrainingData := make([][]string, len(trainingData))
	for i, sentence := range trainingData {
		tokenizedTrainingData[i] = strings.Fields(sentence)
	}

	for i := 0; i < sw2v.Epochs; i++ {
		//totalLoss = 0
		var learningRate = sw2v.LearningRate - sw2v.LearningRate*0.99*float64(i)/float64(sw2v.Epochs)

		for _, words := range tokenizedTrainingData {
			if sw2v.UseCBOW {
				for j, targetWord := range words {
					_, ok := sw2v.Vocabulary[targetWord]
					if !ok {
						_ = sw2v.Vocabulary[UNKToken]
					}
					contextWords := make([]string, 0)
					for k := -sw2v.Window; k <= sw2v.Window; k++ {
						if k == 0 || j+k < 0 || j+k >= len(words) {
							continue
						}
						contextWords = append(contextWords, words[j+k])
					}
					if len(contextWords) == 0 {
						continue
					}
					contextVector := make([]float64, sw2v.VectorSize)
					for _, contextWord := range contextWords {
						contextIndex, ok := sw2v.Vocabulary[contextWord]
						if !ok {
							contextIndex = sw2v.Vocabulary[UNKToken]
						}
						for index, value := range sw2v.WordVectors[contextIndex] {
							contextVector[index] += value
						}
					}

					output := sw2v.ForwardPass([]string{targetWord})
					sw2v.Backpropagate(output, contextVector, learningRate)

					for n := 0; n < sw2v.NegativeSamples; n++ {
						negIndex := wordIndices[rand.IntN(len(wordIndices))]
						if contains(contextWords, sw2v.getWordByIndex(negIndex)) {
							continue
						}
						sw2v.Backpropagate(output, sw2v.WordVectors[negIndex], learningRate)
					}
				}
			} else {
				for j, targetWord := range words {
					_, ok := sw2v.Vocabulary[targetWord]
					if !ok {
						_ = sw2v.Vocabulary[UNKToken]
					}
					for k := -sw2v.Window; k <= sw2v.Window; k++ {
						if k == 0 || j+k < 0 || j+k >= len(words) {
							continue
						}
						contextWord := words[j+k]
						contextIndex, ok := sw2v.Vocabulary[contextWord]
						if !ok {
							contextIndex = sw2v.Vocabulary[UNKToken]
						}
						output := sw2v.ForwardPass([]string{targetWord})
						sw2v.Backpropagate(output, sw2v.WordVectors[contextIndex], learningRate)

						// Negative samples
						for n := 0; n < sw2v.NegativeSamples; n++ {
							negIndex := wordIndices[rand.IntN(len(wordIndices))]
							if negIndex == contextIndex {
								continue
							}
							if sw2v.WordVectors[negIndex] == nil {
								sw2v.WordVectors[negIndex] = make([]float64, sw2v.VectorSize)
							}
							sw2v.Backpropagate(output, sw2v.WordVectors[negIndex], learningRate)
						}
					}
				}
			}
		}

		//fmt.Printf("Epoch %d, Loss: %f\n", i, totalLoss/float64(iterationCount+1))
	}
}

// TrainWord2VecModel initializes, trains, and saves a SimpleWord2Vec model.
func TrainWord2VecModel(trainingDataPath, modelSavePath string, vectorSize, epochs, window, negativeSamples, minWordFrequency int, useCBOW bool) (*SimpleWord2Vec, error) {
	// Initialize a new SimpleWord2Vec model
	sw2v := &SimpleWord2Vec{
		VectorSize:       vectorSize,
		HiddenSize:       vectorSize, // Initialize HiddenSize with vectorSize
		LearningRate:     0.01,       // Default learning rate
		Window:           window,
		Epochs:           epochs,
		NegativeSamples:  negativeSamples,
		UseCBOW:          useCBOW,
		MinWordFrequency: minWordFrequency,
	}

	// Load training data
	trainingData, err := ReadTrainingData(trainingDataPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read word2vec training data: %w", err)
	}

	// Train the model
	fmt.Printf("Starting Word2Vec model training for %d epochs...\n", epochs)
	sw2v.Train(trainingData)
	fmt.Println("Word2Vec model training finished.")

	// Save the trained model
	if err := sw2v.SaveModel(modelSavePath); err != nil {
		return nil, fmt.Errorf("failed to save word2vec model: %w", err)
	}
	fmt.Printf("Word2Vec model saved to %s\n", modelSavePath)

	return sw2v, nil
}
