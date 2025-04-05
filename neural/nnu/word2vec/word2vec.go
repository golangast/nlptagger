package word2vec

import (
	"encoding/gob"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand/v2"
	"os"
	"strings"

	"github.com/golangast/nlptagger/neural/nn/g"
)

// SimpleWord2Vec is a basic Word2Vec implementation in Go.
// Note: This is a highly simplified example for illustration. It
// does NOT include negative sampling or other important optimizations.
// WordVectors represents the word embeddings
type WordVectors map[int][]float64

// Vector represents a word vector
type Vector []float64

type SimpleWord2Vec struct {
	// Core word representation
	Vocabulary  map[string]int
	WordVectors map[int][]float64
	VectorSize  int

	// Context and semantic information
	ContextEmbeddings map[string][]float64
	ContextLabels     map[string]string
	SentenceTags      map[string][]string
	UNKToken          string

	// Hyperparameters and network configuration
	LearningRate        float64
	MaxGrad             float64
	HiddenSize int
	SimilarityThreshold float64
	Window              int
	Epochs              int

	// Weights and biases (neural network parameters)
	Weights [][]float64
	Biases  [][]float64
	Ann     *g.ANN

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

// ReadTrainingData reads the training data from a JSON file.
func ReadTrainingData(filePath string) ([]string, error) {
	// Open the JSON file
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("error opening training data file: %w", err)
	}
	defer file.Close()

	// Read the file content
	data, err := ioutil.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("error reading training data file: %w", err)
	}

	// Unmarshal the JSON data
	var sentences []struct {
		Sentence string `json:"sentence"`
	}
	err = json.Unmarshal(data, &sentences)
	if err != nil {
		return nil, fmt.Errorf("error unmarshalling training data: %w", err)
	}

	var sentenceTexts []string
	for _, item := range sentences {
		sentenceTexts = append(sentenceTexts, item.Sentence)
	}
	return sentenceTexts, nil
}

func (sw2v *SimpleWord2Vec) TrainSentenceContext(sentences []string) map[string][]float64 {
	for _, sentence := range sentences {
		words := strings.Fields(sentence)
		context := make([]float64, sw2v.VectorSize)

		for _, word := range words {
			wordIndex, ok := sw2v.Vocabulary[word]
			if !ok {
				// Handle unknown words, e.g., using UNK token
				wordIndex, ok = sw2v.Vocabulary[sw2v.UNKToken]
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
			wordIndex, ok = sw2v.Vocabulary[sw2v.UNKToken]
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
		sw2v.InitializeWeights()
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
func (sw2v *SimpleWord2Vec) Backpropagate(output, context []float64) {
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
		sw2v.Weights[i][0] -= sw2v.LearningRate * weightGradients[i][0]
	}
	for i := 0; i < sw2v.HiddenSize; i++ {
		sw2v.Biases[i][0] -= sw2v.LearningRate * biasGradients[i][0]
	}
}

// cosineSimilarity calculates the cosine similarity between two vectors.
func cosineSimilarity(v1, v2 []float64) float64 {
	dotProduct := 0.0
	magV1 := 0.0
	magV2 := 0.0

	for i := 0; i < len(v1); i++ {
		dotProduct += v1[i] * v2[i]
		magV1 += v1[i] * v1[i]
		magV2 += v2[i] * v2[i]
	}

	magV1 = math.Sqrt(magV1)
	magV2 = math.Sqrt(magV2)

	if magV1 == 0 || magV2 == 0 {
		return 0.0
	}
	return dotProduct / (magV1 * magV2)
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
	for word, i := range sw2v.Vocabulary {
		if i == index {
			return word
		}
	}
	return "" // Or return an error if appropriate
}

// Helper function to flatten a 2D slice into a 1D slice
func flatten(slice [][]float64) []float64 {
	var result []float64
	for _, row := range slice {
		result = append(result, row...)
	}
	return result
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

func (sw2v *SimpleWord2Vec) Train(trainingData []string) {
	sw2v.Epochs = 2000
	sw2v.LearningRate = 0.01
	for i := 0; i < sw2v.Epochs; i++ {
		//Here you need to add the Train method
	}

}
