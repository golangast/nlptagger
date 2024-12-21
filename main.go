package main

import (
	"encoding/gob"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"strings"

	modeldata "github.com/golangast/nlptagger/neural"
	"github.com/golangast/nlptagger/neural/nn/g"
	"github.com/golangast/nlptagger/neural/nnu"
	"github.com/golangast/nlptagger/neural/nnu/predict"
)

// WordVectors represents the word embeddings
type WordVectors map[int][]float64

// Vector represents a word vector
type Vector []float64

type SimpleWord2Vec struct {
	Vocabulary        map[string]int
	WordVectors       WordVectors
	VectorSize        int
	Window            int
	contextEmbeddings map[string][]float64
	ContextLabels     map[string]string
	// Add fields to store POS tags, NER tags, etc.
	SentenceTags        map[string][]string
	UNKToken            string // Token for unknown words
	LearningRate        float64
	MaxGrad             float64 // Maximum gradient for clipping
	HiddenSize          int     // Size of the hidden layer in the neural network
	Weights             [][]float64
	Biases              [][]float64
	ann                 *g.ANN
	similarityThreshold float64

	// Add other necessary fields for the neural network
}

// convertToMap converts WordVectors to a map[string][]float64.
func convertToMap(wv WordVectors, vocab map[string]int) map[string][]float64 {
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
		sw2v.contextEmbeddings[sentence] = context
	}

	return sw2v.contextEmbeddings
}

// forwardPass performs the forward pass of the RNN.
func (sw2v *SimpleWord2Vec) forwardPass(words []string) []float64 {
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

		newHiddenState := make([]float64, sw2v.VectorSize)
		for i := 0; i < sw2v.VectorSize; i++ {
			weightedSum := 0.0
			for j := 0; j < len(inputVector); j++ {
				weightedSum += sw2v.Weights[i%sw2v.HiddenSize][j] * inputVector[j]
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

// Function to create a map with sequential integer values
func createSequentialMap(keys []string) map[string]int {
	m := make(map[string]int)
	for i, key := range keys {
		m[key] = i
	}
	return m
}

// getContextVector calculates the context vector for a sentence.
func (sw2v *SimpleWord2Vec) getContextVector(sentence string, md *nnu.SimpleNN) []float64 {
	words := strings.Fields(sentence)
	predictedPosTags, predictedNerTags, predictedPhraseTags, predictedDRTags := predict.PredictTags(md, sentence)
	context := make([]float64, sw2v.VectorSize*len(words)+len(predictedPosTags)+len(predictedNerTags)+len(predictedPhraseTags)+len(predictedDRTags))
	offset := 0

	for i, word := range words {
		wordIndex, wordExists := sw2v.Vocabulary[word]
		if !wordExists {
			_, unkExists := sw2v.Vocabulary[sw2v.UNKToken]
			if !unkExists {
				fmt.Printf("Warning: Word '%s' and UNK token not found in vocabulary, skipping\n", word)
				continue
			}
		}

		if _, ok := sw2v.WordVectors[wordIndex]; !ok {
			fmt.Printf("Warning: Word index '%d' not found in WordVectors, skipping\n", wordIndex)
			continue
		}

		// Use sw2v.ann here to find nearest neighbors!
		neighbors, err := sw2v.ann.NearestNeighbors(sw2v.WordVectors[wordIndex], 5)
		if err != nil {
			fmt.Printf("Error getting nearest neighbors: %v\n", err)
		} else {
			// Augment the target word vector with neighbor vectors
			for _, neighbor := range neighbors {
				if len(neighbor.Vector) != sw2v.VectorSize {
					fmt.Printf("Warning: Neighbor vector length (%d) does not match VectorSize (%d) for word '%s'. Skipping neighbor.\n",
						len(neighbor.Vector), sw2v.VectorSize, neighbor.Word)
					continue // Skip this neighbor
				}
			}
			// Augment the target word vector with neighbor vectors
			augmentedVector := make([]float64, sw2v.VectorSize)
			copy(augmentedVector, sw2v.WordVectors[wordIndex]) // Start with the original vector

			for _, neighbor := range neighbors {
				for j := 0; j < sw2v.VectorSize; j++ {
					augmentedVector[j] += neighbor.Vector[j]
				}
			}

			// Average the vectors
			for j := 0; j < sw2v.VectorSize; j++ {
				augmentedVector[j] /= float64(len(neighbors) + 1)
			}
			copy(context[offset:offset+sw2v.VectorSize], augmentedVector)
		}
		offset += sw2v.VectorSize

		// Directly use predicted tags
		if i < len(predictedPosTags) {
			context[offset] = float64(createSequentialMap(predictedPosTags)[predictedPosTags[i]])
			offset++
		}
		if i < len(predictedNerTags) {
			context[offset] = float64(createSequentialMap(predictedNerTags)[predictedNerTags[i]])
			offset++
		}
		if i < len(predictedPhraseTags) {
			context[offset] = float64(createSequentialMap(predictedPhraseTags)[predictedPhraseTags[i]])
			offset++
		}
		if i < len(predictedDRTags) {
			context[offset] = float64(createSequentialMap(predictedDRTags)[predictedDRTags[i]])
			offset++
		}
	}

	return context
}

// calculateLoss calculates the Mean Squared Error (MSE) loss.
func (sw2v *SimpleWord2Vec) calculateLoss(output, context []float64) float64 {
	loss := 0.0
	for i := 0; i < sw2v.VectorSize; i++ {
		diff := output[i] - context[i]
		loss += diff * diff
	}
	loss /= float64(sw2v.VectorSize)
	return loss
}

// findNearestWord finds the nearest neighbor to the given vector.
func (sw2v *SimpleWord2Vec) findNearestWord(vector []float64, words []string) (string, float64) {
	nearestWord := ""
	maxSimilarity := 0.0

	// Convert the target vector to []int if necessary
	// targetVectorInt := make([]int, len(vector))
	// for i, val := range vector {
	// 	targetVectorInt[i] = int(val) // Adjust conversion as needed
	// }

	neighbors, err := sw2v.ann.NearestNeighbors(vector, 10) // Get 10 nearest neighbors using the ANN index

	if err != nil {
		fmt.Printf("Error getting nearest neighbors: %v\n", err)
	} else {
		for _, neighbor := range neighbors {
			if neighbor.Similarity > maxSimilarity && neighbor.Similarity > sw2v.similarityThreshold && neighbor.Word != sw2v.UNKToken && !contains(words, neighbor.Word) {
				maxSimilarity = neighbor.Similarity
				nearestWord = neighbor.Word
			}
		}
	}

	return nearestWord, maxSimilarity
}

// backpropagate performs backpropagation and updates weights and biases.
func (sw2v *SimpleWord2Vec) backpropagate(output, context []float64) {
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

// enhancedSimilarity calculates similarity considering tag information.
func (sw2v *SimpleWord2Vec) enhancedSimilarity(output, wordVector []float64, predictedTags []string, wordTags []string) float64 {
	cosineSim := cosineSimilarity(output, wordVector)

	tagWeight := 0.5 // Adjust weight as needed
	tagSim := 0.0

	// Calculate tag similarity (example for NER tags)
	for _, predictedTag := range predictedTags {
		for _, wordTag := range wordTags {
			if predictedTag == wordTag {
				tagSim += 1.0
				break
			}
		}
	}

	// Normalize tag similarity
	if len(predictedTags) > 0 {
		tagSim /= float64(len(predictedTags))
	}

	// Combine cosine similarity and tag similarity
	similarity := (1-tagWeight)*cosineSim + tagWeight*tagSim

	return similarity
}

// Train function to train the model using the provided JSON data
func (sw2v *SimpleWord2Vec) Train(sentences []string) error {

	// 1. Build Vocabulary
	for _, sentence := range sentences {
		words := strings.Fields(sentence)
		for _, word := range words {
			if _, ok := sw2v.Vocabulary[word]; !ok {
				sw2v.Vocabulary[word] = len(sw2v.Vocabulary)
				sw2v.WordVectors[len(sw2v.Vocabulary)-1] = make([]float64, sw2v.VectorSize)
				// Initialize the word vector (e.g., with random values)
				for i := 0; i < sw2v.VectorSize; i++ {
					sw2v.WordVectors[len(sw2v.Vocabulary)-1][i] = (rand.Float64() - 0.5) / float64(sw2v.VectorSize)
				}
			}
		}
	}

	// Add the UNK token to the vocabulary if it doesn't exist
	if _, ok := sw2v.Vocabulary[sw2v.UNKToken]; !ok {
		sw2v.Vocabulary[sw2v.UNKToken] = len(sw2v.Vocabulary)
		sw2v.WordVectors[len(sw2v.Vocabulary)-1] = make([]float64, sw2v.VectorSize)
		// Initialize the UNK token vector (e.g., with random values)
		for i := 0; i < sw2v.VectorSize; i++ {
			sw2v.WordVectors[len(sw2v.Vocabulary)-1][i] = (rand.Float64() - 0.5) / float64(sw2v.VectorSize)
		}
	}

	sw2v.ann.AddWordVectors(convertToMap(sw2v.WordVectors, sw2v.Vocabulary)) // Populate the index
	//fmt.Printf("Index length before AddWordVectors: %d\n", sw2v.ann.Index.Len())
	sw2v.similarityThreshold = 0.6

	// 3. Train the network
	iteration := 10
	// Initialize weights and biases if not already initialized
	if sw2v.Weights == nil {
		sw2v.Weights = make([][]float64, sw2v.HiddenSize)
		for i := 0; i < sw2v.HiddenSize; i++ {
			sw2v.Weights[i] = make([]float64, sw2v.VectorSize)
			for j := 0; j < sw2v.VectorSize; j++ {
				sw2v.Weights[i][j] = rand.NormFloat64() * math.Sqrt(2.0/float64(sw2v.VectorSize+sw2v.HiddenSize)) // Xavier/Glorot initialization
			}
		}
	}
	if sw2v.Biases == nil {
		sw2v.Biases = make([][]float64, sw2v.HiddenSize)
		for i := 0; i < sw2v.HiddenSize; i++ {
			sw2v.Biases[i] = make([]float64, 1)
			sw2v.Biases[i][0] = rand.NormFloat64() * math.Sqrt(2.0/float64(sw2v.HiddenSize)) // Xavier/Glorot initialization
		}
	}
	//make the model
	md, err := modeldata.ModelData("datas/tagdata/training_data.json")
	if err != nil {
		fmt.Println("Error loading or training model:", err)
	}

	// 3. Train the network
	sw2v.LearningRate = 0.01 // Initial learning rate

	for _, sentence := range sentences {
		context := sw2v.getContextVector(sentence, md) // Check usage in updated context

		words := strings.Fields(sentence)

		output := sw2v.forwardPass(words)

		loss := sw2v.calculateLoss(output, context)
		sw2v.backpropagate(output, context)
		// sw2v.updateContextEmbeddings(sentence, output) // No need to call this separately

		// Print sentence and its corresponding generated context embedding
		fmt.Printf("Iteration %d: Sentence: %s\n", iteration, sentence)
		fmt.Printf("Iteration %d: Loss: %f\n", iteration, loss)

		// Find the nearest context word, excluding words from the input sentence
		if len(words) > 0 { // Only proceed if the sentence is not empty

			neighbors, err := sw2v.ann.NearestNeighbors(output, 10)
			if err != nil {
				fmt.Printf("Error getting nearest neighbors: %v\n", err)
			} else if len(neighbors) > 0 {
				nearestWord := neighbors[0].Word
				maxSimilarity := neighbors[0].Similarity
				fmt.Printf("Iteration %d: Nearest Context Word: %s (Similarity: %.4f)\n", iteration, nearestWord, maxSimilarity)
			}
		} else {
			fmt.Printf("Iteration %d: Skipping nearest context word calculation for empty sentence.\n", iteration)
		}

		// Learning rate schedule
		if iteration%100 == 0 && sw2v.LearningRate > 0.001 {
			sw2v.LearningRate *= 0.95 // Reduce learning rate gradually
		}
		iteration++
	}

	return nil
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

func main() {
	modelFilename := "trained_model.gob"

	// Try to load the model from file
	sw2v, err := LoadModel(modelFilename)
	if err != nil {
		fmt.Println("Error loading the model in loadmodel:", err)
	}

	// if err == nil {
	fmt.Println("Model loaded from", modelFilename)
	// } else {

	// Read training data from file
	trainingData, err := ReadTrainingData("./datas/training_data.json")
	if err != nil {
		fmt.Println("Error reading training data:", err)
	}

	// Initialize SimpleWord2Vec
	sw2v = &SimpleWord2Vec{
		Vocabulary:        make(map[string]int),
		WordVectors:       make(WordVectors),
		VectorSize:        100, // Example vector size
		contextEmbeddings: make(map[string][]float64),
		Window:            10, // Example context window size
		ContextLabels:     make(map[string]string),
		UNKToken:          "<UNK>",
		HiddenSize:        100, // Example hidden size
		LearningRate:      0.01,
		MaxGrad:           20.0,
	}
	sw2v.ann, err = g.NewANN(sw2v.VectorSize, "euclidean")
	if err != nil {
		fmt.Println("Error creating ANN:", err) // Handle the error properly
		return                                  // Exit if ANN creation fails
	}
	sw2v.similarityThreshold = 0.6

	// Train the model
	err = sw2v.Train(trainingData)
	if err != nil {
		fmt.Println("Error training the model:", err)
	}
	// }
	// Save the trained model
	err = sw2v.SaveModel(modelFilename)
	if err != nil {
		fmt.Println("Error saving the model:", err)
	}

	fmt.Println("Model trained and saved to", modelFilename)
}
