package predict

import (
	"encoding/gob"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"os"
	"sort"
	"strings"

	"golang.org/x/exp/rand"

	"github.com/golangast/nlptagger/neural/nn/dr"
	"github.com/golangast/nlptagger/neural/nn/ner"
	"github.com/golangast/nlptagger/neural/nn/phrase"
	"github.com/golangast/nlptagger/neural/nn/pos"
	"github.com/golangast/nlptagger/neural/nnu"
	"github.com/golangast/nlptagger/tagger/tag"
)

// Structure to represent training data in JSON
type TrainingDataJSON struct {
	Sentences []tag.Tag `json:"sentences"`
}

func Predict(nn *nnu.SimpleNN, inputs []float64, posTagVocab map[string]int, nerTagVocab map[string]int, phraseTagVocab map[string]int, drTagVocab map[string]int) (string, string, string, string) {
	predictedPosTag := predictTag(nn, inputs, pos.ForwardPassPos, posTagVocab)
	predictedNerTag := predictTag(nn, inputs, ner.ForwardPassNer, nerTagVocab)
	predictedPhraseTag := predictTag(nn, inputs, phrase.ForwardPassPhrase, phraseTagVocab)
	predictedDRTag := predictTag(nn, inputs, dr.ForwardPassDR, drTagVocab)

	return predictedPosTag, predictedNerTag, predictedPhraseTag, predictedDRTag
}

// predictTag predicts a tag based on the provided forward pass function and vocabulary.
func predictTag(nn *nnu.SimpleNN, inputs []float64, forwardPassFunc func(*nnu.SimpleNN, []float64) []float64, tagVocab map[string]int) string {
	if nn == nil {
		log.Println("Error: Neural network is nil.")
		return "UNK" // Or return an appropriate error value
	}

	predictedOutput := forwardPassFunc(nn, inputs)

	// Create a slice of probability-tag pairs for sorting
	type ProbabilityTagPair struct {
		Probability float64
		Tag         string
	}

	var probabilityTagPairs []ProbabilityTagPair
	for tag, index := range tagVocab {
		if index >= 0 && index < len(predictedOutput) {
			probabilityTagPairs = append(probabilityTagPairs, ProbabilityTagPair{Probability: predictedOutput[index], Tag: tag})
		}
	}
	// Sort the probabilities in descending order
	sort.Slice(probabilityTagPairs, func(i, j int) bool {
		return probabilityTagPairs[i].Probability > probabilityTagPairs[j].Probability
	})

	if len(probabilityTagPairs) > 0 {
		return probabilityTagPairs[0].Tag
	} else {
		log.Println("Error: Empty tag vocabulary.")
		return "UNK"
	}
}

// Backpropagate updates the weights of the neural network using backpropagation.
func Backpropagate(nn *nnu.SimpleNN, totalLoss float64, outputs []float64, learningRate float64, inputs, targets []float64) {

	// 1. Output layer errors.  Calculate the error for each output neuron.
	outputErrors := make([]float64, nn.OutputSize)
	for i := 0; i < nn.OutputSize; i++ {
		// Example: Assuming Mean Squared Error (MSE)
		outputErrors[i] = (outputs[i] - targets[i]) * outputs[i] * (1 - outputs[i]) // Sigmoid derivative
	}

	// 2. Output layer weight update
	for i := 0; i < nn.HiddenSize; i++ { //Hidden layer is the input to Output layer
		for j := 0; j < nn.OutputSize; j++ {
			nn.OutputWeights[i][j] += learningRate * outputErrors[j] * outputs[i]
		}
	}

	// 3. Hidden layer errors. Calculate the error signal for each hidden neuron.
	hiddenErrors := make([]float64, nn.HiddenSize)
	for i := 0; i < nn.HiddenSize; i++ {
		errorSum := 0.0
		for j := 0; j < nn.OutputSize; j++ {
			errorSum += outputErrors[j] * nn.OutputWeights[i][j]
		}
		hiddenErrors[i] = errorSum * outputs[i] * (1 - outputs[i]) // Sigmoid derivative
	}

	// 4. Hidden layer weight updates
	for i := 0; i < nn.HiddenSize; i++ {
		for j := 0; j < nn.InputSize; j++ { // Input layer is the input to Hidden layer
			nn.HiddenWeights[i][j] += learningRate * hiddenErrors[i] * inputs[j]
		}
	}
}
func UpdateWeights(nn *nnu.SimpleNN, gradients float64) {
	index := 0
	for i := 0; i < nn.HiddenSize; i++ { // nn.HiddenSize rows
		for j := 0; j < nn.OutputSize; j++ { // nn.OutputSize columns
			if i < len(nn.HiddenWeights) && j < len(nn.HiddenWeights[i]) {
				nn.HiddenWeights[i][j] += gradients
			}
			index++
		}
	}
	for i := 0; i < nn.HiddenSize; i++ { // nn.HiddenSize rows
		for j := 0; j < nn.OutputSize; j++ { // nn.OutputSize columns
			if i < len(nn.OutputWeights) && j < len(nn.OutputWeights[i]) {
				nn.OutputWeights[i][j] += gradients
			}
			index++
		}
	}
}
func AugmentData(inputs []float64) []float64 {
	maskedInputs := make([]float64, len(inputs))
	copy(maskedInputs, inputs)

	// Number of words to mask (adjust as needed)
	numWordsToMask := int(float64(len(inputs)) * 0.15) // Mask 15% of the words

	// Indices of words to mask
	// Using a map to store masked indices to prevent masking same word twice
	maskedIndices := make(map[int]bool)

	for i := 0; i < numWordsToMask; i++ {
		// Generate a random index
		randomIndex := rand.Intn(len(inputs))

		//check if index has already been masked
		if _, exists := maskedIndices[randomIndex]; exists {
			i-- // Decrement i to repeat the current loop iteration.
			continue
		}

		maskedIndices[randomIndex] = true // Mark index as masked

		// Replace with a special token representation (e.g., [MASK] token)
		// Assuming the special token is at index 0 of your input vocabulary,
		// Adjust the token index if your [MASK] token is at a different index
		maskedInputs[randomIndex] = 0

	}
	return maskedInputs
}
func PredictMaskedWords(nn *nnu.SimpleNN, maskedInputs []float64) []float64 {

	outputs := nnu.ForwardPass(nn, maskedInputs)
	// If outputs is the wrong size, pad it with zeros.
	paddedOutputs := make([]float64, 103)
	copy(paddedOutputs, outputs)
	outputs = paddedOutputs

	return outputs
}

// CalculateOriginalLoss calculates the original loss for your primary task.
// In this example, it calculates the mean squared error loss for a regression task.
func CalculateOriginalLoss(predictedOutput []float64, targetOutput []float64) float64 {
	loss := 0.0
	for i := 0; i < len(predictedOutput); i++ {
		diff := predictedOutput[i] - targetOutput[i]
		loss += diff * diff
	}

	// Return average loss
	return loss / float64(len(predictedOutput))
}

func CalculateMLMLoss(nn *nnu.SimpleNN, maskedInputs []float64, originalInputs []float64, maskedIndices map[int]bool) float64 {
	predictedOutputs := nnu.ForwardPassMLM(nn, maskedInputs)

	totalLoss := 0.0
	numMaskedWords := 0

	for index, isMasked := range maskedIndices {
		if isMasked {
			targetWordIndex := int(originalInputs[index])

			// Clamp targetWordIndex to valid range
			if targetWordIndex < 0 {
				targetWordIndex = 0
			} else if targetWordIndex >= len(predictedOutputs) {
				targetWordIndex = len(predictedOutputs) - 1
			}

			// Safe check to ensure targetWordIndex is within bounds
			if targetWordIndex >= 0 && targetWordIndex < len(predictedOutputs) {
				// Calculate cross-entropy loss
				loss := -math.Log(predictedOutputs[targetWordIndex])
				totalLoss += loss
				numMaskedWords++
			} else {
				log.Printf("Target word index out of range: %d (predictedOutputs length: %d)\n", targetWordIndex, len(predictedOutputs))
			}
		}
	}

	// Average loss over masked words
	if numMaskedWords > 0 {
		return totalLoss / float64(numMaskedWords)
	} else {
		return 0.0 // Handle case where no words were masked
	}
}

func PredictTags(nn *nnu.SimpleNN, sentence string) ([]string, []string, []string, []string) {
	tokenVocab, posTagVocab, nerTagVocab, phraseTagVocab, drTagVocab, _ := CreateVocab()
	// Tokenize the sentence into individual words.
	tokens := strings.Fields(sentence)
	// Create a slice to store the predicted POS tags.
	var predictedPosTags, predictedNerTags, predictedPhraseTags, predictedDRTags []string
	// Iterate over each token in the sentence.
	for _, token := range tokens {
		// Get the index of the token in the vocabulary.
		tokenIndex, ok := tokenVocab[token]
		inputs := make([]float64, nn.InputSize)

		if !ok {
			// Add new token to vocabulary and update the model.gob file
			tokenVocab = AddNewTokenToVocab(token, tokenVocab)
			tokenIndex = tokenVocab[token] // Get the newly assigned index
			saveTokenVocabToGob("model.gob", tokenVocab)
		}

		if tokenIndex >= 0 && tokenIndex < nn.InputSize {
			inputs[tokenIndex] = 1
		} else {
			inputs[0] = 1 // Use index 0 for out-of-range tokenIndex
		}
		// Predict tags using the neural network
		predictedPosTag, predictedNerTag, predictedPhraseTag, predictedDRTag := Predict(nn, inputs, posTagVocab, nerTagVocab, phraseTagVocab, drTagVocab)

		predictedPosTags = append(predictedPosTags, predictedPosTag)
		predictedNerTags = append(predictedNerTags, predictedNerTag)
		predictedPhraseTags = append(predictedPhraseTags, predictedPhraseTag)
		predictedDRTags = append(predictedDRTags, predictedDRTag)
	}
	// Return the list of predicted POS and NER tags.
	return predictedPosTags, predictedNerTags, predictedPhraseTags, predictedDRTags
}

func CreateVocab() (map[string]int, map[string]int, map[string]int, map[string]int, map[string]int, *TrainingDataJSON) {
	trainingData, err := LoadTrainingDataFromJSON("data/training_data.json")
	if err != nil {
		fmt.Println("error loading training data: %w", err)
	}
	// Create vocabularies
	tokenVocab := CreateTokenVocab(trainingData.Sentences)
	posTagVocab := pos.CreatePosTagVocab(trainingData.Sentences)
	nerTagVocab := ner.CreateTagVocabNer(trainingData.Sentences)
	phraseTagVocab := phrase.CreatePhraseTagVocab(trainingData.Sentences)
	drTagVocab := dr.CreateDRTagVocab(trainingData.Sentences)

	return tokenVocab, posTagVocab, nerTagVocab, phraseTagVocab, drTagVocab, trainingData
}

// CreateTokenVocab creates or loads the token vocabulary.
func CreateTokenVocab(trainingData []tag.Tag) map[string]int {
	// Check if the GOB file exists
	if _, err := os.Stat("model.gob"); err == nil {
		// Load vocabulary from GOB file
		tokenVocab, err := loadTokenVocabFromGob("model.gob")
		if err != nil {
			fmt.Println("Error loading vocabulary from GOB:", err)
			return make(map[string]int) // Return empty map on error
		}
		return tokenVocab
	} else {
		// Create and save vocabulary if GOB file doesn't exist
		return createAndSaveTokenVocab(trainingData)
	}
}

// AddNewTokenToVocab adds a new token to the vocabulary and returns the updated vocabulary.
func AddNewTokenToVocab(token string, tokenVocab map[string]int) map[string]int {
	maxIndex := 0
	for _, index := range tokenVocab {
		if index > maxIndex {
			maxIndex = index
		}
	}
	tokenVocab[token] = maxIndex + 1
	return tokenVocab
}

func createAndSaveTokenVocab(trainingData []tag.Tag) map[string]int {
	tokenVocab := make(map[string]int)
	tokenVocab["UNK"] = 0 // Add "UNK" token with index 0
	index := 1

	for _, sentence := range trainingData {
		for _, token := range sentence.Tokens {
			if _, ok := tokenVocab[token]; !ok {
				tokenVocab[token] = index
				index++
			}
		}
	}

	// Save the vocabulary to GOB file
	if err := saveTokenVocabToGob("model.gob", tokenVocab); err != nil {
		fmt.Println("Error saving vocabulary to GOB:", err)
		return make(map[string]int) // Return empty map on error
	}

	return tokenVocab
}

// Function to load training data from a JSON file
func LoadTrainingDataFromJSON(filePath string) (*TrainingDataJSON, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}

	data, err := ioutil.ReadAll(file)
	if err != nil {
		return nil, err
	}

	var trainingData TrainingDataJSON
	err = json.Unmarshal(data, &trainingData)
	if err != nil {
		return nil, err
	}
	file.Close()

	return &trainingData, nil
}

// saveTokenVocabToGob saves the token vocabulary to a GOB file.
func saveTokenVocabToGob(filePath string, tokenVocab map[string]int) error {
	file, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	return encoder.Encode(tokenVocab)
}

// loadTokenVocabFromGob loads the token vocabulary from a GOB file.
func loadTokenVocabFromGob(filePath string) (map[string]int, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	var tokenVocab map[string]int
	err = decoder.Decode(&tokenVocab)
	return tokenVocab, err
}
