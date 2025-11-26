// Package train implements neural network training functions.
// It handles loading data, preparing inputs, running training epochs,
// and evaluating model accuracy.
package train

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"math/rand/v2"
	"os"
	"sort"
	"strings"

	"github.com/zendrulat/nlptagger/neural/nn/dr"
	"github.com/zendrulat/nlptagger/neural/nn/ner"
	"github.com/zendrulat/nlptagger/neural/nn/phrase"
	"github.com/zendrulat/nlptagger/neural/nn/pos"
	"github.com/zendrulat/nlptagger/neural/nnu"
	"github.com/zendrulat/nlptagger/neural/nnu/calc"
	"github.com/zendrulat/nlptagger/neural/nnu/contextvector"
	"github.com/zendrulat/nlptagger/neural/nnu/gobs"
	"github.com/zendrulat/nlptagger/neural/nnu/predict"
	"github.com/zendrulat/nlptagger/neural/nnu/vocab"
	"github.com/zendrulat/nlptagger/neural/nnu/word2vec"
	"github.com/zendrulat/nlptagger/tagger/tag"
)

type TrainingDataJSON struct {
	Sentences []tag.Tag `json:"sentences"`
}

type ContextRelevance struct {
	Iteration           string
	NearestContextWord  string
	Similarity          float64
	ContextualRelevance float64
}

func prepareInputData(tagData tag.Tag, nn *nnu.SimpleNN) []float64 {
	tokenVocab, _, _, _, _, _ := CreateVocab()
	log.Printf("prepareInputData - Size of tokenVocab: %d", len(tokenVocab))
	nn.TokenVocab = tokenVocab
	tokens := tagData.Tokens
	tokenVocabSize := len(nn.TokenVocab)
	maxSentenceSize := nn.InputSize / tokenVocabSize
	nn.Inputs = make([]float64, nn.InputSize)

	// Set the input with the sentence tokens
	for i, token := range tokens {
		// Check if we've exceeded the maximum sentence size
		if i >= maxSentenceSize {
			break
		}

		tokenIndex, ok := nn.TokenVocab[token]
		if !ok {
			tokenIndex = nn.TokenVocab["UNK"]
		}

		for j := 0; j < tokenVocabSize; j++ {
			if tokenIndex == j {
				nn.Inputs[(i)*tokenVocabSize+j] = 1
			}
		}

	}

	return nn.Inputs
}
func TrainAccuracy(trainingData []tag.Tag, nn *nnu.SimpleNN, sw2v *word2vec.SimpleWord2Vec) (float64, float64, float64, float64) {

	posAccuracy, nerAccuracy, phraseAccuracy, drAccuracy := 0.0, 0.0, 0.0, 0.0
	posTotal, nerTotal, phraseTotal, drTotal := 0.0, 0.0, 0.0, 0.0
	posCorrect, nerCorrect, phraseCorrect, drCorrect := 0.0, 0.0, 0.0, 0.0

	// if nn.WeightsIH == nil || len(nn.WeightsIH) == 0 {
	// 	defer func() {
	// 		if r := recover(); r != nil {
	// 			log.Println("TrainAccuracy - Recovered from panic:", r)
	// 			// You might want to log more details about the panic here
	// 		}
	// 	}()

	// 	fmt.Println("TrainAccuracy - WeightsIH is nil or empty")
	// }

	// Initialize weights and biases if not already initialized
	if nn.HiddenWeights == nil {
		nn.HiddenWeights = make([][]float64, nn.HiddenSize)
		for i := 0; i < nn.HiddenSize; i++ {
			nn.HiddenWeights[i] = make([]float64, nn.InputSize)
			for j := 0; j < nn.InputSize; j++ {
				nn.HiddenWeights[i][j] = rand.NormFloat64() * math.Sqrt(2.0/float64(nn.InputSize+nn.HiddenSize))
			}
		}
	}

	if nn.InputSize == 0 || nn.OutputSize == 0 {
		nn.InputSize = len(nn.TokenVocab) * 3
		nn.OutputSize = len(nn.TokenVocab) * 3
		nn.HiddenSize = len(nn.TokenVocab) * 3
		nn.Targets = make([]float64, nn.OutputSize)
		nn.Inputs = make([]float64, nn.InputSize)
		nn.Outputs = make([]float64, nn.OutputSize)

		nn.HiddenBiases = make([]float64, nn.HiddenSize) // Initialize HiddenBiases
		nn.OutputBiases = make([]float64, nn.OutputSize) // Initialize OutputBiases
		for i := range nn.HiddenBiases {
			nn.HiddenBiases[i] = rand.Float64()*0.02 - 0.01 // Initialize with random values
		}
	}

	for epoch := 0; epoch < sw2v.Epochs; epoch++ {
		fmt.Println(epoch)
		for _, taggedSentence := range trainingData {
			nn.Inputs = prepareInputData(taggedSentence, nn)
			maskedIndices, err := prepareMLMInput(nn, taggedSentence.Tokens, nn.TokenVocab)
			if err != nil {
				fmt.Printf("TrainAccuracy - Error: prepareMLMInput failed: %v", err)
				continue
			}

			predictions, err := predict.PredictMaskedWords(nn)
			if err != nil {
				fmt.Printf("TrainAccuracy - Error: PredictMaskedWords failed: %v", err)
				continue
			}
			mlmLoss := predict.CalculateMLMLoss(nn, predictions, nn.Targets, maskedIndices)
			originalOutputs := pos.ForwardPassPos(nn, nn.Inputs)
			originalLoss := predict.CalculateOriginalLoss(nn, originalOutputs, nn.Targets)
			totalLoss := originalLoss + mlmLoss

			nn.Backpropagate(totalLoss, sw2v.LearningRate)
			// Calculate accuracy for the current sentence
			posAcc, nerAcc, phraseAcc, drAcc := calc.CalculateAccuracy(nn, trainingData, nn.TokenVocab, pos.CreatePosTagVocab(trainingData), ner.CreateTagVocabNer(trainingData), phrase.CreatePhraseTagVocab(trainingData), dr.CreateDRTagVocab(trainingData))
			posAccuracy += posAcc
			nerAccuracy += nerAcc
			phraseAccuracy += phraseAcc
			drAccuracy += drAcc

			posTotal++
			nerTotal++
			phraseTotal++
			drTotal++
			if posAcc > 0 {
				posCorrect++
			}
			if nerAcc > 0 {
				nerCorrect++
			}
			if phraseAcc > 0 {
				phraseCorrect++
			}
			if drAcc > 0 {
				drCorrect++
			}
		}
	}

	if posTotal > 0 {
		posAccuracy = posCorrect / posTotal
	} else {
		posAccuracy = 0.0
	}
	if nerTotal > 0 {
		nerAccuracy = nerCorrect / nerTotal
	} else {
		nerAccuracy = 0.0
	}
	if phraseTotal > 0 {
		phraseAccuracy = phraseCorrect / phraseTotal
	} else {
		phraseAccuracy = 0.0
	}
	if drTotal > 0 {
		drAccuracy = drCorrect / drTotal
	} else {
		drAccuracy = 0.0
	}

	fmt.Printf("TrainAccuracy - Final POS Accuracy: %.2f%%", posAccuracy*100)
	fmt.Printf("TrainAccuracy - Final NER Accuracy: %.2f%%\n", nerAccuracy*100)
	fmt.Printf("TrainAccuracy - Final Phrase Accuracy: %.2f%%", phraseAccuracy*100)
	fmt.Printf("TrainAccuracy - Final DR Accuracy: %.2f%%", drAccuracy*100)

	return posAccuracy, nerAccuracy, phraseAccuracy, drAccuracy
}

func TrainModel(trainingData []tag.Tag, nn *nnu.SimpleNN, sw2v *word2vec.SimpleWord2Vec) (*nnu.SimpleNN, error) {

	if nn.WeightsIH == nil || len(nn.WeightsIH) == 0 {
		log.Println("TrainModel - WeightsIH is nil or empty")
	}
	//Ensure vocabulary is created
	tokenVocab, _, _, _, _, _ := CreateVocab()
	nn.TokenVocab = tokenVocab

	nn.MaskedInputs = []float64{}
	nn.Inputs = []float64{}
	nn.Targets = []float64{}

	if nn.WeightsIH == nil || len(nn.WeightsIH) == 0 {
		nn.HiddenBiases = make([]float64, nn.HiddenSize)
		nn.OutputBiases = make([]float64, nn.OutputSize)
	}

	for epoch := 0; epoch < sw2v.Epochs; epoch++ {
		log.Printf("TrainModel - Epoch: %d", epoch)

		for _, taggedSentence := range trainingData {

			// Prepare inputs, targets and maskedindices
			maskedIndices, err := prepareMLMInput(nn, taggedSentence.Tokens, nn.TokenVocab)
			if err != nil {
				fmt.Printf("TrainModel - len(nn.Inputs): %d, len(nn.Targets): %d, len(nn.MaskedIndices): %v", len(nn.Inputs), len(nn.Targets), len(nn.MaskedIndices))
				fmt.Printf("TrainModel - nn.Inputs: %v", nn.Inputs)
				fmt.Printf("TrainModel - nn.Targets: %v", nn.Targets)
				fmt.Printf("TrainModel - nn.MaskedIndices: %v", nn.MaskedIndices)
				fmt.Printf("TrainModel - Error in prepareMLMInput: %v", err)
				fmt.Printf("TrainModel - Error in prepareMLMInput: %v", err)
				continue
			}

			if len(nn.Inputs) == 0 {
				prepareInputData(taggedSentence, nn)
			}

			nn.MaskedInputs = make([]float64, len(nn.Inputs)) // Allocate based on nn.Inputs length
			copy(nn.MaskedInputs, nn.Inputs)
			augmentedInputs := predict.AugmentData(nn.MaskedInputs)

			// Original task loss calculation using augmented input
			if len(nn.Inputs) != nn.InputSize {
				log.Println("TrainModel - Error: len(nn.Inputs) != nn.InputSize")
			}
			originalOutputs := pos.ForwardPassPos(nn, augmentedInputs)
			originalLoss := predict.CalculateOriginalLoss(nn, originalOutputs, nn.Targets) //Use augmentedInputs instead of nn.Inputs

			// MLM Forward pass and loss calculation using augmented input
			predictions, err := predict.PredictMaskedWords(nn)
			if err != nil {
				log.Printf("TrainModel - Error in PredictMaskedWords: %v", err)
				continue
			}
			mlmLoss := predict.CalculateMLMLoss(nn, predictions, nn.Targets, maskedIndices)

			// Combine losses
			totalLoss := originalLoss + mlmLoss

			nn.Backpropagate(totalLoss, sw2v.LearningRate)
		}
	}

	if nn.WeightsHO == nil || len(nn.WeightsHO) == 0 {
		fmt.Println("TrainModel - WeightsHO is nil or empty")
	}

	return nn, nil
}

// Train function to train the model using the provided JSON data
func JsonModelTrain(sw2v *word2vec.SimpleWord2Vec, md *nnu.SimpleNN) (ContextRelevance, error) {
	// make the model
	var c ContextRelevance
	sw2v.Ann.AddWordVectors(word2vec.ConvertToMap(sw2v.WordVectors, md.TokenVocab)) // Populate the index BEFORE the loop
	// 1. Build Vocabulary
	for _, sentence := range md.PSentences {
		for _, word := range strings.Fields(sentence) {
			if _, ok := sw2v.Vocabulary[word]; !ok {
				sw2v.Vocabulary[word] = len(sw2v.Vocabulary)
				sw2v.WordVectors[len(sw2v.Vocabulary)-1] = make([]float64, sw2v.VectorSize)
				// Initialize the word vector (e.g., with random values)
				for i := 0; i < sw2v.VectorSize; i++ {
					sw2v.WordVectors[len(sw2v.Vocabulary)-1][i] = (rand.Float64() - 0.5) / float64(sw2v.VectorSize)
				}

			}
		}
		initialConvertedMap := word2vec.ConvertToMap(sw2v.WordVectors, sw2v.Vocabulary)
		sw2v.Ann.AddWordVectors(initialConvertedMap)

		context := contextvector.GetContextVector(sentence, md, sw2v) // Check usage in updated context
		md.Outputs = sw2v.ForwardPass(strings.Fields(sentence))
		loss := sw2v.CalculateLoss(md.Outputs, context)
		sw2v.Backpropagate(md.Outputs, context, sw2v.LearningRate)
		// Print sentence and its corresponding generated context embedding
		fmt.Printf("Iteration %d: Sentence: %s: Loss: %f\n", sw2v.Epochs, sentence, loss)

		// Find the nearest context word, excluding words from the input sentence

		neighbors, err := sw2v.Ann.NearestNeighbors(sentence, md.Outputs, 10)
		if err != nil {
			fmt.Printf("Error getting nearest neighbors: %v\n", err)
		} else if len(neighbors) > 0 {
			nearestWord := neighbors[0].Word
			maxSimilarity := neighbors[0].Similarity
			ContextualRelevance := neighbors[0].ContextualRelevance
			fmt.Printf("Iteration %d: Nearest Context Word: %s (Similarity: %.4f ContextualRelevance: %.4f)\n", sw2v.Epochs, nearestWord, maxSimilarity, ContextualRelevance)

			c = ContextRelevance{
				Iteration:           fmt.Sprintf("Iteration %d", sw2v.Epochs),
				NearestContextWord:  nearestWord,
				Similarity:          maxSimilarity,
				ContextualRelevance: ContextualRelevance,
			}

		}

		// Learning rate schedule
		if sw2v.Epochs%100 == 0 && sw2v.LearningRate > 0.001 {
			sw2v.LearningRate *= 0.95 // Reduce learning rate gradually
		}
		sw2v.Epochs++
	}

	// Add the UNK token to the vocabulary if it doesn't exist
	if _, ok := sw2v.Vocabulary[word2vec.UNKToken]; !ok {
		sw2v.Vocabulary[word2vec.UNKToken] = len(sw2v.Vocabulary)
		sw2v.WordVectors[len(sw2v.Vocabulary)-1] = make([]float64, sw2v.VectorSize)
		// Initialize the UNK token vector (e.g., with random values)
		for i := 0; i < sw2v.VectorSize; i++ {
			sw2v.WordVectors[len(sw2v.Vocabulary)-1][i] = (rand.Float64() - 0.5) / float64(sw2v.VectorSize)
		}
	}

	sw2v.Ann.AddWordVectors(word2vec.ConvertToMap(sw2v.WordVectors, sw2v.Vocabulary)) // Populate the index
	sw2v.SimilarityThreshold = 0.6

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

	return c, nil
}

// TrainAndSaveModel trains a new neural network model, or loads an existing one if available.
// It then saves the trained model to a file.
func TrainAndSaveModel(modeldirectory string, sw2v *word2vec.SimpleWord2Vec) (*nnu.SimpleNN, error) {
	_, _, _, _, _, trainingData := vocab.CreateVocab(modeldirectory)

	// Delete existing model file if it exists.
	if _, err := os.Stat("./gob_models/trained_model.gob"); err == nil {
		// If the file exists, remove it.
		if err := os.Remove("./gob_models/trained_model.gob"); err != nil {
			// If there's an error during removal, return an error.
			return nil, fmt.Errorf("error deleting model file: %w", err)
		}
	}

	// Load or train the neural network model.
	nn, err := LoadModelOrTrainNew(trainingData, modeldirectory, sw2v)
	if err != nil {
		return nil, fmt.Errorf("error loading or training model: %w", err)
	}

	nnn, err := TrainModel(trainingData.Sentences, nn, sw2v)
	// Save the trained model to a file.
	if err := gobs.SaveModelToGOB(nnn, "./gob_models/trained_model.gob"); err != nil {
		return nil, fmt.Errorf("error saving model: %w", err)
	}

	// Return the trained neural network model and nil error.
	return nn, nil
}

func LoadModelOrTrainNew(trainingData *vocab.TrainingDataJSON, modeldirectory string, sw2v *word2vec.SimpleWord2Vec) (*nnu.SimpleNN, error) {
	tokenVocab, _, _, _, _, _ := CreateVocab()
	defer func() {
		if r := recover(); r != nil {
			log.Println("LoadModelOrTrainNew - Recovered from panic:", r)
			// You might want to log more details about the panic here
		}
	}()

	nn, err := gobs.LoadModelFromGOB("./gob_models/trained_model.gob")
	if err != nil {
		return nn, err
	}

	nn.TokenVocab = tokenVocab
	nn.InputSize = len(tokenVocab) * 3
	nn.OutputSize = len(tokenVocab) * 3
	nn.HiddenSize = len(tokenVocab) * 3
	nn.HiddenWeights = make([][]float64, nn.HiddenSize)
	for i := range nn.HiddenWeights {
		nn.HiddenWeights[i] = make([]float64, nn.InputSize)
		for j := range nn.HiddenWeights[i] {
			nn.HiddenWeights[i][j] = rand.NormFloat64() * math.Sqrt(2.0/float64(nn.InputSize+nn.HiddenSize))
		}
	}
	nn.WeightsHO = make([][]float64, nn.OutputSize)
	for i := range nn.WeightsHO {
		nn.WeightsHO[i] = make([]float64, nn.HiddenSize)
		for j := range nn.WeightsHO[i] {
			nn.WeightsHO[i][j] = rand.NormFloat64() * math.Sqrt(2.0/float64(nn.HiddenSize+nn.OutputSize))
		}
		nn.HiddenBiases = make([]float64, nn.HiddenSize)
		nn.OutputBiases = make([]float64, nn.OutputSize)
	}

	// Load training data
	posaccuracy, neraccuracy, phraseaccuracy, draccuracy := TrainAccuracy(trainingData.Sentences, nn, sw2v)
	fmt.Printf("Final POS Accuracy: %.2f%%\n", posaccuracy*100)
	fmt.Printf("Final NER Accuracy: %.2f%%\n", neraccuracy*100)
	fmt.Printf("Final Phrase Accuracy: %.2f%%\n", phraseaccuracy*100)
	fmt.Printf("Final Dependency relation Accuracy: %.2f%%\n", draccuracy*100)
	// Save the newly trained model
	err = gobs.SaveModelToGOB(nn, "./gob_models/trained_model.gob")
	if err != nil {
		return nil, fmt.Errorf("error saving model: %w", err)
	}
	return nn, nil
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

func CreateVocab() (map[string]int, map[string]int, map[string]int, map[string]int, map[string]int, *TrainingDataJSON) {
	trainingData, err := LoadTrainingDataFromJSON("trainingdata/tagdata/nlp_training_data.json")
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

func CreateTokenVocab(trainingData []tag.Tag) map[string]int {
	tokenVocab := make(map[string]int)
	wordFrequencies := make(map[string]int)
	for _, sentence := range trainingData {
		for _, token := range sentence.Tokens {
			wordFrequencies[token]++ // Increment the frequency count for each token
		}
	}
	minFreq := 5       // Example: Exclude words occurring fewer than 5 times
	vocabSize := 50000 // Example: Limit vocab to 50,000 words
	// Sort words by frequency
	sortedWords := make([]string, 0, len(wordFrequencies))
	for word := range wordFrequencies {
		sortedWords = append(sortedWords, word)
	}
	sort.Slice(sortedWords, func(i, j int) bool {
		return wordFrequencies[sortedWords[i]] > wordFrequencies[sortedWords[j]]
	})

	// Add words to vocabulary up to vocabSize limit
	tokenVocab["UNK"] = 0
	index := 1
	for _, word := range sortedWords {
		if wordFrequencies[word] >= minFreq && index <= vocabSize {
			tokenVocab[word] = index
			index++
		}
	}
	for _, sentence := range trainingData { // Iterate through tag.Tag slice
		for _, token := range sentence.Tokens {
			if _, ok := tokenVocab[token]; !ok {
				tokenVocab[token] = index
				index++
			}
		}
	}

	// If index exceeded vocabulary size
	if index > len(tokenVocab)-1 { // Dynamically determine vocabulary size
		// Handle unknown tokens
		tokenVocab["UNK"] = len(tokenVocab) // Add "UNK" token
		index = len(tokenVocab)             // Update index to reflect new vocabulary size

	}

	return tokenVocab
}

func prepareMLMInput(nn *nnu.SimpleNN, sentence []string, tokenVocab map[string]int) (map[int]bool, error) {

	nn.Inputs = make([]float64, nn.OutputSize)
	nn.Targets = make([]float64, nn.OutputSize)
	nn.MaskedInputs = make([]float64, nn.OutputSize)
	sentenceIndices := make([]float64, nn.OutputSize)
	// Pre-calculate the number of tokens to mask once.
	numTokensToMask := int(0.15 * float64(len(sentence)))

	// fmt.Printf("prepareMLMInput - Size of sentence: %d", len(sentence))
	// fmt.Printf("prepareMLMInput - nn.OutputSize: %d", nn.OutputSize)
	// fmt.Printf("prepareMLMInput - len(nn.Targets): %d", len(nn.Targets))

	maskedIndices := make(map[int]bool)
	maskedIndicesSentence := make(map[int]bool)

	// Populate sentenceIndices
	for i := 0; i < len(sentence); i++ {
		word := sentence[i]
		if index, ok := tokenVocab[word]; ok && index != 0 {
			sentenceIndices[i] = float64(index)
		} else {
			sentenceIndices[i] = float64(tokenVocab["UNK"])
		}
		//fmt.Printf("prepareMLMInput - sentenceIndices[%d]: %f", i, sentenceIndices[i])
	}

	for i := 0; i < nn.OutputSize; i++ {
		if i < len(sentence) {
			nn.MaskedInputs[i] = sentenceIndices[i] // Copy to MaskedInputs
		} else {
			nn.MaskedInputs[i] = 0 // Fill with 0s beyond sentence length
		}
	}
	// More efficient masking using a loop outside the word loop
	for j := 0; j < numTokensToMask; j++ {
		randomIndex := rand.IntN(len(sentence)) //Use rand.Intn()
		if !maskedIndicesSentence[randomIndex] {
			maskedIndicesSentence[randomIndex] = true
			maskedIndices[randomIndex] = true
			nn.Inputs[randomIndex] = 0                             // Mask input
			nn.Targets[randomIndex] = sentenceIndices[randomIndex] // Set target
			// fmt.Printf("prepareMLMInput - nn.Inputs[%d]: %f", randomIndex, nn.Inputs[randomIndex])
			// fmt.Printf("prepareMLMInput - nn.Targets[%d]: %f", randomIndex, nn.Targets[randomIndex])
		} else {
			j-- // retry
		}
	}

	return maskedIndices, nil
}
