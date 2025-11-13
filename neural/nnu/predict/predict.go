// Package predict handles predicting various tags for input text using a neural network.
// It includes functions for forward passing, vocabulary handling, and loss calculation.

package predict

import (
	"encoding/gob"
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
	"github.com/zendrulat/nlptagger/neural/nnu/gobs"
	"github.com/zendrulat/nlptagger/tagger/tag"
)

// VocabConfidence holds the vocabularies and the average confidence score.
type VocabConfidence struct {
	TokenVocab     map[string]int
	PosTagVocab    map[string]int
	NerTagVocab    map[string]int
	PhraseTagVocab map[string]int
	DrTagVocab     map[string]int
	Confidence     float64
}

// vocabCache stores vocabularies for sentences.
var vocabCache = make(map[string]VocabConfidence)

// Structure to represent training data in JSON
type TrainingDataJSON struct {
	Sentences []tag.Tag `json:"sentences"`
}

// predictTag predicts a tag based on the provided forward pass function and vocabulary.
func PredictTag(nn *nnu.SimpleNN, inputs []float64, posTagVocab, nerTagVocab, phraseTagVocab, drTagVocab map[string]int) (string, string, string, string, float64) {
	//fmt.Println("istagged: ", posTagVocab, nerTagVocab, phraseTagVocab, drTagVocab)

	// Create a slice of probability-tag pairs for sorting
	type ProbabilityTagPair struct {
		Probability float64
		Tag         string
	}

	// Function to predict and select tag
	predictAndSelectTag := func(vocab map[string]int, predictedOutput []float64, tagType string) (string, float64) {
		if vocab == nil {
			log.Printf("Error: Nil tag vocabulary for %s.", tagType)
			return "", 0.0
		}

		var probabilityTagPairs []ProbabilityTagPair
		for tag, index := range vocab {
			if index >= 0 && index < len(predictedOutput) {
				probabilityTagPairs = append(probabilityTagPairs, ProbabilityTagPair{Probability: predictedOutput[index], Tag: tag})
			}
		}
		sort.Slice(probabilityTagPairs, func(i, j int) bool {
			return probabilityTagPairs[i].Probability > probabilityTagPairs[j].Probability
		})
		if len(probabilityTagPairs) > 0 {
			return probabilityTagPairs[0].Tag, probabilityTagPairs[0].Probability
		} else {
			log.Printf("Error: Empty tag vocabulary for %s.", tagType)
			return "", 0.0
		}
	}
	nnOutput := nn.ForwardPass(inputs)

	// Predict each tag
	predictedPos, posConfidence := predictAndSelectTag(posTagVocab, nnOutput, "POS")
	predictedNer, nerConfidence := predictAndSelectTag(nerTagVocab, nnOutput, "NER")
	predictedPhrase, phraseConfidence := predictAndSelectTag(phraseTagVocab, nnOutput, "Phrase")
	predictedDR, drConfidence := predictAndSelectTag(drTagVocab, nnOutput, "DR")

	avgConfidence := (posConfidence + nerConfidence + phraseConfidence + drConfidence) / 4.0

	return predictedPos, predictedNer, predictedPhrase, predictedDR, avgConfidence
}

func ForwardPassPos(nn *nnu.SimpleNN, inputs []float64) []float64 {
	if inputs == nil || len(inputs) == 0 {
		log.Println("Warning: Empty or nil inputs provided to ForwardPassPos")
		return []float64{} // Return an empty slice or handle this case as needed
	}
	outputs := nn.ForwardPass(inputs) // Changed nn.ForwardPass
	if len(outputs) == 0 {
		log.Println("Warning: Empty outputs returned from ForwardPass")
		return []float64{} // Return an empty slice if outputs are empty

	}
	return outputs
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
		randomIndex := rand.IntN(len(inputs))

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

func prepareMLMInput(nn *nnu.SimpleNN) error {
	// 4. Masking Logic (Example - 15% masking):
	numTokensToMask := int(0.15 * float64(len(nn.MaskedInputs)))
	maskedIndices := make(map[int]bool)

	for i := 0; i < numTokensToMask; i++ {
		randomIndex := rand.IntN(len(nn.MaskedInputs))
		if !maskedIndices[randomIndex] {
			maskedIndices[randomIndex] = true
			nn.MaskedInputs[randomIndex] = 0 // Replace with mask token ID
		} else {
			i-- // Retry if the index was already masked
		}
	}
	nn.MaskedIndices = make([]int, 0, len(maskedIndices))
	for index := range maskedIndices {
		nn.MaskedIndices = append(nn.MaskedIndices, index)
	}

	// 5. Set Targets (assuming you have a target vocabulary):
	nn.Targets = make([]float64, len(nn.MaskedInputs))
	for i := range nn.Targets {
		if maskedIndices[i] {
			nn.Targets[i] = nn.Inputs[i] // Store the original value for masked tokens
		} else {
			nn.Targets[i] = -1 // Set target to -1 for unmasked tokens
		}
	}

	return nil
}
func PredictMaskedWords(nn *nnu.SimpleNN) ([]float64, error) {
	// Add debugging output before any access to nn.MaskedInputs
	err := prepareMLMInput(nn)
	if err != nil {
		log.Printf("Error in prepareMLMInput: %v\n", err)
		return nil, err
	}
	if nn.Targets == nil {
		log.Println("Error: nn.Targets is nil.")
		return []float64{}, nil
	}
	outputs := nnu.ForwardPassMasked(nn)
	if len(nn.MaskedInputs) == 0 {
		log.Println("Error: nn.MaskedInputs is empty.")
		return []float64{}, nil // Returning empty slice on error
	}

	return outputs, nil
}

// CalculateOriginalLoss calculates the original loss for your primary task.
// In this example, it calculates the mean squared error loss for a regression task.
func CalculateOriginalLoss(nn *nnu.SimpleNN, predictedOutput []float64, targetOutput []float64) float64 {
	// Check lengths
	if len(predictedOutput) != len(nn.Targets) {
		fmt.Printf("Lengths do not match: predictedOutput = %v nn.Targets= %v", len(predictedOutput), len(nn.Targets))
	}
	loss := 0.0
	for i := 0; i < len(predictedOutput); i++ {
		diff := predictedOutput[i] - nn.Targets[i]
		loss += diff * diff
	}

	totalLoss := loss / float64(len(predictedOutput))
	log.Println("CalculateOriginalLoss: totalLoss", totalLoss)

	return totalLoss
}

func CalculateMLMLoss(nn *nnu.SimpleNN, maskedInputs []float64, originalInputs []float64, maskedIndices map[int]bool) float64 {

	predictedOutputs := nn.ForwardPass(maskedInputs)

	if nn.InputSize != len(maskedInputs) || nn.OutputSize != len(predictedOutputs) {
		log.Printf("nn.InputSize (%d) != len(maskedInputs) (%d) || nn.OutputSize (%d) != len(predictedOutputs) (%d)\n", nn.InputSize, len(maskedInputs), nn.OutputSize, len(predictedOutputs))
	}

	totalLoss := 0.0
	numMaskedWords := 0

	for index, isMasked := range maskedIndices {
		if isMasked {
			targetWordIndex := int(originalInputs[index])
			if targetWordIndex < 0 || targetWordIndex >= len(nn.Targets) {
				log.Printf("invalid targetWordIndex: %d, index: %d, length of Targets %d, length outputs: %d", targetWordIndex, index, len(nn.Targets), len(predictedOutputs))
				continue
			}
			if targetWordIndex >= len(predictedOutputs) || targetWordIndex < 0 {
				log.Printf("targetWordIndex (%d) is out of bounds for predictedOutputs (len = %d)\n", targetWordIndex, len(predictedOutputs))
				continue
			}
			// Calculate cross-entropy loss

			loss := -math.Log(predictedOutputs[targetWordIndex])
			if math.IsNaN(loss) {
				log.Printf("loss is not a number %f", loss)
			}
			totalLoss += loss
			numMaskedWords++

		}
	}

	// Average loss over masked words
	if numMaskedWords > 0 {
		totalLoss := totalLoss / float64(numMaskedWords)
		log.Println("CalculateMLMLoss: totalLoss", totalLoss, "numMaskedWords", numMaskedWords)
		return totalLoss
	} else {
		return 0.0 // Handle case where no words were masked
	}
}

func PredictTags(nn *nnu.SimpleNN, sentence []string) ([]string, []string, []string, []string) {
	sentenceKey := strings.Join(sentence, " ")
	var tokenVocab, posTagVocab, nerTagVocab, phraseTagVocab, drTagVocab map[string]int

	if cached, found := vocabCache[sentenceKey]; found {
		tokenVocab = cached.TokenVocab
		posTagVocab = cached.PosTagVocab
		nerTagVocab = cached.NerTagVocab
		phraseTagVocab = cached.PhraseTagVocab
		drTagVocab = cached.DrTagVocab
	} else {
		tokenVocab, posTagVocab, nerTagVocab, phraseTagVocab, drTagVocab, _ = CreateVocab()
	}
	// Create a slice to store the predicted POS tags.
	var predictedPosTags, predictedNerTags, predictedPhraseTags, predictedDRTags []string
	var totalConfidence float64
	// Iterate over each token in the sentence.
	for _, token := range sentence {
		// Get the index of the token in the vocabulary.
		tokenIndex, ok := tokenVocab[token]
		if !ok {
			// Add new token to vocabulary and update the model.gob file
			tokenVocab = AddNewTokenToVocab(token, tokenVocab)
			tokenIndex = tokenVocab[token] // Use the newly assigned index
			if err := saveTokenVocabToGob("./gob_models/model.gob", tokenVocab); err != nil {
				log.Printf("Error saving vocabulary to GOB: %v", err)
				// Handle error appropriately, e.g., return an error or a default value
			}
		}

		inputs := make([]float64, nn.InputSize)
		//Set all values to zero
		for i := range inputs {
			inputs[i] = 0
		}

		if tokenIndex >= 0 && tokenIndex < nn.InputSize {
			inputs[tokenIndex] = 1 // Set the corresponding input to 1
		} else {
			if unkIndex, unkOk := tokenVocab["UNK"]; unkOk {
				inputs[unkIndex] = 1
			}
		}

		// Predict tags using the neural network
		predictedPosTag, predictedNerTag, predictedPhraseTag, predictedDRTag, confidence := PredictTag(nn, inputs, posTagVocab, nerTagVocab, phraseTagVocab, drTagVocab)
		totalConfidence += confidence
		predictedPosTags = append(predictedPosTags, predictedPosTag)
		predictedNerTags = append(predictedNerTags, predictedNerTag)
		predictedPhraseTags = append(predictedPhraseTags, predictedPhraseTag)
		predictedDRTags = append(predictedDRTags, predictedDRTag)
	}

	avgConfidence := totalConfidence / float64(len(sentence))
	if avgConfidence > 0.5 {
		if _, found := vocabCache[sentenceKey]; !found {
			vocabCache[sentenceKey] = VocabConfidence{
				TokenVocab:     tokenVocab,
				PosTagVocab:    posTagVocab,
				NerTagVocab:    nerTagVocab,
				PhraseTagVocab: phraseTagVocab,
				DrTagVocab:     drTagVocab,
				Confidence:     avgConfidence,
			}
		}
	}
	// Return the list of predicted POS and NER tags.
	return predictedPosTags, predictedNerTags, predictedPhraseTags, predictedDRTags
}

func CreateVocab() (map[string]int, map[string]int, map[string]int, map[string]int, map[string]int, *TrainingDataJSON) {
	trainingData, err := LoadTrainingDataFromJSON("trainingdata/tagdata/nlp_training_data.json")
	if err != nil {
		fmt.Println("error loading training data: ", err)
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
	if _, err := os.Stat("./gob_models/model.gob"); err == nil {
		// Load vocabulary from GOB file
		tokenVocab, err := loadTokenVocabFromGob("./gob_models/model.gob")
		if err != nil {
			log.Println("Error loading vocabulary from GOB:", err)
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
	if err := saveTokenVocabToGob(".././gob_models/model.gob", tokenVocab); err != nil {
		log.Println("Error saving vocabulary to GOB:", err)
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

	if _, err := os.Stat(filePath); err == nil {
		gobs.DeleteGobFile(filePath)
		file, err := os.Create(filePath)
		if err != nil {
			return err
		}
		defer file.Close()

		encoder := gob.NewEncoder(file)
		return encoder.Encode(tokenVocab)
	} else {
		file, err := os.Create(filePath)
		if err != nil {
			return err
		}
		defer file.Close()

		encoder := gob.NewEncoder(file)
		return encoder.Encode(tokenVocab)
	}

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
