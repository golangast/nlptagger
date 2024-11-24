package predict

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"strings"

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
	predictedOutput := pos.ForwardPassPos(nn, inputs)
	predictedTagIndex := nnu.MaxIndex(predictedOutput)
	predictedPosTag, _ := pos.IndexToPosTag(posTagVocab, predictedTagIndex)
	predictedOutputNer := ner.ForwardPassNer(nn, inputs)
	predictedTagIndexNer := nnu.MaxIndex(predictedOutputNer)

	predictedNerTag, _ := ner.IndexToNerTag(nerTagVocab, predictedTagIndexNer)
	predictedPhraseTag, _ := phrase.IndexToPhraseTag(phraseTagVocab, predictedTagIndex)
	predictedDRTag, _ := dr.IndexToDRTag(drTagVocab, predictedTagIndex)
	return predictedPosTag, predictedNerTag, predictedPhraseTag, predictedDRTag
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
			// Handle unknown words using the UNK token index
			tokenIndex = tokenVocab["UNK"]
			fmt.Printf("Token '%s' not found in vocabulary. Using UNK.\n", token)
		}

		inputs[tokenIndex] = 1

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

func CreateTokenVocab(trainingData []tag.Tag) map[string]int {
	tokenVocab := make(map[string]int)
	tokenVocab["UNK"] = 0 // Add "UNK" token initially
	index := 1
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

		// Optionally, print a warning message
		//fmt.Println("Warning: Vocabulary size exceeded. Adding 'UNK' token.")
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
