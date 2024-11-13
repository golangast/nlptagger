package nnu

import (
	"encoding/json"
	"io/ioutil"
	"os"

	"github.com/golangast/nlptagger/tagger/tag"
)

// Structure to represent training data in JSON
type TrainingDataJSON struct {
	Sentences []tag.Tag `json:"sentences"`
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
func IndexToPosTag(posTagVocab map[string]int, predictedTagIndex int) (string, bool) {
	for tag, index := range posTagVocab {
		if index == predictedTagIndex {
			return tag, true
		}
	}
	return "", false
}
func IndexToNerTag(nerTagVocab map[string]int, predictedTagIndex int) (string, bool) {
	for tag, index := range nerTagVocab {
		if index == predictedTagIndex {
			return tag, true
		}
	}
	return "", false
}
func IndexToPhraseTag(phraseTagVocab map[string]int, predictedTagIndex int) (string, bool) {
	for tag, index := range phraseTagVocab {
		if index == predictedTagIndex {
			return tag, true
		}
	}
	return "", false
}
func IndexToDRTag(drTagVocab map[string]int, predictedTagIndex int) (string, bool) {
	for tag, index := range drTagVocab {
		if index == predictedTagIndex {
			return tag, true
		}
	}
	return "", false
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
