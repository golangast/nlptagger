package vocab

import (
	"encoding/gob"
	"fmt"
	"os"

	"github.com/golangast/nlptagger/neural/nn/dr"
	"github.com/golangast/nlptagger/neural/nn/ner"
	"github.com/golangast/nlptagger/neural/nn/phrase"
	"github.com/golangast/nlptagger/neural/nn/pos"
	"github.com/golangast/nlptagger/neural/nnu/train"
	"github.com/golangast/nlptagger/tagger/tag"
)

func CreateVocab() (map[string]int, map[string]int, map[string]int, map[string]int, map[string]int, *train.TrainingDataJSON) {
	trainingData, err := train.LoadTrainingDataFromJSON("datas/tagdata/training_data.json")
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

	}

	return tokenVocab
}

// CreateAndSaveVocab creates a vocabulary from training data and saves it as a GOB file.
func CreateAndSaveVocab(trainingData []tag.Tag, filePath string) (map[string]int, error) {
	vocabulary := make(map[string]int)
	vocabulary["UNK"] = 0 // Add "UNK" token with index 0
	index := 1

	for _, sentence := range trainingData {
		for _, token := range sentence.Tokens {
			if _, ok := vocabulary[token]; !ok {
				vocabulary[token] = index
				index++
			}
		}
	}

	file, err := os.Create(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	return vocabulary, encoder.Encode(vocabulary)
}
