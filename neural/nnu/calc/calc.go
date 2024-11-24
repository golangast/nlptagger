package calc

import (
	"github.com/golangast/nlptagger/neural/nn/dr"
	"github.com/golangast/nlptagger/neural/nn/ner"
	"github.com/golangast/nlptagger/neural/nn/phrase"
	"github.com/golangast/nlptagger/neural/nn/pos"
	"github.com/golangast/nlptagger/neural/nnu"
	"github.com/golangast/nlptagger/neural/nnu/predict"
	"github.com/golangast/nlptagger/tagger/tag"
)

func CalculateError(targetTag string, outputs []float64, trainingData []tag.Tag, nn *nnu.SimpleNN) ([]float64, map[string]int, map[string]int, map[string]int, map[string]int) {
	// Create a slice to store the target output values.
	targetOutput := make([]float64, nn.OutputSize)
	// Create a vocabulary of POS tags from the training data.
	posTagVocab := pos.CreatePosTagVocab(trainingData)
	// Get the index of the target POS tag in the vocabulary.
	targetTagIndex, ok := posTagVocab[targetTag]
	if ok {
		// Set the corresponding element in the target output to 1.
		targetOutput[targetTagIndex] = 1
	}

	nerTagVocab := ner.CreateTagVocabNer(trainingData) // Use NER vocab
	targetNerTagIndex, ok := nerTagVocab[targetTag]
	// If the target POS tag is found in the vocabulary...
	if ok {
		// Set the corresponding element in the target output to 1.
		targetOutput[targetNerTagIndex] = 1
	}
	phraseTagVocab := phrase.CreatePhraseTagVocab(trainingData) // Use NER vocab
	targetPhraseTagIndex, ok := phraseTagVocab[targetTag]
	// If the target POS tag is found in the vocabulary...
	if ok {
		// Set the corresponding element in the target output to 1.
		targetOutput[targetPhraseTagIndex] = 1
	}
	drTagVocab := dr.CreateDRTagVocab(trainingData) // Use NER vocab
	targetDRTagIndex, ok := drTagVocab[targetTag]
	// If the target POS tag is found in the vocabulary...
	if ok {
		// Set the corresponding element in the target output to 1.
		targetOutput[targetDRTagIndex] = 1
	}
	// Create a slice to store the errors for each output neuron.
	errors := make([]float64, nn.OutputSize)
	// Calculate the error for each output neuron.
	for i := range errors {
		errors[i] = targetOutput[i] - outputs[i]
	}
	return errors, posTagVocab, nerTagVocab, phraseTagVocab, drTagVocab
}

// Function to calculate accuracy
func CalculateAccuracy(nn *nnu.SimpleNN, trainingData []tag.Tag, tokenVocab map[string]int, posTagVocab map[string]int, nerTagVocab map[string]int, phraseTagVocab map[string]int, drTagVocab map[string]int) (float64, float64, float64, float64) {
	poscorrectPredictions := 0
	nercorrectPredictions := 0
	phrasecorrectPredictions := 0
	drcorrectPredictions := 0
	postotalPredictions := 0
	nertotalPredictions := 0
	phrasetotalPredictions := 0
	drtotalPredictions := 0

	for _, taggedSentence := range trainingData {
		for i := range taggedSentence.Tokens {
			inputs := make([]float64, nn.InputSize)
			tokenIndex, ok := tokenVocab[taggedSentence.Tokens[i]]
			if ok {
				inputs[tokenIndex] = 1
			} else {
				inputs[tokenVocab["UNK"]] = 1 // Handle unknown tokens
			}

			predictedPosTag, predictedNerTag, predictedPhraseTag, predictedDRTag := predict.Predict(nn, inputs, posTagVocab, nerTagVocab, phraseTagVocab, drTagVocab)

			if predictedPosTag == taggedSentence.PosTag[i] {
				poscorrectPredictions++
			}
			if predictedNerTag == taggedSentence.NerTag[i] {
				nercorrectPredictions++
			}
			if predictedPhraseTag == taggedSentence.PhraseTag[i] {
				phrasecorrectPredictions++
			}
			if len(taggedSentence.Dependencies) > i { // Check if Dependency slice has an element at index i
				if predictedDRTag == taggedSentence.Dependencies[i].Dep {
					drcorrectPredictions++
				}
			} else {
				inputs[drTagVocab["UNK"]] = 1 // Handle unknown tokens
				//fmt.Printf("Warning: Dependency tag missing for token %d in sentence: %v\n", i, taggedSentence.Tokens)
				// Handle the missing tag appropriately.
				// Options:
				// 1. Skip the token for DR accuracy calculation (current behavior).
				// 2. Count it as an incorrect prediction.
				// 3. Use a default value for comparison.

			}
			postotalPredictions++
			nertotalPredictions++
			phrasetotalPredictions++
			drtotalPredictions++
		}
	}

	return float64(poscorrectPredictions) / float64(postotalPredictions), float64(nercorrectPredictions) / float64(nertotalPredictions), float64(phrasecorrectPredictions) / float64(phrasetotalPredictions), float64(drcorrectPredictions) / float64(drtotalPredictions)
}
