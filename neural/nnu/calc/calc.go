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
	targetOutput := make([]float64, nn.OutputSize)
	vocabularies := []func([]tag.Tag) map[string]int{
		pos.CreatePosTagVocab,
		ner.CreateTagVocabNer,
		phrase.CreatePhraseTagVocab,
		dr.CreateDRTagVocab,
	}
	vocabs := make([]map[string]int, len(vocabularies))

	for i, createVocab := range vocabularies {
		vocabs[i] = createVocab(trainingData)
		tagIndex, ok := vocabs[i][targetTag]
		if ok && tagIndex < nn.OutputSize { // Check index boundaries
			targetOutput[tagIndex] = 1
		}
	}

	errors := make([]float64, nn.OutputSize)
	for i := range errors {
		errors[i] = targetOutput[i] - outputs[i]
	}
	return errors, vocabs[0], vocabs[1], vocabs[2], vocabs[3]
}

// Function to calculate accuracy
func CalculateAccuracy(nn *nnu.SimpleNN, trainingData []tag.Tag, tokenVocab map[string]int, posTagVocab map[string]int, nerTagVocab map[string]int, phraseTagVocab map[string]int, drTagVocab map[string]int) (float64, float64, float64, float64) {
	var (
		poscorrectPredictions    = 0
		nercorrectPredictions    = 0
		phrasecorrectPredictions = 0
		drcorrectPredictions     = 0
		postotalPredictions      = 0
		nertotalPredictions      = 0
		phrasetotalPredictions   = 0
		drtotalPredictions       = 0
	)

	for _, taggedSentence := range trainingData {
		for i := range taggedSentence.Tokens {
			inputs := make([]float64, nn.InputSize)
			tokenIndex, ok := tokenVocab[taggedSentence.Tokens[i]]
			if ok {
				inputs[tokenIndex] = 1
			} else {
				inputs[tokenVocab["UNK"]] = 1
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
			if i < len(taggedSentence.Dependencies) {
				if predictedDRTag == taggedSentence.Dependencies[i].Dep {
					drcorrectPredictions++
				} else {
					// Explicitly handle the case where dependency tag is missing
				}
			}
			postotalPredictions++
			nertotalPredictions++
			phrasetotalPredictions++
			drtotalPredictions++
		}
	}

	pacc := float64(poscorrectPredictions) / float64(postotalPredictions)
	nacc := float64(nercorrectPredictions) / float64(nertotalPredictions)
	phacc := float64(phrasecorrectPredictions) / float64(phrasetotalPredictions)
	dracc := float64(drcorrectPredictions) / float64(drtotalPredictions)

	return pacc, nacc, phacc, dracc
}
