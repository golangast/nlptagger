// Package contextvector computes context vectors for text.
// It's pretty simple, just averages word vectors for now.

package contextvector

import (
	"strings"

	"github.com/zendrulat/nlptagger/neural/nn/g"
	"github.com/zendrulat/nlptagger/neural/nnu"
	"github.com/zendrulat/nlptagger/neural/nnu/word2vec"
)

// GetContextVector calculates the context vector for a sentence.
func GetContextVector(sentence string, md *nnu.SimpleNN, sw2v *word2vec.SimpleWord2Vec) []float64 {
	contextVector := g.CalculateContextVector(sentence, sw2v.Ann.Index)
	contextVector = padOrTrimVector(contextVector, sw2v.VectorSize)

	return contextVector
}

// padOrTrimVector pads or trims a vector to the target length.
func padOrTrimVector(vector []float64, targetLength int) []float64 {
	currentLength := len(vector)
	if currentLength == targetLength {
		return vector
	}

	if currentLength < targetLength {
		// Pad with zeros
		paddedVector := make([]float64, targetLength)
		copy(paddedVector, vector)
		return paddedVector
	} else {
		// Trim to target length
		return vector[:targetLength]
	}
}

// CalculateContextVector computes the context vector for a given sentence.
func CalculateContextVector(sentence string, index map[string][]float64) []float64 {
	words := strings.Fields(sentence)
	var contextVector []float64
	var numWordsInIndex int
	for _, word := range words {
		if wordVector, exists := index[word]; exists {
			if contextVector == nil {
				contextVector = make([]float64, len(wordVector))
			}
			for i, val := range wordVector {
				contextVector[i] += val
			}
			numWordsInIndex++
		}
	}
	if numWordsInIndex > 0 {
		for i := range contextVector {
			contextVector[i] /= float64(numWordsInIndex)
		}
	}
	return contextVector
}
