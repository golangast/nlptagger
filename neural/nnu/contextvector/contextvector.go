package contextvector

import (
	"fmt"
	"strings"

	"github.com/golangast/nlptagger/neural/nnu"
	"github.com/golangast/nlptagger/neural/nnu/predict"
	"github.com/golangast/nlptagger/neural/nnu/word2vec"
)

// getContextVector calculates the context vector for a sentence.
func GetContextVector(sentence string, md *nnu.SimpleNN, sw2v *word2vec.SimpleWord2Vec) []float64 {
	words := strings.Fields(sentence)

	predictedPosTags, predictedNerTags, predictedPhraseTags, predictedDRTags := predict.PredictTags(md, words)

	// context := make([]float64, sw2v.VectorSize*len(words)+len(predictedPosTags)+len(predictedNerTags)+len(predictedPhraseTags)+len(predictedDRTags))
	// offset := 0
	// Correct calculation of context size:
	contextSize := sw2v.VectorSize * len(words)
	contextSize += len(predictedPosTags)
	contextSize += len(predictedNerTags)
	contextSize += len(predictedPhraseTags)
	contextSize += len(predictedDRTags)
	context := make([]float64, contextSize) // Allocate the correct size
	offset := 0
	// Create sequential maps for each tag type
	posTagMap := createSequentialMap(predictedPosTags)
	nerTagMap := createSequentialMap(predictedNerTags)
	phraseTagMap := createSequentialMap(predictedPhraseTags)
	drTagMap := createSequentialMap(predictedDRTags)

	// Calculate max values for normalization
	maxPosVal := float64(len(posTagMap))
	maxNerVal := float64(len(nerTagMap))
	maxPhraseVal := float64(len(phraseTagMap))
	maxDrVal := float64(len(drTagMap))

	// Weights for word vectors and tag components
	wordVectorWeight := 0.8 // Weight for the word vectors
	posTagWeight := 0.8     // Weight for POS tags
	nerTagWeight := 0.8     // Weight for NER tags
	phraseTagWeight := 0.8  // Weight for phrase tags
	drTagWeight := 0.8      // Weight for dependency relation tags

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
		neighbors, err := sw2v.Ann.NearestNeighbors(sentence, sw2v.WordVectors[wordIndex], 5) // Get 5 nearest neighbors using the ANN index
		if err != nil {
			fmt.Printf("Error getting nearest neighbors: %v\n", err)
		} else {
			augmentedVector := make([]float64, sw2v.VectorSize)
			copy(augmentedVector, sw2v.WordVectors[wordIndex]) // Start with the original vector

			// Average the vectors
			for j := 0; j < sw2v.VectorSize; j++ {
				augmentedVector[j] /= float64(len(neighbors) + 1)
				context[offset+j] = augmentedVector[j] * wordVectorWeight
			}
			copy(context[offset:offset+sw2v.VectorSize], augmentedVector) // Add the modified vector to the context
			offset += sw2v.VectorSize                                     // Increment offset for the next word
		}

		// Directly use predicted tags with normalization
		if i < len(predictedPosTags) {
			context[offset] = float64(posTagMap[predictedPosTags[i]]) / maxPosVal * posTagWeight
			offset++
		}
		if i < len(predictedNerTags) {
			context[offset] = float64(nerTagMap[predictedNerTags[i]]) / maxNerVal * nerTagWeight
			offset++
		}
		if i < len(predictedPhraseTags) {
			context[offset] = float64(phraseTagMap[predictedPhraseTags[i]]) / maxPhraseVal * phraseTagWeight
			offset++
		}
		if i < len(predictedDRTags) {
			context[offset] = float64(drTagMap[predictedDRTags[i]]) / maxDrVal * drTagWeight
			offset++
		}
	}
	return context
}

// Function to create a map with sequential integer values
func createSequentialMap(keys []string) map[string]int {
	m := make(map[string]int)
	for i, key := range keys {
		m[key] = i
	}
	return m
}
