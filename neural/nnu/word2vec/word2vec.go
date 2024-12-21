package word2vec

import (
	"fmt"
	"math/rand/v2"
	"strings"
)

// Vocabulary represents the mapping of words to unique IDs
type Vocabulary map[string]int

// WordVectors represents the word embeddings
type WordVectors map[int][]float64

// SimpleWord2Vec is a basic Word2Vec implementation in Go.
// Note: This is a highly simplified example for illustration. It
// does NOT include negative sampling or other important optimizations.
type SimpleWord2Vec struct {
	Vocabulary  Vocabulary
	WordVectors WordVectors
	VectorSize  int
	Window      int
}

func NewSimpleWord2Vec(vectorSize, window int) *SimpleWord2Vec {
	return &SimpleWord2Vec{
		Vocabulary:  make(Vocabulary),
		WordVectors: make(WordVectors),
		VectorSize:  vectorSize,
		Window:      window,
	}
}

func (w2v *SimpleWord2Vec) Train(sentences []string, epochs int) {

	// Build vocabulary
	wordID := 0
	for _, sentence := range sentences {
		for _, word := range strings.Fields(sentence) {
			if _, ok := w2v.Vocabulary[word]; !ok {
				w2v.Vocabulary[word] = wordID
				w2v.WordVectors[wordID] = make([]float64, w2v.VectorSize)
				for i := range w2v.WordVectors[wordID] {
					w2v.WordVectors[wordID][i] = rand.Float64() // Initialize with random values
				}
				wordID++
			}
		}
	}

	// Training loop (Highly simplified example)
	for epoch := 0; epoch < epochs; epoch++ {
		for _, sentence := range sentences {
			words := strings.Fields(sentence)
			for i := 0; i < len(words); i++ {
				// ... (Simplified training for demonstration) ...
				// Skip the implementation of the core training logic since the
				// negative sampling optimization needs to be included.

				fmt.Printf("Processing word '%s' in epoch %d\n", words[i], epoch)
			}
		}
	}

}
