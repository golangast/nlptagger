package g

import (
	"math"
	"sort"

	"github.com/google/btree"
)

type ANN struct {
	Index *btree.BTreeG[WordVector]
}

type WordVector struct {
	Word   string
	Vector []float64
}

type Neighbor struct { // Define a Neighbor struct
	Word       string
	Vector     []float64
	Similarity float64
}

func NewANN(vectorSize int, metric string) (*ANN, error) {
	ann := &ANN{
		Index: btree.NewG(2, func(a, b WordVector) bool {
			return a.Word < b.Word // Placeholder comparison - replace with meaningful logic!
		}),
	}
	return ann, nil
}

func (ann *ANN) AddWordVectors(vocabulary map[string][]float64) {
	for word, vector := range vocabulary {
		wv := WordVector{Word: word, Vector: vector}
		ann.Index.ReplaceOrInsert(wv)
	}
}

func (ann *ANN) NearestNeighbors(vector []float64, k int) ([]Neighbor, error) { // Return []Neighbor
	var neighbors []Neighbor
	ann.Index.Ascend(func(item WordVector) bool {
		similarity := cosineSimilarity(vector, item.Vector)
		neighbors = append(neighbors, Neighbor{Word: item.Word, Vector: item.Vector, Similarity: similarity})
		return true
	})

	// Sort neighbors by similarity
	sort.Slice(neighbors, func(i, j int) bool {
		return neighbors[i].Similarity > neighbors[j].Similarity
	})

	if len(neighbors) > k {
		return neighbors[:k], nil
	}
	return neighbors, nil
}

// cosineSimilarity calculates the cosine similarity between two vectors.
func cosineSimilarity(v1, v2 []float64) float64 {
	// Check if vectors have different lengths
	if len(v1) != len(v2) {
		return 0.0 // Or handle the error as you prefer
	}

	dotProduct := 0.0
	magV1 := 0.0
	magV2 := 0.0

	for i := 0; i < len(v1); i++ {
		dotProduct += v1[i] * v2[i]
		magV1 += math.Pow(v1[i], 2)
		magV2 += math.Pow(v2[i], 2)
	}

	magV1 = math.Sqrt(magV1)
	magV2 = math.Sqrt(magV2)

	if magV1 == 0 || magV2 == 0 {
		return 0.0
	}
	return dotProduct / (magV1 * magV2)
}
