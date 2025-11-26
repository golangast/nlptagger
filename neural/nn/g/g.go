// Package g provides a basic implementation for an Approximate Nearest Neighbor search.

package g

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand/v2"
	"sort"
	"strconv"
	"strings"

	"github.com/zendrulat/nlptagger/tagger"
)

type ANN struct {
	Index map[string][]float64
}
type WordVector struct {
	Word   string    `json:"words"`
	Vector []float64 `json:"vectors"`
}

type Neighbor struct { // Define a Neighbor struct
	Word                string
	Vector              []float64
	Similarity          float64
	ContextualRelevance float64
}

func NewANN(vectorSize int, metric string) (*ANN, error) {
	ann := &ANN{
		Index: make(map[string][]float64), // Initialize the map
	}
	return ann, nil
}

func (ann *ANN) AddWordVectors(vocabulary map[string][]float64) {
	for word, vector := range vocabulary {
		ann.Index[word] = vector // Directly assign to the map
	}
}

func (ann *ANN) NearestNeighbors(sentence string, vector []float64, k int) ([]Neighbor, error) {
	//fmt.Println("Context of the sentence:", sentence)
	neighbors := make([]Neighbor, 0)

	var sentenceWords []string
	// Get all words in the sentence
	filteredsentence := tagger.Filtertags(sentence)
	sentenceWords = strings.Split(filteredsentence, " ")
	contextVector := CalculateContextVector(filteredsentence, ann.Index)
	vector = addVectors(vector, contextVector)
	var contextMostRelated string
	var maxSimilarity = 0.0
	for word, wordVector := range ann.Index {
		contextualRelevance := CosineSimilarity(wordVector, contextVector)
		if contextualRelevance > maxSimilarity {
			maxSimilarity = contextualRelevance

			contextMostRelated = word
		}

		// Filter words by sentence
		var contains bool = false

		if len(sentenceWords) > 0 {
			for _, wordInSentence := range sentenceWords {
				if wordInSentence == word {
					contains = true
					break
				}
			}

			if contains {

				similarity := CosineSimilarity(vector, wordVector)
				contextualRelevance := CosineSimilarity(wordVector, contextVector)
				neighbors = append(neighbors, Neighbor{
					Word:                word,
					Vector:              wordVector,
					Similarity:          similarity,
					ContextualRelevance: contextualRelevance,
				})
			}
		} else {
			//fmt.Println("Sentence words is empty, skipping nearest neighbors")
			similarity := CosineSimilarity(vector, wordVector)
			contextualRelevance := CosineSimilarity(wordVector, contextVector)
			neighbors = append(neighbors, Neighbor{Word: word, Vector: wordVector, Similarity: similarity, ContextualRelevance: contextualRelevance})

		}
		// trainingdata, err := LoadTrainingData("trainingdata/context/data/training_data.json")
		// if err != nil {
		// 	fmt.Println("Error loading training data:", err)
		// }
		// generatedText, err := ann.generateText(sentence, trainingdata)
		// if err != nil {
		// 	fmt.Println("Error generating text:", err)
		// 	//fallback to the description
		// 	fmt.Println("Context of the sentence:", filteredsentence)

		// } else {
		// 	fmt.Println("Generated text:", generatedText)
		// }
		// Track the top 3 most related words
		type WordSimilarity struct {
			Word       string
			Similarity float64
		}
		var topRelatedWords []WordSimilarity
		for word, wordVector := range ann.Index {
			contextualRelevance := CosineSimilarity(wordVector, contextVector)
			topRelatedWords = append(topRelatedWords, WordSimilarity{Word: word, Similarity: contextualRelevance})
		}

		sort.Slice(topRelatedWords, func(i, j int) bool {
			return topRelatedWords[i].Similarity > topRelatedWords[j].Similarity
		})
		if len(topRelatedWords) > 3 {
			topRelatedWords = topRelatedWords[:3]
		}
		// Create the context description
		contextDescription := fmt.Sprintf("The context of the sentence is related to: '%s'", contextMostRelated)
		if maxSimilarity < 0.3 {
			contextDescription = fmt.Sprintf("The context of the sentence is not very related to any known word")
		}
		contextDescription += fmt.Sprintf(", using the words: '%s'. With a relation of: %s", strings.Join(sentenceWords, ", "), strconv.FormatFloat(maxSimilarity, 'f', 6, 64))

		// Print the context description
		//fmt.Println(contextDescription)

		//fmt.Println("The context of the sentence is related to: '" + contextMostRelated + "', using the words: '" + strings.Join(sentenceWords, ", ") + "'. With a relation of: " + strconv.FormatFloat(maxSimilarity, 'f', 6, 64))

		// Sort neighbors by similarity
		sort.Slice(neighbors, func(i, j int) bool {
			return neighbors[i].Similarity > neighbors[j].Similarity
		})

		if len(neighbors) > k {
			return neighbors[:k], nil
		}
	}
	//fmt.Println("The context of the sentence is related to: '" + contextMostRelated + "', using the words: '" + strings.Join(sentenceWords, ", ") + "'. With a relation of: " + strconv.FormatFloat(maxSimilarity, 'f', 6, 64))

	return neighbors, nil
}

// cosineSimilarity calculates the cosine similarity between two vectors.
func CosineSimilarity(v1, v2 []float64) float64 {
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
func addVectors(v1, v2 []float64) []float64 {
	if len(v1) != len(v2) {
		panic("Vectors must have the same length for addition")
	}
	result := make([]float64, len(v1))
	for i := range v1 {
		result[i] = v1[i] + v2[i]
	}
	return result
}

// filterBySentenceSimilarity filters the sentences by similarity
func FilterBySentenceSimilarity(sentence string, sentences []TrainingData, index map[string][]float64) []TrainingData {
	type SentenceSimilarity struct {
		Data       TrainingData
		Similarity float64
	}

	var sentenceSimilarities []SentenceSimilarity
	for _, data := range sentences {
		sentenceVector := CalculateContextVector(sentence, index)
		dataVector := CalculateContextVector(data.Sentence, index)
		similarity := CosineSimilarity(sentenceVector, dataVector)
		sentenceSimilarities = append(sentenceSimilarities, SentenceSimilarity{Data: data, Similarity: similarity})
	}
	sort.Slice(sentenceSimilarities, func(i, j int) bool {
		return sentenceSimilarities[i].Similarity > sentenceSimilarities[j].Similarity
	})

	var sortedSentences []TrainingData
	for _, data := range sentenceSimilarities {
		sortedSentences = append(sortedSentences, data.Data)
	}
	return sortedSentences
}

// ... (Your existing code - ANN struct, Neighbor struct, cosineSimilarity, calculateContextVector, addVectors) ...
type TrainingData struct {
	Sentence string `json:"sentence"`
	Context  string `json:"context"`
}

func LoadTrainingData(filepath string) ([]TrainingData, error) {
	// Load data
	fileContent, err := ioutil.ReadFile(filepath)
	if err != nil {
		return nil, err
	}

	var data []TrainingData
	err = json.Unmarshal(fileContent, &data)
	if err != nil {
		return nil, err
	}

	return data, nil
}

// generateText generates text based on the context of the input sentence
func (ann *ANN) generateText(sentence string, trainingData []TrainingData) (string, error) {
	//Get the context of the sentence
	filteredsentence := tagger.Filtertags(sentence)
	contextVector := CalculateContextVector(filteredsentence, ann.Index)
	//get the most related word to the context
	var contextMostRelated string
	var maxSimilarity = 0.0
	for word, wordVector := range ann.Index {
		contextualRelevance := CosineSimilarity(wordVector, contextVector)
		if contextualRelevance > maxSimilarity {
			maxSimilarity = contextualRelevance

			contextMostRelated = word
		}
	}
	// Create a set of keywords
	keywords := strings.Fields(filteredsentence)
	keywords = append(keywords, contextMostRelated)

	// Find matching sentences
	var matchingSentences []TrainingData
	for _, data := range trainingData {
		if containsAnyKeyword(data.Sentence, keywords) {
			matchingSentences = append(matchingSentences, data)
		}
	}

	// Filter by sentence similarity
	if len(matchingSentences) > 0 {
		matchingSentences = FilterBySentenceSimilarity(sentence, matchingSentences, ann.Index)
	}

	// Select a random sentence
	if len(matchingSentences) > 0 {
		rand.IntN(100)
		randomIndex := rand.IntN(len(matchingSentences))
		return matchingSentences[randomIndex].Sentence, nil
	}
	return "", fmt.Errorf("no matching sentences found")
}

// containsAnyKeyword checks if the sentence contains any of the keywords
func containsAnyKeyword(sentence string, keywords []string) bool {
	sentenceLower := strings.ToLower(sentence)
	for _, keyword := range keywords {
		keywordLower := strings.ToLower(keyword)
		if strings.Contains(sentenceLower, keywordLower) {
			return true
		}
	}
	return false
}