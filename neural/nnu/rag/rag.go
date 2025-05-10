// Package rag provides functions for Retrieval Augmented Generation (RAG).

package rag

import (
	"bufio"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"

	"github.com/golangast/nlptagger/neural/nnu/word2vec"
)

// common English stop words
var stopWords = map[string]bool{
	"a": true, "an": true, "and": true, "are": true, "as": true, "at": true,
	"be": true, "by": true, "for": true, "from": true, "has": true, "he": true,
	"in": true, "is": true, "it": true, "its": true, "of": true, "on": true,
	"that": true, "the": true, "to": true, "was": true, "were": true, "will": true,
	"with": true, "this": true, "have": true, "or": true, "they": true, "their": true, "had": true, "would": true, "which": true, "not": true,
	"been": true, "can": true, "also": true, "could": true, "than": true, "but": true, "should": true, "do": true,
	"more": true, "about": true, "if": true, "such": true, "into": true, "so": true, "where": true, "how": true, "only": true, "any": true, "other": true, "some": true,
	"during": true,
}

// RagDocument struct
type RagDocument struct {
	ID              string             `json:"ID"`
	Content         string             `json:"Content"`
	Embedding       []float64          `json:"Embedding"`
	TermFrequencies map[string]float64 `json:"TermFrequencies"`
}

// RagDocuments type
type RagDocuments struct {
	Documents []*RagDocument
	IDF       map[string]float64
}

// CalculateIDF calculates the IDF for each term and modifies the RagDocuments IDF map.
func (docs *RagDocuments) CalculateIDF() {
	totalDocuments := float64(len(docs.Documents))
	docs.IDF = make(map[string]float64)

	termCounts := make(map[string]int)
	for _, doc := range docs.Documents {
		for term := range doc.TermFrequencies {
			termCounts[term]++
		}
	}

	for term, count := range termCounts {
		docs.IDF[term] = math.Log(totalDocuments / float64(count))
	}
}

func NewRagDocuments() RagDocuments {
	return RagDocuments{Documents: []*RagDocument{}, IDF: make(map[string]float64)}
}

// VecDense struct for vector operations
type VecDense struct {
	data []float64
}

// NewVecDense creates a new VecDense.
func NewVecDense(n int) *VecDense {
	return &VecDense{
		data: make([]float64, n),
	}
}

// CosineSimilarityVecDense calculates the cosine similarity between two VecDense vectors.
func CosineSimilarityVecDense(a, b *VecDense) float64 {
	if len(a.data) != len(b.data) {
		panic("vectors must have the same dimension")
	}

	var dotProduct float64
	var normA float64
	var normB float64

	for i := range a.data {
		dotProduct += a.data[i] * b.data[i]
		normA += a.data[i] * a.data[i]
		normB += b.data[i] * b.data[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

// Search function to find similar documents
func (docs RagDocuments) Search(commandVector []float64, command string, similarityThreshold float64) []RagDocument {
	// Remove preliminary filtering by directy considering all docs

	var relevantDocs []RagDocument
	for _, doc := range docs.Documents {
		commandVecDense := NewVecDense(len(commandVector))
		commandVecDense.data = commandVector
		similarity := CosineSimilarityVecDense(commandVecDense, &VecDense{data: doc.Embedding})

		// Apply the similarity threshold
		if similarity > 0.1 {
			relevantDocs = append(relevantDocs, *doc)
			// fmt.Println("doc.ID:", doc.ID, "similarity:", similarity, "len(relevantDocs):", len(relevantDocs))
		}
	}

	return reRankDocuments(relevantDocs, command, docs.IDF)
}

// reRankDocuments re-ranks documents based on similarity and keyword matching.
func reRankDocuments(docs []RagDocument, command string, globalIDF map[string]float64) []RagDocument {
	if len(docs) < 1 {
		return docs
	}

	type scoredDoc struct {
		doc   RagDocument
		score float64
	}

	var scoredDocs []scoredDoc
	for _, doc := range docs {
		docVecDense := NewVecDense(len(doc.Embedding))
		docVecDense.data = doc.Embedding
		similarity := CosineSimilarityVecDense(NewVecDense(len(docs[0].Embedding)), docVecDense)

		// Calculate TF-IDF weighted keyword score
		keywordScore := countKeywords(doc.Content, command, doc.TermFrequencies, globalIDF)
		// Combine similarity and keyword score for the re-ranking score
		score := keywordScore + similarity
		scoredDocs = append(scoredDocs, scoredDoc{doc, float64(score)})
	}

	// Sort documents by score (descending)
	sort.Slice(scoredDocs, func(i, j int) bool {
		return scoredDocs[i].score > scoredDocs[j].score
	})

	// Extract re-ranked documents
	reRankedDocs := make([]RagDocument, len(scoredDocs))
	for i, scoredDoc := range scoredDocs {
		reRankedDocs[i] = scoredDoc.doc
	}
	return reRankedDocs
}

// countKeywords counts how many words from the command are present in the content.
func countKeywords(content, command string, docTF map[string]float64, globalIDF map[string]float64) float64 {
	commandWords := strings.Fields(strings.ToLower(command))
	var totalScore float64

	for _, word := range commandWords {
		// Use TF-IDF to score the importance of the keyword
		tf := docTF[word]      // Term Frequency in the current document
		idf := globalIDF[word] // Inverse Document Frequency across all documents
		totalScore += tf * idf
	}
	return totalScore
}

// calculateTF calculates the term frequency for each word in the document.
func calculateTF(content string) map[string]float64 {
	tf := make(map[string]float64)
	words := strings.Fields(strings.ToLower(content))
	totalWords := float64(len(words))

	for _, word := range words {
		tf[word]++
	}

	for word, count := range tf {
		tf[word] = count / totalWords
	}

	return tf
}

// ReadRagDocuments reads RagDocuments from a file, either JSON or plain text.
func ReadRagDocuments(filename string, sw2v *word2vec.SimpleWord2Vec) (RagDocuments, error) {
	if strings.HasSuffix(filename, ".txt") || strings.HasSuffix(filename, ".md") {
		return ReadPlainTextDocuments(filename, sw2v)
	}
	return ReadPlainTextDocuments(filename, sw2v)
}

// ReadPlainTextDocuments reads a plain text file and creates RagDocuments for each paragraph.
func ReadPlainTextDocuments(filename string, sw2v *word2vec.SimpleWord2Vec) (RagDocuments, error) {
	file, err := os.Open(filename)
	if err != nil {
		fmt.Println("error opening file: %w", err)
	}
	defer file.Close()

	docs := NewRagDocuments()

	scanner := bufio.NewScanner(file)
	var paragraph strings.Builder
	for scanner.Scan() {
		log.Println("Processing line in file:", filename)
		line := scanner.Text()
		if line == "" {

			if paragraph.Len() > 0 {
				doc := &RagDocument{
					ID:              fmt.Sprintf("paragraph-%d", len(docs.Documents)+1),
					Content:         paragraph.String(),
					TermFrequencies: calculateTF(paragraph.String()),
				}
				docs.Documents = append(docs.Documents, doc)

				embedding, err := embedParagraph(paragraph.String(), sw2v)
				if err != nil {
					return docs, fmt.Errorf("error embedding paragraph: %w", err)
				}
				log.Printf("Embedding for paragraph: %s", paragraph.String())
				log.Printf("Embedding result: %v", embedding)
				doc.Embedding = embedding
				paragraph.Reset()

			}

		} else {
			paragraph.WriteString(line)
			paragraph.WriteString(" ")
		}
	}

	if paragraph.Len() > 0 {
		doc := &RagDocument{
			ID:              fmt.Sprintf("paragraph-%d", len(docs.Documents)+1),
			Content:         paragraph.String(),
			TermFrequencies: calculateTF(paragraph.String()),
		}
		docs.Documents = append(docs.Documents, doc)

		embedding, err := embedParagraph(paragraph.String(), sw2v)
		if err != nil {
			return docs, fmt.Errorf("error embedding paragraph: %w", err)
		}
		doc.Embedding = embedding
		paragraph.Reset()

	}

	return docs, scanner.Err()

}

// embedParagraph embeds the paragraph using the Word2Vec model.
func embedParagraph(paragraph string, sw2v *word2vec.SimpleWord2Vec) ([]float64, error) {
	words := strings.Fields(paragraph)
	var filteredWords []string
	for _, word := range words {
		if !stopWords[strings.ToLower(word)] && len(word) > 0 {
			filteredWords = append(filteredWords, word)
		}
	}
	if len(filteredWords) == 0 {
		return nil, fmt.Errorf("no embeddings found for paragraph after stop words")
	}
	var embeddings [][]float64
	for _, word := range filteredWords {
		word = strings.TrimSpace(word)
		if word == "" {
			continue
		}
		vocabIndex, ok := sw2v.Vocabulary[word]
		if !ok {
			continue
		}
		if vector, ok := sw2v.WordVectors[vocabIndex]; ok {
			embeddings = append(embeddings, vector)
		} else {
			log.Printf("Word '%s' not found in vocabulary.", word)
		}

	}

	if len(embeddings) == 0 { // If no words in the paragraph are in the vocabulary
		log.Println("No embeddings found for any words in paragraph. Returning zero vector.")
		return make([]float64, sw2v.VectorSize), nil // Return zero vector
	}

	paragraphVector := make([]float64, sw2v.VectorSize)

	for _, vec := range embeddings {
		for i, val := range vec {
			paragraphVector[i] += val
		}
	}
	for i := range paragraphVector {
		paragraphVector[i] /= float64(len(embeddings))
	}
	return paragraphVector, nil
}

// Helper function to generate random embeddings (as you mentioned)
func GenerateRandomEmbedding(size int) []float64 {
	embedding := make([]float64, size)
	for i := range embedding {
		embedding[i] = rand.Float64()
	}
	return embedding
}

// G struct
type G struct {
	layers int
	size   int
	input  []float64
	output []float64
}

// NewG creates a new G instance.
func NewG(layers, size int) *G {
	return &G{
		layers: layers,
		size:   size,
		input:  make([]float64, size),
		output: make([]float64, size),
	}
}
