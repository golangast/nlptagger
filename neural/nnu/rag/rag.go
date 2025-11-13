// Package rag provides functions for Retrieval Augmented Generation (RAG).

package rag

import (
	"bufio"
	"encoding/gob"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"

	"github.com/zendrulat/nlptagger/neural/nnu"
	"github.com/zendrulat/nlptagger/neural/nnu/word2vec"
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
	var relevantDocs []RagDocument
	for _, doc := range docs.Documents {
		commandVecDense := NewVecDense(len(commandVector))
		commandVecDense.data = commandVector
		similarity := CosineSimilarityVecDense(commandVecDense, &VecDense{data: doc.Embedding}) // Use doc.Embedding directly here

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

const trainingDataPath = ".././trainingdata/ragdata/rag_data.json"

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

// averageEmbeddings calculates the average of a slice of embeddings.
func averageEmbeddings(embeddings [][]float64, vectorSize int) []float64 {
	averagedEmbedding := make([]float64, vectorSize)
	if len(embeddings) == 0 {
		return averagedEmbedding // Return zero vector if no embeddings
	}

	for _, embedding := range embeddings {
		for i, val := range embedding {
			averagedEmbedding[i] += val
		}
	}

	// Divide by the number of embeddings to get the average
	for i := range averagedEmbedding {
		averagedEmbedding[i] /= float64(len(embeddings))
	}

	return averagedEmbedding
}

// generateQueryEmbedding generates an embedding for a query string
// using the provided SimpleWord2Vec model.
func GenerateQueryEmbedding(query string, sw2v *word2vec.SimpleWord2Vec) ([]float64, error) {
	words := strings.Fields(strings.ToLower(query))
	var filteredWords []string
	for _, word := range words {
		if !stopWords[word] && len(word) > 0 {
			filteredWords = append(filteredWords, word)
		}
	}

	if len(filteredWords) == 0 {
		return make([]float64, sw2v.VectorSize), nil // Return zero vector if no words
	}

	var embeddings [][]float64
	for _, word := range filteredWords {
		if vocabIndex, ok := sw2v.Vocabulary[word]; ok {
			if vector, ok := sw2v.WordVectors[vocabIndex]; ok {
				embeddings = append(embeddings, vector)
			}
		}
	}

	if len(embeddings) == 0 {
		return make([]float64, sw2v.VectorSize), nil // Return zero vector if no words found in vocabulary
	}
	return averageEmbeddings(embeddings, sw2v.VectorSize), nil
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

// SaveRagModelToGOB saves a SimpleNN model associated with RAG to a GOB file.
func SaveRagModelToGOB(model *nnu.SimpleNN, filePath string) error {
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("error creating gob file: %w", err)
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	err = encoder.Encode(model)
	if err != nil {
		return fmt.Errorf("error encoding model to gob: %w", err)
	}

	return nil
}

// LoadRagModelFromGOB loads a SimpleNN model associated with RAG from a GOB file.
func LoadRagModelFromGOB(filePath string) (*nnu.SimpleNN, error) {

	if _, err := os.Stat(filePath); err == nil {
		file, err := os.Open(filePath)
		if err != nil {
			return nil, fmt.Errorf("error opening gob file: %w", err)
		}
		defer file.Close()

		decoder := gob.NewDecoder(file)
		var model nnu.SimpleNN
		err = decoder.Decode(&model)
		if err != nil {
			return nil, fmt.Errorf("error decoding model from gob: %w", err)
		}

		return &model, nil
	} else if os.IsNotExist(err) {
		loadedNN := nnu.NewSimpleNN("trainingdata/tagdata/nlp_training_data.json")
		err = SaveRagModelToGOB(loadedNN, filePath)
		if err != nil {
			fmt.Println("Error saving trained RAG model: ", err)
		}
		return loadedNN, nil

	} else {
		fmt.Println("Error checking file:", err)
	}
	return nil, nil

}

// RagForwardPass performs a forward pass through the neural network,
// augmented with RAG. It takes a SimpleNN model, RagDocuments,
// and a query embedding as input.
func RagForwardPass(nn *nnu.SimpleNN, docs RagDocuments, queryEmbedding []float64) ([]float64, error) {
	// The 'docs' parameter is a RagDocuments struct, which is expected // Updated comment
	// to have its 'Documents' field populated with a slice of RagDocument.

	// 1. Search for relevant documents using the query embedding
	relevantDocs := docs.Search(queryEmbedding, "", 0.1)

	averageDocEmbedding := make([]float64, len(queryEmbedding))

	for _, doc := range relevantDocs {
		// 2. Pool relevant document embeddings (e.g., by averaging)
		// This creates a fixed-size representation of the retrieved documents.
		if len(doc.Embedding) != len(queryEmbedding) {
			// Handle embedding size mismatch if necessary, e.g., log a warning
			log.Printf("Warning: Document embedding size (%d) does not match query embedding size (%d). Document ID: %s", len(doc.Embedding), len(queryEmbedding), doc.ID)
		}
		for i := range averageDocEmbedding {
			averageDocEmbedding[i] += doc.Embedding[i]
		}
	}
	// Calculate the average if there are relevant documents
	if len(relevantDocs) > 0 {
		for i := range averageDocEmbedding {
			averageDocEmbedding[i] /= float64(len(relevantDocs))
		}
	} else {
		fmt.Println("Debug: RagForwardPass - No relevant documents found, averageDocEmbedding remains zero.")
	}

	// 3. Combine query embedding and the pooled document embedding
	combinedInput := append(queryEmbedding, averageDocEmbedding...)

	// 4. Check if the combined input size matches the neural network's input size.
	// This is crucial because the input layer of the neural network expects a fixed size.
	if len(combinedInput) != nn.InputSize {
		// If this error occurs after implementing pooling, it indicates an issue
		// with how the combined input size was calculated or how the neural network
		// was initialized.
		return nil, fmt.Errorf("combined input size (%d) does not match neural network input size (%d) after pooling", len(combinedInput), nn.InputSize)
	}

	// Call the existing forward pass logic from nnu.go
	// Assuming SimpleNN has a method like CalculateOutputs() or similar
	// You might need to adapt the SimpleNN struct or add a method to it
	// to expose the internal forward pass steps.
	nn.Inputs = combinedInput

	// For now, let's directly use the calculation logic based on our knowledge
	// of SimpleNN structure, assuming the weights and biases are initialized.
	// Calculate hidden layer outputs
	// For now, let's directly use the calculation logic based on our knowledge
	// of SimpleNN structure, assuming the weights and biases are initialized.
	// Calculate hidden layer outputs
	hiddenOutputs := make([]float64, nn.HiddenSize)
	for i := 0; i < nn.HiddenSize; i++ {
		// Add checks for nn.WeightsIH and nn.Inputs before accessing
		if nn.WeightsIH == nil || i >= len(nn.WeightsIH) || len(nn.WeightsIH[i]) != nn.InputSize {
			return nil, fmt.Errorf("nn.WeightsIH is nil or has incorrect dimensions at row %d. Expected column size %d, got %d", i, nn.InputSize, len(nn.WeightsIH[i]))
		}
		if nn.Inputs == nil || len(nn.Inputs) != nn.InputSize {
			return nil, fmt.Errorf("nn.Inputs is nil or has incorrect length (%d), expected (%d)", len(nn.Inputs), nn.InputSize)
		}

		sum := 0.0
		for j := 0; j < nn.InputSize; j++ {
			// Make sure nn.Inputs is set with combinedInput before this loop
			sum += nn.WeightsIH[i][j] * nn.Inputs[j]
		}
		// Assuming a sigmoid activation function as in nnu.go
		// Ensure nn.Sigmoid method exists and is accessible
		hiddenOutputs[i] = 1.0 / (1.0 + math.Exp(-sum))
	}

	nn.HiddenOutputs = hiddenOutputs

	// Calculate output layer outputs
	outputOutputs := make([]float64, nn.OutputSize)
	for i := 0; i < nn.OutputSize; i++ {
		sum := 0.0
		// Add checks for nn.WeightsHO and hiddenOutputs before accessing
		if nn.WeightsHO == nil || i >= len(nn.WeightsHO) || len(nn.WeightsHO[i]) != nn.HiddenSize {
			return nil, fmt.Errorf("nn.WeightsHO is nil or has incorrect dimensions at row %d. Expected column size %d, got %d", i, nn.HiddenSize, len(nn.WeightsHO[i]))
		}
		if 0 > len(nn.HiddenOutputs) || len(nn.HiddenOutputs) != nn.HiddenSize {
			return nil, fmt.Errorf("hiddenOutputs is nil or has incorrect length (%d), expected (%d)", len(hiddenOutputs), nn.HiddenSize)
		}
		for j := 0; j < nn.HiddenSize; j++ {
			sum += nn.WeightsHO[i][j] * hiddenOutputs[j]
		}
		// Assuming a sigmoid activation function for the output layer as well
		outputOutputs[i] = 1.0 / (1.0 + math.Exp(-sum))
	}

	nn.Outputs = outputOutputs // Update the network's output field
	return outputOutputs, nil
}

// RagTraining trains the SimpleNN model with RAG.
// It takes the neural network model, RagDocuments, training data (query embeddings and target outputs),
// and training parameters (learning rate, epochs) as input.
func RagTraining(nn *nnu.SimpleNN, docs RagDocuments, trainingData map[string][]float64, epochs int, sw2v *word2vec.SimpleWord2Vec, similarityThreshold float64) error { // Added similarityThreshold
	// Initialize training process
	// Assuming trainingData is a map where keys are query strings
	// and values are the corresponding target output vectors (e.g., document embeddings).
	// You will need to generate query embeddings for the queries in your training data.

	// The `sw2v` parameter is required to generate query embeddings.
	for epoch := 0; epoch < epochs; epoch++ {
		fmt.Println("Epoch:", epoch+1)
		for queryString, targetOutput := range trainingData {
			// Generate query embedding
			queryEmbedding, err := GenerateQueryEmbedding(queryString, sw2v)
			if err != nil {
				log.Printf("Error generating query embedding for '%s': %v", queryString, err)
				continue // Skip this training example if embedding generation fails
			}

			// Perform RAG-augmented forward pass
			predictedOutput, err := RagForwardPass(nn, docs, queryEmbedding) // Removed hardcoded threshold
			if err != nil {
				fmt.Println("Error during RAG forward pass for query ", queryString, err)
				continue // Skip this training example if forward pass fails
			}

			// --- Backpropagation ---

			outputErrors := make([]float64, nn.OutputSize)
			for i := 0; i < nn.OutputSize; i++ {
				outputErrors[i] = targetOutput[i] - predictedOutput[i]
			}

			// Calculate output layer gradients
			// Assuming sigmoid activation for the output layer
			outputGradients := make([]float64, nn.OutputSize)
			for i := 0; i < nn.OutputSize; i++ {
				outputGradients[i] = outputErrors[i] * predictedOutput[i] * (1 - predictedOutput[i])
			}

			// Calculate hidden layer errors (error backpropagated from the output layer)
			hiddenErrors := make([]float64, nn.HiddenSize)
			for i := 0; i < nn.HiddenSize; i++ {
				sum := 0.0
				for j := 0; j < nn.OutputSize; j++ {
					sum += nn.WeightsHO[j][i] * outputErrors[j]
				}
				hiddenErrors[i] = sum
			}

			// Calculate hidden layer gradients (based on hidden error and derivative of activation)
			// Assuming sigmoid activation for the hidden layer
			hiddenGradients := make([]float64, nn.HiddenSize)
			for i := 0; i < nn.HiddenSize; i++ {
				// We need the hidden layer outputs from the forward pass for the derivative
				// Assuming you stored them in the SimpleNN struct
				hiddenOutput := nn.HiddenOutputs[i] // Assuming you added HiddenOutputs field to SimpleNN
				hiddenGradients[i] = hiddenErrors[i] * hiddenOutput * (1 - hiddenOutput)
			}

			// Update output layer weights (WeightsHO) using an iterator
			outputWeightIterator := nn.NewOutputWeightIterator()
			for outputWeightIterator.Next() {
				i, j, weight := outputWeightIterator.Current()
				// Calculate gradient for this specific weight
				gradient := outputGradients[i] * nn.HiddenOutputs[j] // Using calculated gradients and hidden outputs
				outputWeightIterator.Update(weight + nn.LearningRate*gradient)
			}

			// Update hidden layer weights (WeightsIH) using an iterator
			hiddenWeightIterator := nn.NewHiddenWeightIterator()
			for hiddenWeightIterator.Next() {
				i, j, weight := hiddenWeightIterator.Current()
				// Calculate gradient for this specific weight
				gradient := hiddenGradients[i] * nn.Inputs[j] // Using calculated gradients and inputs
				hiddenWeightIterator.Update(weight + nn.LearningRate*gradient)
			}

			// Update biases using iterators
			outputBiasIterator := nn.NewOutputBiasIterator()
			for outputBiasIterator.Next() {
				i, bias := outputBiasIterator.Current()
				// Calculate gradient for this specific bias (same as output gradient)
				gradient := outputGradients[i]
				outputBiasIterator.Update(bias + nn.LearningRate*gradient)
			}

			hiddenBiasIterator := nn.NewHiddenBiasIterator()
			for hiddenBiasIterator.Next() {
				i, bias := hiddenBiasIterator.Current()
				// Calculate gradient for this specific bias (same as hidden gradient)
				gradient := hiddenGradients[i]
				hiddenBiasIterator.Update(bias + nn.LearningRate*gradient)
			}
		}
	}
	return nil
}

// generateExampleTrainingData creates some example training data for RAG training.
// In a real application, you would generate this data based on your specific task and dataset.
func GenerateExampleTrainingData(docs []RagDocument, sw2v *word2vec.SimpleWord2Vec) (map[string][]float64, error) {
	trainingData := make(map[string][]float64)

	// Read the training data from the JSON file
	file, err := os.Open(trainingDataPath)
	if err != nil {
		return nil, fmt.Errorf("error opening training data file: %w", err)
	}
	defer file.Close()

	decoder := json.NewDecoder(file)
	var rawData []struct {
		ID      string `json:"ID"`
		Content string `json:"Content"`
	}
	if err := decoder.Decode(&rawData); err != nil {
		return nil, fmt.Errorf("error decoding training data JSON: %w", err)
	}

	// Map content (queries) to document indices
	for i, dataEntry := range rawData {
		query := dataEntry.Content

		// Generate embedding for the query
		_, err := GenerateQueryEmbedding(query, sw2v)
		if err != nil {
			log.Printf("Error generating embedding for query '%s': %v", query, err)
			continue
		}
		var targetOutput []float64

		// The target output is the embedding of the document at the current index
		if i < len(docs) {
			targetOutput = docs[i].Embedding
		} else {
			log.Printf("Document index %d out of bounds for docs slice (length %d) for query '%s'", i, len(docs), query)
			continue
		}

		// Add to training data
		trainingData[query] = targetOutput
	}

	return trainingData, nil
}
