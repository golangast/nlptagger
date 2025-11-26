package answer

import (
	"errors"
	"fmt"
	"log"
	"math"

	"github.com/zendrulat/nlptagger/neural/nnu"
	"github.com/zendrulat/nlptagger/neural/nnu/rag"
	"github.com/zendrulat/nlptagger/neural/nnu/word2vec"
)

// AnswerQuestion takes a user query, a trained RAG model, documents, and a Word2Vec model,
// and returns an answer using the RAG system.
func AnswerQuestion(userQuery string, trainedNN *nnu.SimpleNN, ragDocs rag.RagDocuments, sw2v *word2vec.SimpleWord2Vec) (string, error) {
	// Generate query embedding
	queryEmbedding, err := rag.GenerateQueryEmbedding(userQuery, sw2v)
	if err != nil {
		return "", fmt.Errorf("error generating embedding for user query: %w", err)
	}

	// Perform RAG forward pass
	predictedOutput, err := rag.RagForwardPass(trainedNN, ragDocs, queryEmbedding)
	if err != nil {
		return "", fmt.Errorf("error during RAG forward pass: %w", err)
	}

	// Interpret the predictedOutput to get the answer string
	// YOU NEED TO CUSTOMIZE THIS FUNCTION based on your model's output
	answerText, answer, err := interpretOutputAsAnswer(predictedOutput, ragDocs, sw2v)
	if err != nil {
		return "", fmt.Errorf("error interpreting model output: %w", err)
	}

	fmt.Println("answerText: ", answerText)
	fmt.Println("answer: ", answer)

	return answerText, nil
}

// interpretOutputAsAnswer is a placeholder function.
// YOU MUST CUSTOMIZE THIS function based on how your RAG model's output should be interpreted.
// It takes the predictedOutput from the neural network and the RagDocuments
// (in case you need information from the retrieved documents) and returns the answer string.
func interpretOutputAsAnswer(predictedOutput []float64, docs rag.RagDocuments, sw2v *word2vec.SimpleWord2Vec) (string, string, error) {
	fmt.Println("predictedOutput: ", predictedOutput)

	// Check if the output is a single probability (for models that output only relevance/confidence)
	if len(predictedOutput) == 1 {
		probability := predictedOutput[0]
		fmt.Printf("Received single probability output: %f\n", probability)

		// Define a threshold for confidence
		confidenceThreshold := 0.5 // Example threshold, adjust as needed

		if probability > confidenceThreshold {
			// Use the relevant documents to form an answer
			// Assuming docs.Documents contains the retrieved documents
			// and they are ordered by relevance (most relevant first)
			relevantDocsContent := ""
			numDocsToInclude := 3 // Adjust this number as needed
			for i, doc := range docs.Documents {
				if i >= numDocsToInclude {
					break // Stop after including the top N documents
				}
				relevantDocsContent += doc.Content + "\n" // Concatenate document content
			}

			if relevantDocsContent != "" {
				// You might want to add more sophisticated logic here
				// to summarize or select parts of the content
				answer := "Based on the provided documents, a potential answer is: " + relevantDocsContent
				return answer, "", nil // Return the formulated answer
			} else {
				return "No relevant documents found to form an answer.", "", nil
			}
		} else {
			return "Confidence too low to provide a specific answer from the documents.", "", nil
		}
	}

	// If not a single probability, check for span extraction indices
	if len(predictedOutput) < 2 {
		return "", "", errors.New("predictedOutput does not contain enough indices for span extraction")
	}

	// Original span extraction logic (if you still need it for other cases)
	if len(predictedOutput) < 3 {
		return "", "", fmt.Errorf("predictedOutput does not contain enough indices for span extraction, expected at least 3 for span, got %d", len(predictedOutput))
	}

	// Assuming the indices refer to the first document in docs.Documents
	if len(docs.Documents) == 0 {
		return "", "", errors.New("no documents available to extract answer from")
	}

	probability := predictedOutput[0]
	// Assuming RagDocument has a Content field
	docContent := docs.Documents[0].Content

	// Convert float output to integer indices (adjust based on your model output type)
	startIndex := int(math.Round(predictedOutput[0]))
	endIndex := int(math.Round(predictedOutput[1]))

	// Basic index validation
	if startIndex < 0 || endIndex >= len(docContent) || startIndex > endIndex {
		// You might want to return a default answer or an error depending on your needs
		log.Printf("Warning: Invalid predicted indices: start=%d, end=%d for document length %d", startIndex, endIndex, len(docContent))
		return "Could not find a specific answer in the documents.", "", nil // Or return an error
	}
	answer := docContent[startIndex : endIndex+1] // Extract the span
	// Define your threshold

	vocab := sw2v.Vocabulary // Or wherever your vocabulary is

	if len(predictedOutput) != len(vocab) {
		return "", "", fmt.Errorf("predictedOutput size (%d) does not match vocabulary size (%d)", len(predictedOutput), len(vocab))
	}

	// Find the index with the highest probability
	maxProbIndex := 0
	maxProb := predictedOutput[0]
	for i := 1; i < len(predictedOutput); i++ {
		if predictedOutput[i] > maxProb {
			maxProb = predictedOutput[i]
			maxProbIndex = i
		}
	}

	// Find the word corresponding to the index in the vocabulary
	var answerWord string
	for word, index := range vocab {
		if index == maxProbIndex {
			answerWord = word
			break
		}
	}

	if answerWord == "" {
		return "", "", fmt.Errorf("could not find word for predicted index %d in vocabulary", maxProbIndex)
	}

	yesThreshold := 0.6

	if probability > yesThreshold {
		return "Yes", answerWord, nil
	} else {
		return "No", answerWord, nil
	}

	return answer, answerWord, nil

}
