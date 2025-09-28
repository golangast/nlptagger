package main

import (
	"flag"
	"fmt"
	"log"
	"math" // Keep math for softmax
	"math/rand"
	"sort"

	moemodel "nlptagger/neural/moe/model" // Keep moemodel
	mainvocab "nlptagger/neural/nnu/vocab"
	"nlptagger/neural/tensor"
	"nlptagger/neural/tokenizer"
	"nlptagger/tagger/nertagger"
	"nlptagger/tagger/postagger" // Added this line
	"nlptagger/tagger/tag"
)

type Prediction struct {
	TokenID    int
	Word       string
	Confidence float64
}

func getTopNPredictions(probabilities []float64, vocab []string, n int) []Prediction {
	predictions := make([]Prediction, 0, len(probabilities))
	for i, p := range probabilities {
		if i < 2 { // Skip <pad> and UNK
			continue
		}
		if i < len(vocab) {
			word := vocab[i]
			predictions = append(predictions, Prediction{
				TokenID:    i,
				Word:       word,
				Confidence: p * 100.0,
			})
		}
	}

	// Sort predictions by confidence
	sort.Slice(predictions, func(i, j int) bool {
		return predictions[i].Confidence > predictions[j].Confidence
	})

	if len(predictions) < n {
		return predictions
	}
	return predictions[:n]
}

var (
	query        = flag.String("query", "", "Query for MoE inference")
	maxSeqLength = flag.Int("maxlen", 32, "Maximum sequence length")
)

func main() {
	rand.Seed(1) // Seed the random number generator for deterministic behavior
	flag.Parse()

	if *query == "" {
		log.Fatal("Please provide a query using the -query flag.")
	}

	// Define paths
	const vocabPath = "gob_models/query_vocabulary.gob"
	const moeModelPath = "gob_models/moe_classification_model.gob"
	const childIntentVocabPath = "gob_models/child_intent_vocabulary.gob"
	const parentIntentVocabPath = "gob_models/parent_intent_vocabulary.gob"

	// Setup input vocabulary
	vocabulary, err := mainvocab.LoadVocabulary(vocabPath)
	if err != nil {
		log.Fatalf("Failed to set up input vocabulary: %v", err)
	}

	// Setup child intent vocabulary
	childIntentVocabulary, err := mainvocab.LoadVocabulary(childIntentVocabPath)
	if err != nil {
		log.Fatalf("Failed to set up child intent vocabulary: %v", err)
	}

	// Setup parent intent vocabulary
	parentIntentVocabulary, err := mainvocab.LoadVocabulary(parentIntentVocabPath)
	if err != nil {
		log.Fatalf("Failed to set up parent intent vocabulary: %v", err)
	}

	// Create tokenizer
	tokenizer, err := tokenizer.NewTokenizer(vocabulary)
	if err != nil {
		log.Fatalf("Failed to create tokenizer: %w", err)
	}

	// Load the trained MoEClassificationModel
	model, err := moemodel.LoadMoEClassificationModelFromGOB(moeModelPath, vocabulary.Size(), parentIntentVocabulary.Size(), childIntentVocabulary.Size(), *maxSeqLength)
	if err != nil {
		log.Fatalf("Failed to load MoE model: %v", err)
	}

	model.SetMode(false) // Set the model to evaluation mode

	if model.BertModel != nil {
		if model.BertModel.BertEmbeddings != nil {
		} else {
			fmt.Println("BertEmbeddings is nil")
		}
	} else {
		fmt.Println("BertModel is nil")
	}

	model.Description() // Print the model description

	log.Printf("---\n--- DEBUG: Child Intent Vocabulary (TokenToWord): %v ---", childIntentVocabulary.TokenToWord)
	log.Printf("---\n--- DEBUG: Parent Intent Vocabulary (TokenToWord): %v ---", parentIntentVocabulary.TokenToWord)

	log.Printf("Running MoE inference for query: \"%s\"", *query)

	// Encode the query

	tokenIDs, err := tokenizer.Encode(*query)
	if err != nil {
		log.Fatalf("Failed to encode query: %v", err)
	}

	// Get the actual vocabulary size of the loaded model's embedding layer
	// This is the true number of embeddings available in the loaded model.
	modelEmbeddingVocabSize := model.BertModel.BertEmbeddings.WordEmbeddings.Weight.Shape[0]
	unkTokenID := vocabulary.WordToToken["UNK"]

	// Map out-of-vocabulary tokens to UNK token ID
	for i, id := range tokenIDs {
		if id >= modelEmbeddingVocabSize { // Use the actual embedding layer size
			tokenIDs[i] = unkTokenID
		}
	}

	// POS Tagging and NER Tagging
	// POS Tagging
	// The Postagger function takes a string and returns a tag.Tag
	// It internally tokenizes the string, so we pass the original query.
	var t tag.Tag = postagger.Postagger(*query) // Call Postagger with the query string

	// NER Tagging
	// Nertagger expects a tag.Tag with Tokens and PosTag already populated.
	// We use the 't' returned by Postagger.
	t = nertagger.Nertagger(t)
	t = nertagger.NerNounCheck(t)
	t = nertagger.NerVerbCheck(t)

	nerTagIDsData := make([]float64, len(t.NerTag))
	nerTagToID := nertagger.NerTagToIDMap()
	for i, nerTag := range t.NerTag {
		if id, ok := nerTagToID[nerTag]; ok {
			nerTagIDsData[i] = float64(id)
		} else {
			nerTagIDsData[i] = 0.0 // Default to 0 if tag not in map
		}
	}

	// Pad or truncate the sequence to a fixed length
	originalTokenIDs := make([]int, len(tokenIDs))
	copy(originalTokenIDs, tokenIDs) // Keep a copy of tokenIDs before padding/truncating

	if len(tokenIDs) > *maxSeqLength {
		tokenIDs = tokenIDs[:*maxSeqLength] // Truncate from the end
	} else {
		for len(tokenIDs) < *maxSeqLength {
			tokenIDs = append(tokenIDs, vocabulary.PaddingTokenID) // Appends padding
		}
	}
	inputData := make([]float64, len(tokenIDs))
	for i, id := range tokenIDs {
		inputData[i] = float64(id)
	}
	inputTensor := tensor.NewTensor([]int{1, len(inputData)}, inputData, false) // RequiresGrad=false for inference

	// --- Start of new logic for tokenTypeIDs ---

	tokenTypeData := make([]float64, len(tokenIDs))

	// Create a set of intent words for efficient lookup
	intentWords := make(map[string]struct{})
	for _, word := range parentIntentVocabulary.TokenToWord {
		intentWords[word] = struct{}{}
	}
	for _, word := range childIntentVocabulary.TokenToWord {
		intentWords[word] = struct{}{}
	}

	// Iterate over the original token IDs to get individual words
	for i, id := range originalTokenIDs {
		word := vocabulary.GetWord(id) // Get the word for the token ID
		if _, ok := intentWords[word]; ok {
			if i < len(tokenTypeData) {
				tokenTypeData[i] = 1.0 // Mark as important
			}
		}
	}
	// Pad or truncate tokenTypeData to maxSeqLength
	if len(tokenTypeData) > *maxSeqLength {
		tokenTypeData = tokenTypeData[:*maxSeqLength]
	} else {
		for len(tokenTypeData) < *maxSeqLength {
			tokenTypeData = append(tokenTypeData, 0.0) // Pad with 0s
		}
	}

	tokenTypeTensor := tensor.NewTensor([]int{1, len(tokenIDs)}, make([]float64, len(tokenIDs)), false)
	// --- End of new logic for tokenTypeIDs ---

	// Dummy posTagIDs and nerTagIDs (as they are not used in this context)
	posTagIDs := tensor.NewTensor([]int{1, len(tokenIDs)}, make([]float64, len(tokenIDs)), false)

	if len(nerTagIDsData) > *maxSeqLength {
		nerTagIDsData = nerTagIDsData[:*maxSeqLength]
	} else {
		for len(nerTagIDsData) < *maxSeqLength {
			nerTagIDsData = append(nerTagIDsData, 0.0)
		}
	}
	nerTagIDs := tensor.NewTensor([]int{1, len(nerTagIDsData)}, nerTagIDsData, false)

	// Forward pass
	parentLogits, childLogits, err := model.Forward(inputTensor, tokenTypeTensor, posTagIDs, nerTagIDs)
	if err != nil {
		log.Fatalf("MoE model forward pass failed: %v", err)
	}

	// Interpret parent intent output
	parentProbabilities := softmax(parentLogits.Data)
	topParentPredictions := getTopNPredictions(parentProbabilities, parentIntentVocabulary.TokenToWord, 3)

	fmt.Println("--- Top 3 Parent Intent Predictions ---")
	for _, p := range topParentPredictions {
		importance := ""
		if p.Confidence > 50.0 {
			importance = " (Important)"
		}
		fmt.Printf("  - Word: %-20s (Confidence: %.2f%%)%s\n", p.Word, p.Confidence, importance)
	}
	fmt.Println("------------------------------------")

	// Interpret child intent output
	childProbabilities := softmax(childLogits.Data)
	topChildPredictions := getTopNPredictions(childProbabilities, childIntentVocabulary.TokenToWord, 3)

	fmt.Println("--- Top 3 Child Intent Predictions ---")
	for _, p := range topChildPredictions {
		importance := ""
		if p.Confidence > 50.0 {
			importance = " (Important)"
		}
		fmt.Printf("  - Word: %-20s (Confidence: %.2f%%)%s\n", p.Word, p.Confidence, importance)
	}
	fmt.Println("-----------------------------------")

	if len(topParentPredictions) > 0 && len(topChildPredictions) > 0 {
		predictedParentWord := topParentPredictions[0].Word
		predictedChildWord := topChildPredictions[0].Word
		fmt.Printf("\nDescription: The model's top prediction is an action related to %s, specifically to %s.\n", predictedParentWord, predictedChildWord)
	}
}

// softmax applies the softmax function to a slice of float64.
func softmax(logits []float64) []float64 {
	if len(logits) == 0 {
		return []float64{}
	}
	maxLogit := logits[0]
	for _, logit := range logits {
		if logit > maxLogit {
			maxLogit = logit
		}
	}
	expSum := 0.0
	for _, logit := range logits {
		expSum += math.Exp(logit - maxLogit)
	}

	probabilities := make([]float64, len(logits))
	for i, logit := range logits {
		probabilities[i] = math.Exp(logit-maxLogit) / expSum
	}
	return probabilities
}
