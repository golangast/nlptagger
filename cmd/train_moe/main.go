package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"os/signal"
	"runtime/pprof" // Added for profiling
	"strings"
	"syscall"

	"nlptagger/neural/moe"
	. "nlptagger/neural/nn"
	mainvocab "nlptagger/neural/nnu/vocab"
	"nlptagger/neural/nnu/word2vec"
	. "nlptagger/neural/tensor"
	"nlptagger/neural/tokenizer"
)

// IntentTrainingExample represents a single training example for intent classification.
type IntentTrainingExample struct {
	Query        string `json:"query"`
	ParentIntent string `json:"parent_intent"`
	ChildIntent  string `json:"child_intent"`
	Description  string `json:"description"`
	Sentence     string `json:"sentence"`
}

// IntentTrainingData represents the structure of the intent training data JSON.
type IntentTrainingData []IntentTrainingExample

// LoadIntentTrainingData loads the intent training data from a JSON file.
func LoadIntentTrainingData(filePath string) (*IntentTrainingData, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open training data file %s: %w", filePath, err)
	}
	defer file.Close()

	bytes, err := io.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("failed to read training data file %s: %w", filePath, err)
	}

	var data IntentTrainingData
	err = json.Unmarshal(bytes, &data)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal training data JSON from %s: %w", filePath, err)
	}

	return &data, nil
}

// TokenizeAndConvertToIDs tokenizes a text and converts tokens to their corresponding IDs, handling padding/truncation.
func TokenizeAndConvertToIDs(text string, tokenizer *tokenizer.Tokenizer, vocabulary *mainvocab.Vocabulary, maxLen int) ([]int, error) {

	tokenIDs, err := tokenizer.Encode(text)
	if err != nil {
		return nil, fmt.Errorf("failed to encode text: %w", err)
	}

	// Handle empty input strings by returning a slice with only the PaddingTokenID
	if len(tokenIDs) == 0 && maxLen > 0 {
		tokenIDs = make([]int, maxLen)
		for i := range tokenIDs {
			tokenIDs[i] = vocabulary.PaddingTokenID
		}
		return tokenIDs, nil
	} else if len(tokenIDs) == 0 {
		return []int{vocabulary.PaddingTokenID}, nil
	}

	if maxLen > 0 {
		if len(tokenIDs) > maxLen {
			tokenIDs = tokenIDs[:maxLen]
		} else if len(tokenIDs) < maxLen {
			paddingSize := maxLen - len(tokenIDs)
			padding := make([]int, paddingSize)
			for i := range padding {
				padding[i] = vocabulary.PaddingTokenID
			}
			tokenIDs = append(tokenIDs, padding...)
		}
	}
	return tokenIDs, nil
}

// TrainIntentMoEModel trains the MoEClassificationModel.
func TrainIntentMoEModel(model *moe.IntentMoE, data *IntentTrainingData, epochs int, learningRate float64, batchSize int, queryVocab, parentIntentVocab, childIntentVocab, sentenceVocab *mainvocab.Vocabulary, queryTokenizer, sentenceTokenizer *tokenizer.Tokenizer, maxSequenceLength int, profileFile *os.File) error {

	if model == nil {
		return errors.New("cannot train a nil model")
	}
	if data == nil || len(*data) == 0 {
		return errors.New("no training data provided")
	}

	optimizer := NewOptimizer(model.Parameters(), learningRate, 1.0) // Using a clip value of 1.0

	for epoch := 0; epoch < epochs; epoch++ {
		log.Printf("Epoch %d/%d", epoch+1, epochs)
		totalLoss := 0.0
		numBatches := 0
		// Create batches for training
		for i := 0; i < len(*data); i += batchSize {
			end := i + batchSize
			if end > len(*data) {
				end = len(*data)
			}
			batch := (*data)[i:end]

			loss, err := trainIntentMoEBatch(model, optimizer, batch, queryVocab, parentIntentVocab, childIntentVocab, sentenceVocab, queryTokenizer, sentenceTokenizer, maxSequenceLength)
			if err != nil {
				log.Printf("Error training batch: %v", err)
				continue // Or handle error more strictly
			}
			totalLoss += loss
			numBatches++
		}
		if numBatches > 0 {
			log.Printf("Epoch %d, Average Loss: %f", epoch+1, totalLoss/float64(numBatches))
		}
	}

	return nil
}

// trainIntentMoEBatch performs a single training step on a batch of data.
func trainIntentMoEBatch(intentMoEModel *moe.IntentMoE, optimizer Optimizer, batch IntentTrainingData, queryVocab, parentIntentVocab, childIntentVocab, sentenceVocab *mainvocab.Vocabulary, queryTokenizer, sentenceTokenizer *tokenizer.Tokenizer, maxSequenceLength int) (float64, error) {
	optimizer.ZeroGrad()

	batchSize := len(batch)

	inputIDsBatch := make([]int, batchSize*maxSequenceLength)
	parentIntentIDsBatch := make([]int, batchSize)
	childIntentIDsBatch := make([]int, batchSize)
	sentenceIDsBatch := make([]int, batchSize*maxSequenceLength)

	for i, example := range batch {
		// Tokenize and pad query
		queryTokens, err := TokenizeAndConvertToIDs(example.Query, queryTokenizer, queryVocab, maxSequenceLength)
		if err != nil {
			return 0, fmt.Errorf("query tokenization failed for item %d: %w", i, err)
		}
		copy(inputIDsBatch[i*maxSequenceLength:(i+1)*maxSequenceLength], queryTokens)

		// Get parent and child intent IDs
		parentIntentIDsBatch[i] = parentIntentVocab.GetTokenID(example.ParentIntent)
		childIntentIDsBatch[i] = childIntentVocab.GetTokenID(example.ChildIntent)

		// Prepend BOS and append EOS to the sentence for training
		trainingSentence := "<s> " + example.Sentence + " </s>"
		// Tokenize and pad sentence
		sentenceTokens, err := TokenizeAndConvertToIDs(trainingSentence, sentenceTokenizer, sentenceVocab, maxSequenceLength)
		if err != nil {
			return 0, fmt.Errorf("sentence tokenization failed for item %d: %w", i, err)
		}
		copy(sentenceIDsBatch[i*maxSequenceLength:(i+1)*maxSequenceLength], sentenceTokens)
	}

	// Convert input IDs to a Tensor (embeddings will be handled by the model)
	inputTensor := NewTensor([]int{batchSize, maxSequenceLength}, convertIntsToFloat64s(inputIDsBatch), false)
	sentenceTensor := NewTensor([]int{batchSize, maxSequenceLength}, convertIntsToFloat64s(sentenceIDsBatch), false)

	// Forward pass through the IntentMoE model
	parentLogits, childLogits, sentenceLogits, contextVector, err := intentMoEModel.Forward(inputTensor, sentenceTensor)
	if err != nil {
		return 0, fmt.Errorf("IntentMoE model forward pass failed: %w", err)
	}

	// Calculate loss for the parent and child intents
	parentLoss, parentGrad := CrossEntropyLoss(parentLogits, parentIntentIDsBatch, parentIntentVocab.PaddingTokenID)
	childLoss, childGrad := CrossEntropyLoss(childLogits, childIntentIDsBatch, childIntentVocab.PaddingTokenID)

	// Calculate loss for the sentence
	sentenceLoss := 0.0
	var sentenceGrads []*Tensor
	for t := 0; t < maxSequenceLength; t++ {
		targets := make([]int, batchSize)
		for i := 0; i < batchSize; i++ {
			targets[i] = sentenceIDsBatch[i*maxSequenceLength+t]
		}
		loss, grad := CrossEntropyLoss(sentenceLogits[t], targets, sentenceVocab.PaddingTokenID)
		sentenceLoss += loss
		sentenceGrads = append(sentenceGrads, grad)
	}

	sentenceLossWeight := 0.1
	totalLoss := parentLoss + childLoss + sentenceLoss*sentenceLossWeight
	for _, grad := range sentenceGrads {
		grad.MulScalar(sentenceLossWeight)
	}
	allGrads := []*Tensor{parentGrad, childGrad}
	allGrads = append(allGrads, sentenceGrads...)

	// Backward pass
	err = intentMoEModel.Backward(allGrads...)
	if err != nil {
		return 0, fmt.Errorf("IntentMoE model backward pass failed: %w", err)
	}

	optimizer.Step()

	// Log guessed sentence
	guessedIDs, err := intentMoEModel.GreedySearchDecode(contextVector, maxSequenceLength, sentenceVocab.GetTokenID("<s>"), sentenceVocab.GetTokenID("</s>"))
	if err != nil {
		log.Printf("Error decoding guessed sentence: %v", err)
	} else {
		guessedSentence, err := sentenceTokenizer.Decode(guessedIDs)
		if err != nil {
			log.Printf("Error decoding guessed sentence: %v", err)
		} else {
			log.Printf("Guessed sentence: %s", guessedSentence)
		}
		log.Printf("Target sentence: %s", batch[0].Sentence)
	}

	return totalLoss, nil
}

func convertIntsToFloat64s(input []int) []float64 {
	output := make([]float64, len(input))
	for i, v := range input {
		output[i] = float64(v)
	}
	return output
}

func convertW2VVocab(w2vVocab map[string]int) *mainvocab.Vocabulary {
	vocab := mainvocab.NewVocabulary()
	vocab.WordToToken = w2vVocab
	maxID := 0
	for _, id := range w2vVocab {
		if id > maxID {
			maxID = id
		}
	}
	vocab.TokenToWord = make([]string, maxID+1)
	for token, id := range w2vVocab {
		vocab.TokenToWord[id] = token
	}
	return vocab
}

func BuildVocabularies(dataPath string) (*mainvocab.Vocabulary, *mainvocab.Vocabulary, *mainvocab.Vocabulary, *mainvocab.Vocabulary, error) {
	queryVocabulary := mainvocab.NewVocabulary()
	parentIntentVocabulary := mainvocab.NewVocabulary()
	childIntentVocabulary := mainvocab.NewVocabulary()
	sentenceVocabulary := mainvocab.NewVocabulary()

	intentTrainingData, err := LoadIntentTrainingData(dataPath)
	if err != nil {
		return nil, nil, nil, nil, fmt.Errorf("failed to load intent training data from %s: %w", dataPath, err)
	}

	for _, pair := range *intentTrainingData {
		// Use the same tokenizer logic as during inference to build the vocabulary
		tokenizedQuery := tokenizer.Tokenize(strings.ToLower(pair.Query))
		for _, word := range tokenizedQuery {
			queryVocabulary.AddToken(word)
		}
		parentIntentVocabulary.AddToken(pair.ParentIntent)
		childIntentVocabulary.AddToken(pair.ChildIntent)

		// Add BOS and EOS tokens to the sentence when building the vocabulary
		trainingSentence := "<s> " + pair.Sentence + " </s>"
		tokenizedSentence := tokenizer.Tokenize(strings.ToLower(trainingSentence))
		for _, word := range tokenizedSentence {
			sentenceVocabulary.AddToken(word)
		}
	}

	// Explicitly add BOS and EOS tokens to the sentence vocabulary
	sentenceVocabulary.BosID = sentenceVocabulary.GetTokenID("<s>")
	sentenceVocabulary.EosID = sentenceVocabulary.GetTokenID("</s>")

	return queryVocabulary, parentIntentVocabulary, childIntentVocabulary, sentenceVocabulary, nil
}

func main() {
	const intentTrainingDataPath = "./trainingdata/intent_data.json"
	const word2vecModelPath = "gob_models/word2vec_model.gob"

	// Start CPU profiling
	f, err := os.Create("cpu.prof")
	if err != nil {
		log.Fatal("could not create CPU profile: ", err)
	}
	if err := pprof.StartCPUProfile(f); err != nil {
		log.Fatal("could not start CPU profile: ", err)
	}

	// Set up a channel to listen for interrupt signals
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Goroutine to handle graceful shutdown on signal
	go func() {
		<-sigChan // Block until a signal is received
		log.Println("Received interrupt signal. Stopping CPU profile and closing file.")
		pprof.StopCPUProfile()
		f.Close()
		os.Exit(0) // Exit gracefully
	}()

	// Define training parameters
	epochs := 10
	learningRate := 0.001
	batchSize := 16
	parentIntentVocabularySavePath := "gob_models/parent_intent_vocabulary.gob"
	childIntentVocabularySavePath := "gob_models/child_intent_vocabulary.gob"
	sentenceVocabularySavePath := "gob_models/sentence_vocabulary.gob"

	// Load Word2Vec model
	word2vecModel, err := word2vec.LoadModel(word2vecModelPath)
	if err != nil {
		log.Fatalf("Failed to load Word2Vec model: %v", err)
	}

	// Create query vocabulary from word2vec model
	queryVocabulary := convertW2VVocab(word2vecModel.Vocabulary)

	// Try to load other vocabularies first
	parentIntentVocabulary, err := mainvocab.LoadVocabulary(parentIntentVocabularySavePath)
	if err != nil {
		log.Println("Failed to load parent intent vocabulary, creating a new one.")
	}
	childIntentVocabulary, err := mainvocab.LoadVocabulary(childIntentVocabularySavePath)
	if err != nil {
		log.Println("Failed to load child intent vocabulary, creating a new one.")
	}
	sentenceVocabulary, err := mainvocab.LoadVocabulary(sentenceVocabularySavePath)
	if err != nil {
		log.Println("Failed to load sentence vocabulary, creating a new one.")
	}

	if parentIntentVocabulary == nil || childIntentVocabulary == nil || sentenceVocabulary == nil {
		log.Println("Building vocabularies from scratch...")
		_, parentIntentVocabulary, childIntentVocabulary, sentenceVocabulary, err = BuildVocabularies(intentTrainingDataPath)
		if err != nil {
			log.Fatalf("Failed to build vocabularies: %v", err)
		}
	}

	log.Printf("Query Vocabulary (after load/create): Size=%d", len(queryVocabulary.WordToToken))
	log.Printf("Parent Intent Vocabulary (after load/create): Size=%d", len(parentIntentVocabulary.WordToToken))
	log.Printf("Child Intent Vocabulary (after load/create): Size=%d", len(childIntentVocabulary.WordToToken))
	log.Printf("Sentence Vocabulary (after load/create): Size=%d", len(sentenceVocabulary.WordToToken))

	// Load Intent training data
	intentTrainingData, err := LoadIntentTrainingData(intentTrainingDataPath)
	if err != nil {
		log.Fatalf("Failed to load intent training data from %s: %v", intentTrainingDataPath, err)
	}
	log.Printf("Loaded %d training examples.", len(*intentTrainingData))

	// After vocabularies are fully populated, determine vocab sizes and create/load model
	inputVocabSize := len(queryVocabulary.WordToToken)
	parentVocabSize := len(parentIntentVocabulary.WordToToken)
	childVocabSize := len(childIntentVocabulary.WordToToken)
	sentenceVocabSize := len(sentenceVocabulary.WordToToken)
	embeddingDim := word2vecModel.VectorSize // Use vector size from word2vec
	numExperts := 2
	maxSequenceLength := 32 // Max length for input query and output description
	maxAttentionHeads := 5

	log.Printf("Query Vocabulary Size: %d", inputVocabSize)
	log.Printf("Parent Intent Vocabulary Size: %d", parentVocabSize)
	log.Printf("Child Intent Vocabulary Size: %d", childVocabSize)
	log.Printf("Sentence Vocabulary Size: %d", sentenceVocabSize)

	var intentMoEModel *moe.IntentMoE // Declare intentMoEModel here

	modelSavePath := "gob_models/moe_classification_model.gob"

	// Always create a new IntentMoE model for now to debug gob loading
	log.Printf("Creating a new IntentMoE model.")
	intentMoEModel, err = moe.NewIntentMoE(inputVocabSize, embeddingDim, numExperts, parentVocabSize, childVocabSize, sentenceVocabSize, maxAttentionHeads, word2vecModel)
	if err != nil {
		log.Fatalf("Failed to create new IntentMoE model: %v", err)
	}

	// Create tokenizers once after vocabularies are loaded/created
	queryTokenizer, err := tokenizer.NewTokenizer(queryVocabulary)
	if err != nil {
		log.Fatalf("Failed to create query tokenizer: %v", err)
	}
	sentenceTokenizer, err := tokenizer.NewTokenizer(sentenceVocabulary)
	if err != nil {
		log.Fatalf("Failed to create sentence tokenizer: %v", err)
	}

	// Train the model
	err = TrainIntentMoEModel(intentMoEModel, intentTrainingData, epochs, learningRate, batchSize, queryVocabulary, parentIntentVocabulary, childIntentVocabulary, sentenceVocabulary, queryTokenizer, sentenceTokenizer, maxSequenceLength, f)
	if err != nil {
		log.Fatalf("Failed to train IntentMoE model: %v", err)
	}

	// Save the trained model
	fmt.Printf("Saving IntentMoE model to %s", modelSavePath)
	err = moe.SaveIntentMoEModelToGOB(intentMoEModel, modelSavePath)
	if err != nil {
		log.Fatalf("Failed to save IntentMoE model: %v", err)
	}

	// Save the vocabularies
	queryVocabularySavePath := "gob_models/query_vocabulary.gob"
	err = queryVocabulary.Save(queryVocabularySavePath)
	if err != nil {
		log.Fatalf("Failed to save query vocabulary: %v", err)
	}
	err = parentIntentVocabulary.Save(parentIntentVocabularySavePath)
	if err != nil {
		log.Fatalf("Failed to save parent intent vocabulary: %v", err)
	}
	err = childIntentVocabulary.Save(childIntentVocabularySavePath)
	if err != nil {
		log.Fatalf("Failed to save child intent vocabulary: %v", err)
	}
	err = sentenceVocabulary.Save(sentenceVocabularySavePath)
	if err != nil {
		log.Fatalf("Failed to save sentence vocabulary: %v", err)
	}
}
