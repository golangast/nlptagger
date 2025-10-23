package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"runtime"
	"runtime/pprof"
	"strings"

	"nlptagger/neural/moe"
	. "nlptagger/neural/nn"
	mainvocab "nlptagger/neural/nnu/vocab"
	"nlptagger/neural/semantic"
	. "nlptagger/neural/tensor"
	"nlptagger/neural/tokenizer"
)


	

// IntentTrainingExample represents a single training example with a query and its intents.
type IntentTrainingExample struct {
	Query          string                  `json:"query"`
	SemanticOutput semantic.SemanticOutput `json:"semantic_output"`
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

func writeMemProfile(epoch int) {
    f, err := os.Create(fmt.Sprintf("mem_epoch_%d.prof", epoch))
    if err != nil {
        log.Fatal("could not create memory profile: ", err)
    }
    defer f.Close() // error handling omitted for example
    runtime.GC() // get up-to-date statistics
    if err := pprof.WriteHeapProfile(f); err != nil {
        log.Fatal("could not write memory profile: ", err)
    }
}

// TrainIntentMoEModel trains the MoEClassificationModel.
func TrainIntentMoEModel(model *moe.IntentMoE, data *IntentTrainingData, epochs int, learningRate float64, batchSize int, queryVocab, semanticOutputVocab *mainvocab.Vocabulary, queryTokenizer, semanticOutputTokenizer *tokenizer.Tokenizer, maxSequenceLength int) error {

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

			loss, err := trainIntentMoEBatch(model, optimizer, batch, queryVocab, semanticOutputVocab, queryTokenizer, semanticOutputTokenizer, maxSequenceLength)
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
		writeMemProfile(epoch + 1)
	}

	return nil
}

// trainIntentMoEBatch performs a single training step on a batch of data.
func trainIntentMoEBatch(intentMoEModel *moe.IntentMoE, optimizer Optimizer, batch IntentTrainingData, queryVocab, semanticOutputVocab *mainvocab.Vocabulary, queryTokenizer, semanticOutputTokenizer *tokenizer.Tokenizer, maxSequenceLength int) (float64, error) {
	optimizer.ZeroGrad()

	batchSize := len(batch)

	inputIDsBatch := make([]int, batchSize*maxSequenceLength)
	semanticOutputIDsBatch := make([]int, batchSize*maxSequenceLength)

	for i, example := range batch {
		// Tokenize and pad query
		queryTokens, err := TokenizeAndConvertToIDs(example.Query, queryTokenizer, queryVocab, maxSequenceLength)
		if err != nil {
			return 0, fmt.Errorf("query tokenization failed for item %d: %w", i, err)
		}
		copy(inputIDsBatch[i*maxSequenceLength:(i+1)*maxSequenceLength], queryTokens)

		// Serialize and tokenize semantic output
		semanticOutputJSON, err := json.Marshal(example.SemanticOutput)
		if err != nil {
			return 0, fmt.Errorf("failed to marshal semantic output: %w", err)
		}

		trainingSemanticOutput := "<s> " + string(semanticOutputJSON) + " </s>"
		semanticOutputTokens, err := TokenizeAndConvertToIDs(trainingSemanticOutput, semanticOutputTokenizer, semanticOutputVocab, maxSequenceLength)
		if err != nil {
			return 0, fmt.Errorf("semantic output tokenization failed for item %d: %w", i, err)
		}
				log.Printf("Query: %s", example.Query)
		log.Printf("Tokenized Query: %v", queryTokens)
		log.Printf("Semantic Output: %s", trainingSemanticOutput)
		log.Printf("Tokenized Semantic Output: %v", semanticOutputTokens)
		copy(semanticOutputIDsBatch[i*maxSequenceLength:(i+1)*maxSequenceLength], semanticOutputTokens)
	}

	// Convert input IDs to a Tensor (embeddings will be handled by the model)
	inputTensor := NewTensor([]int{batchSize, maxSequenceLength}, convertIntsToFloat64s(inputIDsBatch), false)
	semanticOutputTensor := NewTensor([]int{batchSize, maxSequenceLength}, convertIntsToFloat64s(semanticOutputIDsBatch), false)

	// Forward pass through the IntentMoE model
	semanticOutputLogits, _, err := intentMoEModel.Forward(inputTensor, semanticOutputTensor)
	if err != nil {
		return 0, fmt.Errorf("IntentMoE model forward pass failed: %w", err)
	}

	// Calculate loss for the semantic output
	semanticOutputLoss := 0.0
	semanticOutputGrads := make([]*Tensor, maxSequenceLength)
	for t := 0; t < maxSequenceLength; t++ {
		targets := make([]int, batchSize)
		for i := 0; i < batchSize; i++ {
			targets[i] = semanticOutputIDsBatch[i*maxSequenceLength+t]
		}
		loss, grad := CrossEntropyLoss(semanticOutputLogits[t], targets, semanticOutputVocab.PaddingTokenID)
		semanticOutputLoss += loss
		semanticOutputGrads[t] = grad
	}

	totalLoss := semanticOutputLoss

	// Backward pass
	err = intentMoEModel.Backward(semanticOutputGrads...)
	if err != nil {
		return 0, fmt.Errorf("IntentMoE model backward pass failed: %w", err)
	}

	optimizer.Step()

	// Log guessed sentence
	/*
		guessedIDs, err := intentMoEModel.GreedySearchDecode(contextVector, maxSequenceLength, semanticOutputVocab.GetTokenID("<s>"), semanticOutputVocab.GetTokenID("</s>"))
		if err != nil {
			log.Printf("Error decoding guessed sentence: %v", err)
		} else {
			guessedSentence, err := semanticOutputTokenizer.Decode(guessedIDs)
			if err != nil {
				log.Printf("Error decoding guessed sentence: %v", err)
			} else {
				log.Printf("Guessed semantic output: %s", guessedSentence)
			}
			targetJSON, _ := json.Marshal(batch[0].SemanticOutput)
			log.Printf("Target semantic output: %s", string(targetJSON))
		}
	*/

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

func BuildVocabularies(dataPath string) (*mainvocab.Vocabulary, *mainvocab.Vocabulary, error) {
	queryVocabulary := mainvocab.NewVocabulary()
	semanticOutputVocabulary := mainvocab.NewVocabulary()

	semanticTrainingData, err := LoadIntentTrainingData(dataPath)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to load semantic training data from %s: %w", dataPath, err)
	}

	for _, pair := range *semanticTrainingData {
		// Use the same tokenizer logic as during inference to build the vocabulary
		tokenizedQuery := tokenizer.Tokenize(strings.ToLower(pair.Query))
		for _, word := range tokenizedQuery {
			queryVocabulary.AddToken(word)
		}

		semanticOutputJSON, err := json.Marshal(pair.SemanticOutput)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to marshal semantic output: %w", err)
		}

		// Add BOS and EOS tokens to the sentence when building the vocabulary
		trainingSemanticOutput := "<s> " + string(semanticOutputJSON) + " </s>"
		tokenizedSemanticOutput := tokenizer.Tokenize(trainingSemanticOutput)
		for _, word := range tokenizedSemanticOutput {
			semanticOutputVocabulary.AddToken(word)
		}
	}

	// Explicitly add BOS and EOS tokens to the sentence vocabulary
	semanticOutputVocabulary.BosID = semanticOutputVocabulary.GetTokenID("<s>")
	semanticOutputVocabulary.EosID = semanticOutputVocabulary.GetTokenID("</s>")

	return queryVocabulary, semanticOutputVocabulary, nil
}

func main() {
	const trainingDataPath = "./trainingdata/intent_data.json"
	const semanticTrainingDataPath = "./trainingdata/semantic_output_data.json"
	const word2vecModelPath = "gob_models/word2vec_model.gob"

	// Define training parameters
	epochs := 10
	learningRate := 0.001
	batchSize := 1
	semanticOutputVocabularySavePath := "gob_models/semantic_output_vocabulary.gob"

	/*
		// Load Word2Vec model
		word2vecModel, err := word2vec.LoadModel(word2vecModelPath)
		if err != nil {
			log.Fatalf("Failed to load Word2Vec model: %v", err)
		}

		// Create query vocabulary from word2vec model
		queryVocabulary := convertW2VVocab(word2vecModel.Vocabulary)
	*/

	// Try to load other vocabularies first
	semanticOutputVocabulary, err := mainvocab.LoadVocabulary(semanticOutputVocabularySavePath)
	if err != nil {
		log.Println("Failed to load semantic output vocabulary, creating a new one.")
	}

	var queryVocabulary *mainvocab.Vocabulary
	if semanticOutputVocabulary == nil || queryVocabulary == nil {
		log.Println("Building vocabularies from scratch...")
		queryVocabulary, semanticOutputVocabulary, err = BuildVocabularies(semanticTrainingDataPath)
		if err != nil {
			log.Fatalf("Failed to build vocabularies: %v", err)
		}
	}

	log.Printf("Query Vocabulary (after load/create): Size=%d", len(queryVocabulary.WordToToken))
	log.Printf("Semantic Output Vocabulary (after load/create): Size=%d", len(semanticOutputVocabulary.WordToToken))

	// Load Intent training data
	semanticTrainingData, err := LoadIntentTrainingData(semanticTrainingDataPath)
	if err != nil {
		log.Fatalf("Failed to load semantic training data from %s: %v", semanticTrainingDataPath, err)
	}
	log.Printf("Loaded %d training examples.", len(*semanticTrainingData))

	// After vocabularies are fully populated, determine vocab sizes and create/load model
	inputVocabSize := len(queryVocabulary.WordToToken)
	semanticOutputVocabSize := len(semanticOutputVocabulary.WordToToken)
	embeddingDim := 25 // Use vector size from word2vec
	numExperts := 1
	maxSequenceLength := 32 // Max length for input query and output description
	maxAttentionHeads := 1

	log.Printf("Query Vocabulary Size: %d", inputVocabSize)
	log.Printf("Semantic Output Vocabulary Size: %d", semanticOutputVocabSize)
	log.Printf("Embedding Dimension: %d", embeddingDim)

	var intentMoEModel *moe.IntentMoE // Declare intentMoEModel here

	modelSavePath := "gob_models/moe_classification_model.gob"

	// Always create a new IntentMoE model for now to debug gob loading
	log.Printf("Creating a new IntentMoE model.")
	intentMoEModel, err = moe.NewIntentMoE(inputVocabSize, embeddingDim, numExperts, 0, 0, semanticOutputVocabSize, maxAttentionHeads, nil)
	if err != nil {
		log.Fatalf("Failed to create new IntentMoE model: %v", err)
	}

	// Create tokenizers once after vocabularies are loaded/created
	queryTokenizer, err := tokenizer.NewTokenizer(queryVocabulary)
	if err != nil {
		log.Fatalf("Failed to create query tokenizer: %v", err)
	}
	semanticOutputTokenizer, err := tokenizer.NewTokenizer(semanticOutputVocabulary)
	if err != nil {
		log.Fatalf("Failed to create semantic output tokenizer: %v", err)
	}



	// Train the model
	err = TrainIntentMoEModel(intentMoEModel, semanticTrainingData, epochs, learningRate, batchSize, queryVocabulary, semanticOutputVocabulary, queryTokenizer, semanticOutputTokenizer, maxSequenceLength)
	if err != nil {
		log.Fatalf("Failed to train IntentMoE model: %v", err)
	}

	// Save the trained model
	fmt.Printf("Saving IntentMoE model to %s\n", modelSavePath)
	writeMemProfile(11) // Memory profile before saving
	modelFile, err := os.Create(modelSavePath)
	if err != nil {
		log.Fatalf("Failed to create model file: %v", err)
	}
	defer modelFile.Close()
	err = moe.SaveIntentMoEModelToGOB(intentMoEModel, modelFile)
	if err != nil {
		log.Fatalf("Failed to save IntentMoE model: %v", err)
	}
	writeMemProfile(12) // Memory profile after saving

	// Save the vocabularies
	queryVocabularySavePath := "gob_models/query_vocabulary.gob"
	err = queryVocabulary.Save(queryVocabularySavePath)
	if err != nil {
		log.Fatalf("Failed to save query vocabulary: %v", err)
	}
	err = semanticOutputVocabulary.Save(semanticOutputVocabularySavePath)
	if err != nil {
		log.Fatalf("Failed to save semantic output vocabulary: %v", err)
	}

	
}

