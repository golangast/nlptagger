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

	"nlptagger/neural/moe" // Import the MoE package for Seq2SeqMoE
	. "nlptagger/neural/nn"
	mainvocab "nlptagger/neural/nnu/vocab"
	. "nlptagger/neural/tensor"
	"nlptagger/neural/tokenizer"
)

// Seq2SeqTrainingExample represents a single input-output pair for Seq2Seq training.
type Seq2SeqTrainingExample struct {
	Query       string `json:"Query"`
	Description string `json:"description"`
}

// Seq2SeqTrainingData represents the structure of the Seq2Seq training data JSON.
type Seq2SeqTrainingData []Seq2SeqTrainingExample

// LoadSeq2SeqTrainingData loads the Seq2Seq training data from a JSON file.
func LoadSeq2SeqTrainingData(filePath string) (*Seq2SeqTrainingData, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open training data file %s: %w", filePath, err)

	}
	defer file.Close()

	bytes, err := io.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("failed to read training data file %s: %w", filePath, err)
	}

	var data Seq2SeqTrainingData
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

// TrainMoEModel trains the MoEClassificationModel.
func TrainSeq2SeqMoEModel(model *moe.Seq2SeqMoE, data *Seq2SeqTrainingData, epochs int, learningRate float64, batchSize int, queryVocab, descriptionVocab *mainvocab.Vocabulary, queryTokenizer, descriptionTokenizer *tokenizer.Tokenizer, maxSequenceLength int, profileFile *os.File) error {

	if model == nil {
		return errors.New("cannot train a nil model")
	}
	if data == nil || len(*data) == 0 {
		return errors.New("no training data provided")
	}

	optimizer := NewOptimizer(model.Parameters(), learningRate, 5.0) // Using a clip value of 5.0

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
			log.Printf("Starting batch %d/%d in Epoch %d", numBatches+1, (len(*data)+batchSize-1)/batchSize, epoch+1) // Added log

			loss, err := trainSeq2SeqMoEBatch(model, optimizer, batch, queryVocab, descriptionVocab, queryTokenizer, descriptionTokenizer, maxSequenceLength)
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

// trainSeq2SeqMoEBatch performs a single training step on a batch of data.
func trainSeq2SeqMoEBatch(seq2seqMoEModel *moe.Seq2SeqMoE, optimizer Optimizer, batch Seq2SeqTrainingData, queryVocab, descriptionVocab *mainvocab.Vocabulary, queryTokenizer, descriptionTokenizer *tokenizer.Tokenizer, maxSequenceLength int) (float64, error) {
	log.Println("Starting trainSeq2SeqMoEBatch")
	optimizer.ZeroGrad()
	log.Println("ZeroGrad completed.")

	batchSize := len(batch)

	inputIDsBatch := make([]int, batchSize*maxSequenceLength)
	targetDescriptionIDsBatch := make([]int, batchSize*maxSequenceLength)

	log.Println("Starting tokenization and ID conversion.")
	for i, example := range batch {
		// Tokenize and pad query
		queryTokens, err := TokenizeAndConvertToIDs(example.Query, queryTokenizer, queryVocab, maxSequenceLength)
		if err != nil {
			return 0, fmt.Errorf("query tokenization failed for item %d: %w", i, err)
		}
		copy(inputIDsBatch[i*maxSequenceLength:(i+1)*maxSequenceLength], queryTokens)

		// Tokenize and pad description
		descriptionTokens, err := TokenizeAndConvertToIDs(example.Description, descriptionTokenizer, descriptionVocab, maxSequenceLength)
		if err != nil {
			return 0, fmt.Errorf("description tokenization failed for item %d: %w", i, err)
		}
		copy(targetDescriptionIDsBatch[i*maxSequenceLength:(i+1)*maxSequenceLength], descriptionTokens)
	}
	log.Println("Tokenization and ID conversion completed.")

	// Convert input IDs to a Tensor (embeddings will be handled by the model)
	inputTensor := NewTensor([]int{batchSize, maxSequenceLength}, convertIntsToFloat64s(inputIDsBatch), false)
	targetTensor := NewTensor([]int{batchSize, maxSequenceLength}, convertIntsToFloat64s(targetDescriptionIDsBatch), false)
	log.Println("Tensors created.")

	// Forward pass through the Seq2SeqMoE model
	descriptionLogitsSequence, err := seq2seqMoEModel.Forward(inputTensor, targetTensor)
	if err != nil {
		return 0, fmt.Errorf("Seq2SeqMoE model forward pass failed: %w", err)
	}
	log.Println("Completed forward pass.")

	// Calculate loss for the generated description sequence
	totalLoss := 0.0
	var allGrads []*Tensor

	log.Println("Starting loss calculation.")
	for t, logitsTensorForTimeStep := range descriptionLogitsSequence {
		// Extract target IDs for the current time step across the batch
		targetIDsForTimeStep := make([]int, batchSize)
		for b := 0; b < batchSize; b++ {
			targetIDsForTimeStep[b] = targetDescriptionIDsBatch[b*maxSequenceLength+t]
		}

		loss, grad := CrossEntropyLoss(logitsTensorForTimeStep, targetIDsForTimeStep, descriptionVocab.PaddingTokenID)
		totalLoss += loss
		allGrads = append(allGrads, grad)
	}
	log.Println("Loss calculation completed.")

	// Backward pass
	err = seq2seqMoEModel.Backward(allGrads)
	if err != nil {
		return 0, fmt.Errorf("Seq2SeqMoE model backward pass failed: %w", err)
	}
	log.Println("Completed backward pass.")

	optimizer.Step()
	log.Println("Optimizer step completed.")

	return totalLoss, nil
}

func convertIntsToFloat64s(input []int) []float64 {
	output := make([]float64, len(input))
	for i, v := range input {
		output[i] = float64(v)
	}
	return output
}

func main() {
	// Start CPU profiling
	log.Println("Attempting to create cpu.prof file...")
	f, err := os.Create("cpu.prof")
	if err != nil {
		log.Fatal("could not create CPU profile: ", err)
	}
	log.Println("Starting CPU profile...")
	if err := pprof.StartCPUProfile(f); err != nil {
		log.Fatal("could not start CPU profile: ", err)
	}
	log.Println("CPU profiling started.")

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
	epochs := 1
	learningRate := 0.001
	batchSize := 32
	queryVocabularySavePath := "gob_models/query_vocabulary.gob"
	descriptionVocabularySavePath := "gob_models/description_vocabulary.gob"

	// Define paths for training data
	const seq2seqTrainingDataPath = "./trainingdata/wikiqa_seq2seq_training.json" // Using transformed WikiQA data

	// Try to load vocabularies first
	queryVocabulary, err := mainvocab.LoadVocabulary(queryVocabularySavePath)
	if err != nil {
		log.Println("Failed to load query vocabulary, creating a new one.")
		queryVocabulary = mainvocab.NewVocabulary()
	}

	descriptionVocabulary, err := mainvocab.LoadVocabulary(descriptionVocabularySavePath)
	if err != nil {
		log.Println("Failed to load description vocabulary, creating a new one.")
		descriptionVocabulary = mainvocab.NewVocabulary()
	}

	// Load Seq2Seq training data
	seq2seqTrainingData, err := LoadSeq2SeqTrainingData(seq2seqTrainingDataPath)
	if err != nil {
		log.Fatalf("Failed to load Seq2Seq training data from %s: %v", seq2seqTrainingDataPath, err)
	}
	log.Printf("Loaded %d training examples.", len(*seq2seqTrainingData))

	// If vocabularies are empty, populate them from training data
	if len(queryVocabulary.TokenToWord) <= 2 || len(descriptionVocabulary.TokenToWord) <= 2 { // <= 2 to account for pad and unk
		log.Println("Populating vocabularies from training data...")

		// Populate vocabularies with tokens from training data
		for _, pair := range *seq2seqTrainingData {
			// Add query tokens to query vocabulary
			queryWords := strings.Fields(strings.ToLower(pair.Query))
			for _, word := range queryWords {
				queryVocabulary.AddToken(word)
			}

			// Add description tokens to description vocabulary
			descriptionWords := strings.Fields(strings.ToLower(pair.Description))
			for _, word := range descriptionWords {
				descriptionVocabulary.AddToken(word)
			}
		}
	}

	log.Printf("Description Vocabulary TokenToWord: %v", descriptionVocabulary.TokenToWord)

	// After vocabularies are fully populated, determine vocab sizes and create/load model
	inputVocabSize := len(queryVocabulary.TokenToWord)
	outputVocabSize := len(descriptionVocabulary.TokenToWord)
	embeddingDim := 128
	numExperts := 2
	maxAttentionHeads := 4 // Added this line
	// k := 2 // k is for top-k experts, not directly used in Seq2SeqMoE New function
	maxSequenceLength := 32 // Max length for input query and output description

	log.Printf("Query Vocabulary Size: %d", inputVocabSize)
	log.Printf("Description Vocabulary Size: %d", outputVocabSize)

	var seq2seqMoEModel *moe.Seq2SeqMoE // Declare seq2seqMoEModel here

	modelSavePath := "gob_models/moe_model.gob"

	// Try to load Seq2SeqMoE model
	seq2seqMoEModel, err = moe.LoadSeq2SeqMoEModelFromGOB(modelSavePath)
	if err != nil {
		log.Printf("Failed to load Seq2SeqMoE model, creating a new one: %v", err)
		seq2seqMoEModel, err = moe.NewSeq2SeqMoE(inputVocabSize, embeddingDim, numExperts, outputVocabSize, maxAttentionHeads)
		if err != nil {
			log.Fatalf("Failed to create new Seq2SeqMoE model: %v", err)
		}
	} else {
		log.Printf("Loaded Seq2SeqMoE model from %s", modelSavePath)
	}

	// Create tokenizers once after vocabularies are loaded/created
	queryTokenizer, err := tokenizer.NewTokenizer(queryVocabulary)
	if err != nil {
		log.Fatalf("Failed to create query tokenizer: %v", err)
	}
	descriptionTokenizer, err := tokenizer.NewTokenizer(descriptionVocabulary)
	if err != nil {
		log.Fatalf("Failed to create description tokenizer: %v", err)
	}

	// Train the model
	err = TrainSeq2SeqMoEModel(seq2seqMoEModel, seq2seqTrainingData, epochs, learningRate, batchSize, queryVocabulary, descriptionVocabulary, queryTokenizer, descriptionTokenizer, maxSequenceLength, f)
	if err != nil {
		log.Fatalf("Failed to train Seq2SeqMoE model: %v", err)
	}

	// Save the trained model
	fmt.Printf("Saving Seq2SeqMoE model to %s", modelSavePath)
	err = moe.SaveSeq2SeqMoEModelToGOB(seq2seqMoEModel, modelSavePath)
	if err != nil {
		log.Fatalf("Failed to save Seq2SeqMoE model: %v", err)
	}

	// Save the vocabularies
	err = queryVocabulary.Save(queryVocabularySavePath)
	if err != nil {
		log.Fatalf("Failed to save query vocabulary: %v", err)
	}
	err = descriptionVocabulary.Save(descriptionVocabularySavePath)
	if err != nil {
		log.Fatalf("Failed to save description vocabulary: %v", err)
	}
}
