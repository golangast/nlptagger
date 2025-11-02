package main

import (
	"bufio"
	"encoding/gob"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"runtime"
	"runtime/pprof"
	"strings"

	"github.com/zendrulat/nlptagger/neural/moe"
	. "github.com/zendrulat/nlptagger/neural/nn"
	mainvocab "github.com/zendrulat/nlptagger/neural/nnu/vocab"
	"github.com/zendrulat/nlptagger/neural/semantic"
	tensor "github.com/zendrulat/nlptagger/neural/tensor"
	"github.com/zendrulat/nlptagger/neural/tokenizer"
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
	runtime.GC()    // get up-to-date statistics
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
		serializedSemanticOutput := serializeSemanticOutput(example.SemanticOutput)
		trainingSemanticOutput := "<s> " + serializedSemanticOutput + " </s>"
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
	inputTensor := tensor.NewTensor([]int{batchSize, maxSequenceLength}, convertIntsToFloat64s(inputIDsBatch), false)
	semanticOutputTensor := tensor.NewTensor([]int{batchSize, maxSequenceLength}, convertIntsToFloat64s(semanticOutputIDsBatch), false)

	// Forward pass through the IntentMoE model
	semanticOutputLogits, _, err := intentMoEModel.Forward(inputTensor, semanticOutputTensor)
	if err != nil {
		return 0, fmt.Errorf("IntentMoE model forward pass failed: %w", err)
	}

	// Calculate loss for the semantic output
	semanticOutputLoss := 0.0
	semanticOutputGrads := make([]*tensor.Tensor, maxSequenceLength)
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

func serializeSemanticOutput(so semantic.SemanticOutput) string {
	var parts []string

	// Operation
	parts = append(parts, fmt.Sprintf("operation:%s", so.Operation))

	// TargetResource
	parts = append(parts, serializeTargetResource(*so.TargetResource))

	// Context
	if so.Context.UserRole != "" {
		parts = append(parts, fmt.Sprintf("context_user_role:%s", so.Context.UserRole))
	}

	return strings.Join(parts, " ")
}

func serializeTargetResource(tr semantic.Resource) string {
	var parts []string

	parts = append(parts, fmt.Sprintf("resource_type:%s", tr.Type))
	parts = append(parts, fmt.Sprintf("resource_name:%s", tr.Name))

	// Properties
	if tr.Properties != nil {
		for k, v := range tr.Properties {
			parts = append(parts, fmt.Sprintf("property_%s:%v", k, v))
		}
	}

	// Children
	if len(tr.Children) > 0 {
		childrenParts := []string{}
		for _, child := range tr.Children {
			childrenParts = append(childrenParts, serializeTargetResource(child))
		}
		parts = append(parts, fmt.Sprintf("children:[%s]", strings.Join(childrenParts, " ")))
	}

	return strings.Join(parts, " ")
}

// BuildVocabularies builds vocabularies from intent_data.json and WikiQA-train.txt.
func BuildVocabularies(intentDataPath, wikiQADataPath string) (*mainvocab.Vocabulary, *mainvocab.Vocabulary, error) {
	queryVocabulary := mainvocab.NewVocabulary()
	semanticOutputVocabulary := mainvocab.NewVocabulary()

	// Process intent_data.json
	intentFile, err := os.Open(intentDataPath)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to open intent training data file %s: %w", intentDataPath, err)
	}
	defer intentFile.Close()

	bytes, err := io.ReadAll(intentFile)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to read intent training data file %s: %w", intentDataPath, err)
	}

	var intents []IntentTrainingExample
	if err := json.Unmarshal(bytes, &intents); err != nil {
		return nil, nil, fmt.Errorf("failed to unmarshal intent training data JSON from %s: %w", intentDataPath, err)
	}

	for _, example := range intents {
		// Query vocabulary
		tokenizedQuery := tokenizer.Tokenize(strings.ToLower(example.Query))
		for _, word := range tokenizedQuery {
			queryVocabulary.AddToken(word)
		}

		// Semantic output vocabulary
		serializedSemanticOutput := serializeSemanticOutput(example.SemanticOutput)
		trainingSemanticOutput := "<s> " + serializedSemanticOutput + " </s>"
		tokenizedSemanticOutput := tokenizer.Tokenize(trainingSemanticOutput)
		for _, word := range tokenizedSemanticOutput {
			semanticOutputVocabulary.AddToken(word)
		}
	}

	// Process WikiQA-train.txt
	wikiQAFile, err := os.Open(wikiQADataPath)
	if err != nil {
		log.Printf("Warning: Failed to open WikiQA training data %s: %v. Continuing without it.", wikiQADataPath, err)
	} else {
		defer wikiQAFile.Close()

		scanner := bufio.NewScanner(wikiQAFile)
		for scanner.Scan() {
			line := scanner.Text()
			parts := strings.Split(line, "\t")
			if len(parts) >= 2 {
				// Tokenize question part
				questionTokens := tokenizer.Tokenize(strings.ToLower(parts[0]))
				for _, token := range questionTokens {
					queryVocabulary.AddToken(token)
				}
				// Tokenize answer part
				answerTokens := tokenizer.Tokenize(strings.ToLower(parts[1]))
				for _, token := range answerTokens {
					queryVocabulary.AddToken(token)
				}
			}
		}

		if err := scanner.Err(); err != nil {
			log.Printf("Warning: Error reading WikiQA training data %s: %v. Continuing.", wikiQADataPath, err)
		}
	}

	// Explicitly add BOS and EOS tokens to the semantic output vocabulary
	semanticOutputVocabulary.AddToken("<s>")
	semanticOutputVocabulary.AddToken("</s>")
	semanticOutputVocabulary.BosID = semanticOutputVocabulary.GetTokenID("<s>")
	semanticOutputVocabulary.EosID = semanticOutputVocabulary.GetTokenID("</s>")

	return queryVocabulary, semanticOutputVocabulary, nil
}

func main() {
	gob.Register((*tensor.Operation)(nil))
	gob.Register((*tensor.Tensor)(nil))
	gob.Register(&moe.FeedForwardExpert{})
	gob.Register(&moe.MoELayer{})
	gob.Register(&tensor.AddOperation{})
	gob.Register(&tensor.MatmulOperation{})
	gob.Register(&tensor.AddWithBroadcastOperation{})
	gob.Register(&tensor.DivScalarOperation{})
	gob.Register(&tensor.MulScalarOperation{})
	gob.Register(&tensor.MulOperation{})
	gob.Register(&tensor.SelectOperation{})
	gob.Register(&tensor.TanhOperation{})
	gob.Register(&tensor.SigmoidOperation{})
	gob.Register(&tensor.LogOperation{})
	gob.Register(&tensor.SumOperation{})
	gob.Register(&tensor.ConcatOperation{})
	gob.Register(&tensor.SplitOperation{})
	gob.Register(&tensor.EmbeddingLookupOperation{})
	gob.Register(&tensor.SoftmaxOperation{})
	const intentDataPath = "trainingdata/semantic_output_data.json"
	const wikiQADataPath = "./trainingdata/WikiQA-train.txt"
	const word2vecModelPath = "gob_models/word2vec_model.gob"

	// Define training parameters
	epochs := 200
	learningRate := 0.001
	batchSize := 32
	semanticOutputVocabularySavePath := "gob_models/semantic_output_vocabulary.gob"

	var queryVocabulary *mainvocab.Vocabulary
	var semanticOutputVocabulary *mainvocab.Vocabulary
	log.Println("Building vocabularies from intent_data.json and WikiQA-train.txt...")
	queryVocabulary, semanticOutputVocabulary, err := BuildVocabularies(intentDataPath, wikiQADataPath)
	if err != nil {
		log.Fatalf("Failed to build vocabularies: %v", err)
	}

	log.Printf("Query Vocabulary (after build): Size=%d", queryVocabulary.Size())
	log.Printf("Semantic Output Vocabulary (after build): Size=%d", semanticOutputVocabulary.Size())

	// Load Intent training data
	semanticTrainingData, err := LoadIntentTrainingData(intentDataPath)
	if err != nil {
		log.Fatalf("Failed to load semantic training data from %s: %v", intentDataPath, err)
	}
	log.Printf("Loaded %d training examples.", len(*semanticTrainingData))

	// After vocabularies are fully populated, determine vocab sizes and create/load model
	inputVocabSize := queryVocabulary.Size()
	semanticOutputVocabSize := semanticOutputVocabulary.Size()
	embeddingDim := 64
	numExperts := 4
	maxSequenceLength := 128
	maxAttentionHeads := 4

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

	// Test loading the model right after saving
	log.Println("Testing model loading right after saving...")
	_, err = moe.LoadIntentMoEModelFromGOB(modelSavePath)
	if err != nil {
		log.Fatalf("Failed to load model right after saving: %v", err)
	}
	log.Println("Successfully loaded model right after saving.")

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

