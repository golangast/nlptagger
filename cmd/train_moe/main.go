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
	"github.com/zendrulat/nlptagger/neural/nnu/word2vec"
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
		log.Printf("DEBUG: Serialized Semantic Output (raw): %s", serializedSemanticOutput)
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
	semanticOutputLogits, contextVector, err := intentMoEModel.Forward(inputTensor, semanticOutputTensor)
	if err != nil {
		return 0, fmt.Errorf("IntentMoE model forward pass failed: %w", err)
	}

	// Calculate loss for the semantic output
	semanticOutputLoss := 0.0
	semanticOutputGrads := make([]*tensor.Tensor, len(semanticOutputLogits))
	for t := 0; t < len(semanticOutputLogits); t++ {
		targets := make([]int, batchSize)
		for i := 0; i < batchSize; i++ {
			targets[i] = semanticOutputIDsBatch[i*maxSequenceLength+t+1]
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
	// Create a temporary struct to hold the data for JSON marshaling.
	// This helps in flattening the structure if needed or selectively including fields.
	// For now, we directly marshal the SemanticOutput struct.
	jsonBytes, err := json.Marshal(so)
	if err != nil {
		log.Printf("Error marshalling semantic output to JSON: %v", err)
		return "" // Return an empty string or handle error as appropriate
	}
	return string(jsonBytes)
}

func serializeTargetResource(tr semantic.Resource) string {
	// This function is no longer called by serializeSemanticOutput, but we'll keep it
	// in case it's used elsewhere. Or we can remove it if it's confirmed to be unused.
	var parts []string
	resourceType := strings.ReplaceAll(tr.Type, "::", "_")
	parts = append(parts, fmt.Sprintf("resource_type:%s", resourceType))
	parts = append(parts, fmt.Sprintf("resource_name:%s", tr.Name))
	if tr.Properties != nil {
		for k, v := range tr.Properties {
			parts = append(parts, fmt.Sprintf("property_%s:%v", k, v))
		}
	}
	for _, child := range tr.Children {
		childParts := serializeTargetResource(child)
		parts = append(parts, childParts)
	}
	return strings.Join(parts, ", ")
}

// BuildVocabularies builds vocabularies from intent_data.json and WikiQA-train.txt.
func BuildVocabularies(intentDataPath, wikiQADataPath string, word2vecModel *word2vec.SimpleWord2Vec) (*mainvocab.Vocabulary, *mainvocab.Vocabulary, error) {
	localQueryVocabulary := mainvocab.NewVocabulary() // Always start with a fresh vocabulary
	log.Printf("queryVocabulary size after initial creation: %d", localQueryVocabulary.Size())

	if word2vecModel != nil {
		// Add all words from the Word2Vec model to the new queryVocabulary
		// This will assign new, contiguous IDs starting from where queryVocabulary currently is.
		for word := range word2vecModel.Vocabulary {
			localQueryVocabulary.AddToken(word)
		}
		log.Printf("queryVocabulary size after adding words from Word2Vec: %d", localQueryVocabulary.Size())
	}
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
		// Always add query tokens to queryVocabulary
		tokenizedQuery := tokenizer.Tokenize(strings.ToLower(example.Query))
		for _, word := range tokenizedQuery {
			localQueryVocabulary.AddToken(word)
		}

		// Semantic output vocabulary
		serializedSemanticOutput := serializeSemanticOutput(example.SemanticOutput)
		trainingSemanticOutput := "<s> " + serializedSemanticOutput + " </s>"
		tokenizedSemanticOutput := tokenizer.Tokenize(trainingSemanticOutput)
		for _, word := range tokenizedSemanticOutput {
			semanticOutputVocabulary.AddToken(word)
		}
	}

	// Process WikiQA-train.txt (always add to query vocab)
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
					localQueryVocabulary.AddToken(token)
				}
				// Tokenize answer part
				answerTokens := tokenizer.Tokenize(strings.ToLower(parts[1]))
				for _, token := range answerTokens {
					localQueryVocabulary.AddToken(token)
				}
			}
		}

		if err := scanner.Err(); err != nil {
			log.Printf("Warning: Error reading WikiQA training data %s: %v. Continuing.", wikiQADataPath, err)
		}
	}

	log.Printf("queryVocabulary size before final return: %d", localQueryVocabulary.Size())
	// Explicitly add BOS and EOS tokens to the semantic output vocabulary
	semanticOutputVocabulary.AddToken("<s>")
	semanticOutputVocabulary.AddToken("</s>")
	semanticOutputVocabulary.BosID = semanticOutputVocabulary.GetTokenID("<s>")
	semanticOutputVocabulary.EosID = semanticOutputVocabulary.GetTokenID("</s>")

	return localQueryVocabulary, semanticOutputVocabulary, nil
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

	// Load the Word2Vec model
	log.Printf("Loading Word2Vec model from %s...", word2vecModelPath)
	word2vecModel, err := word2vec.LoadModel(word2vecModelPath)
	if err != nil {
		log.Fatalf("Failed to load Word2Vec model: %v", err)
	}
	log.Println("Word2Vec model loaded successfully.")

	// Define training parameters
	epochs := 100
	learningRate := 0.001
	batchSize := 32
	semanticOutputVocabularySavePath := "gob_models/semantic_output_vocabulary.gob"

	var queryVocabulary *mainvocab.Vocabulary
	var semanticOutputVocabulary *mainvocab.Vocabulary
	log.Println("Building vocabularies from intent_data.json and WikiQA-train.txt...")
	queryVocabulary, semanticOutputVocabulary, err = BuildVocabularies(intentDataPath, wikiQADataPath, word2vecModel)
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
	intentMoEModel, err = moe.NewIntentMoE(inputVocabSize, embeddingDim, numExperts, 0, 0, semanticOutputVocabSize, maxAttentionHeads, word2vecModel)
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
	modelFile, err := os.Create(modelSavePath)
	if err != nil {
		log.Fatalf("Failed to create model file: %v", err)
	}
	defer modelFile.Close()
	err = moe.SaveIntentMoEModelToGOB(intentMoEModel, modelFile)
	if err != nil {
		log.Fatalf("Failed to save IntentMoE model: %v", err)
	}

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
