package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"time"

	"github.com/zendrulat/nlptagger/neural/nn"
	"github.com/zendrulat/nlptagger/neural/nnu/seq2seq"
	"github.com/zendrulat/nlptagger/neural/nnu/vocab"
	"github.com/zendrulat/nlptagger/neural/tensor"
	"github.com/zendrulat/nlptagger/neural/tokenizer"
)

// CommandData represents the structure of each entry in software_commands.json
type CommandData struct {
	Query       string `json:"Query"`
	Description string `json:"description"`
}

var (
	trainingDataPath = flag.String("training_data_path", "trainingdata/software_commands.json", "Path to the training data JSON file")
	modelSavePath    = flag.String("model_save_path", "gob_models/seq2seq_description_model.gob", "Path to save the trained Seq2Seq model")
	inputVocabPath   = flag.String("input_vocab_path", "gob_models/seq2seq_input_vocab.gob", "Path to save the input vocabulary")
	outputVocabPath  = flag.String("output_vocab_path", "gob_models/seq2seq_output_vocab.gob", "Path to save the output vocabulary")

	embeddingDim = flag.Int("embedding_dim", 64, "Dimension of word embeddings")
	hiddenDim    = flag.Int("hidden_dim", 128, "Dimension of LSTM hidden states")
	learningRate = flag.Float64("learning_rate", 0.001, "Learning rate for the optimizer")
	epochs       = flag.Int("epochs", 2, "Number of training epochs")
	batchSize    = flag.Int("batch_size", 32, "Batch size for training")
	maxSeqLen    = flag.Int("max_seq_len", 16, "Maximum sequence length for padding")
)

func main() {
	flag.Parse()
	rand.Seed(time.Now().UnixNano())

	log.Printf("Loading training data from %s", *trainingDataPath)
	data, err := loadTrainingData(*trainingDataPath)
	if err != nil {
		log.Fatalf("Failed to load training data: %v", err)
	}

	log.Println("Creating vocabularies...")
	inputVocab, outputVocab := createVocabularies(data)

	log.Printf("Input vocabulary size: %d", inputVocab.Size())
	log.Printf("Output vocabulary size: %d", outputVocab.Size())

	// Save vocabularies
	if err := inputVocab.Save(*inputVocabPath); err != nil {
		log.Fatalf("Failed to save input vocabulary: %v", err)
	}
	if err := outputVocab.Save(*outputVocabPath); err != nil {
		log.Fatalf("Failed to save output vocabulary: %v", err)
	}
	log.Printf("Vocabularies saved to %s and %s", *inputVocabPath, *outputVocabPath)

	log.Println("Initializing tokenizer...")
	// The tokenizer uses the input vocabulary for encoding queries
	queryTokenizer, err := tokenizer.NewTokenizer(inputVocab)
	if err != nil {
		log.Fatalf("Failed to create query tokenizer: %v", err)
	}

	log.Println("Initializing Seq2Seq model...")
	model, err := seq2seq.NewSeq2Seq(
		inputVocab.Size(),
		outputVocab.Size(),
		*embeddingDim,
		*hiddenDim,
		queryTokenizer,
		outputVocab,
	)
	if err != nil {
		log.Fatalf("Failed to create Seq2Seq model: %v", err)
	}

	// Optimizer
	optimizer := nn.NewOptimizer(model.Parameters(), *learningRate, 0.0)

	log.Println("Starting training...")
	for epoch := 0; epoch < *epochs; epoch++ {
		log.Printf("Epoch %d/%d", epoch+1, *epochs)
		// Shuffle data for each epoch
		rand.Shuffle(len(data), func(i, j int) {
			data[i], data[j] = data[j], data[i]
		})

		totalLoss := 0.0
		numBatches := (len(data) + *batchSize - 1) / *batchSize

		for i := 0; i < numBatches; i++ {
			start := i * *batchSize
			end := (i + 1) * *batchSize
			if end > len(data) {
				end = len(data)
			}
			batch := data[start:end]

			// Prepare batch tensors
			inputBatch, targetBatch, err := prepareBatch(batch, inputVocab, outputVocab, queryTokenizer, *maxSeqLen)
			if err != nil {
				log.Fatalf("Failed to prepare batch: %v", err)
			}

			// Zero gradients
			optimizer.ZeroGrad()

			// Forward pass
			predictions, err := model.Forward(inputBatch, targetBatch)
			if err != nil {
				log.Fatalf("Forward pass failed: %v", err)
			}

			// Calculate loss (Cross-Entropy Loss)
			loss, err := calculateLoss(predictions, targetBatch, outputVocab.PaddingTokenID)
			if err != nil {
				log.Fatalf("Loss calculation failed: %v", err)
			}
			totalLoss += loss.Data[0]

			// Backward pass
			if err := loss.Backward(tensor.NewTensor([]int{1}, []float64{1.0}, false)); err != nil {
				log.Fatalf("Backward pass failed: %v", err)
			}

			// Update weights
			optimizer.Step()
		}
		log.Printf("Epoch %d Loss: %.4f", epoch+1, totalLoss/float64(numBatches))
	}

	log.Printf("Training complete. Saving model to %s", *modelSavePath)
	if err := model.Save(*modelSavePath); err != nil {
		log.Fatalf("Failed to save model: %v", err)
	}
}

func loadTrainingData(filePath string) ([]CommandData, error) {
	fileContent, err := ioutil.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	var data []CommandData
	if err := json.Unmarshal(fileContent, &data); err != nil {
		return nil, fmt.Errorf("failed to unmarshal JSON: %w", err)
	}
	return data, nil
}

func createVocabularies(data []CommandData) (*vocab.Vocabulary, *vocab.Vocabulary) {
	inputVocab := vocab.NewVocabulary()
	outputVocab := vocab.NewVocabulary()

	for _, entry := range data {
		// Add query words to input vocabulary
		words := tokenizer.Tokenize(entry.Query) // Assuming a simple space tokenizer for now
		for _, word := range words {
			inputVocab.AddToken(word)
		}

		// Add description words to output vocabulary
		descWords := tokenizer.Tokenize(entry.Description) // Assuming a simple space tokenizer for now
		for _, word := range descWords {
			outputVocab.AddToken(word)
		}
	}
	return inputVocab, outputVocab
}

func prepareBatch(batch []CommandData, inputVocab, outputVocab *vocab.Vocabulary, queryTokenizer *tokenizer.Tokenizer, maxSeqLen int) (*tensor.Tensor, *tensor.Tensor, error) {
	batchSize := len(batch)
	inputIDsBatch := make([][]int, batchSize)
	targetIDsBatch := make([][]int, batchSize)

	for i, entry := range batch {
		// Encode query
		inputTokens, err := queryTokenizer.Encode(entry.Query)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to encode query: %w", err)
		}
		inputIDsBatch[i] = padSequence(inputTokens, inputVocab.PaddingTokenID, maxSeqLen)

		// Encode target description (add BOS and EOS tokens)
		targetTokens := tokenizer.Tokenize(entry.Description)
		targetIDs := make([]int, 0, len(targetTokens)+2)
		targetIDs = append(targetIDs, outputVocab.BosID)
		for _, word := range targetTokens {
			targetIDs = append(targetIDs, outputVocab.GetTokenID(word))
		}
		targetIDs = append(targetIDs, outputVocab.EosID)
		targetIDsBatch[i] = padSequence(targetIDs, outputVocab.PaddingTokenID, maxSeqLen)
	}

	// Convert to Tensors
	inputTensorData := make([]float64, batchSize*maxSeqLen)
	for i, seq := range inputIDsBatch {
		for j, id := range seq {
			inputTensorData[i*maxSeqLen+j] = float64(id)
		}
	}
	inputTensor := tensor.NewTensor([]int{batchSize, maxSeqLen}, inputTensorData, true)

	targetTensorData := make([]float64, batchSize*maxSeqLen)
	for i, seq := range targetIDsBatch {
		for j, id := range seq {
			targetTensorData[i*maxSeqLen+j] = float64(id)
		}
	}
	targetTensor := tensor.NewTensor([]int{batchSize, maxSeqLen}, targetTensorData, true)

	return inputTensor, targetTensor, nil
}

func padSequence(ids []int, padID, maxLen int) []int {
	if len(ids) >= maxLen {
		return ids[:maxLen]
	}
	padded := make([]int, maxLen)
	copy(padded, ids)
	for i := len(ids); i < maxLen; i++ {
		padded[i] = padID
	}
	return padded
}

// calculateLoss computes the Cross-Entropy Loss.
// predictions: [batch_size, seq_len, vocab_size]
// targets: [batch_size, seq_len]
func calculateLoss(predictions, targets *tensor.Tensor, paddingID int) (*tensor.Tensor, error) {
	batchSize := predictions.Shape[0]
	seqLen := predictions.Shape[1]
	vocabSize := predictions.Shape[2]

	// Reshape predictions to [batch_size * seq_len, vocab_size]
	predictionsFlat, err := predictions.Reshape([]int{batchSize * seqLen, vocabSize})
	if err != nil {
		return nil, fmt.Errorf("failed to reshape predictions for loss calculation: %w", err)
	}

	// Apply LogSoftmax
	logSoftmax, err := predictionsFlat.Softmax(1)
	if err != nil {
		return nil, fmt.Errorf("failed to apply softmax: %w", err)
	}
	logSoftmax, err = logSoftmax.Log()
	if err != nil {
		return nil, fmt.Errorf("failed to apply log: %w", err)
	}

	// Reshape targets to [batch_size * seq_len]
	targetsFlat, err := targets.Reshape([]int{batchSize * seqLen})
	if err != nil {
		return nil, fmt.Errorf("failed to reshape targets for loss calculation: %w", err)
	}

	// Negative Log Likelihood
	losses := []*tensor.Tensor{}
	numTokens := 0.0

	for i := 0; i < batchSize*seqLen; i++ {
		targetID := int(targetsFlat.Data[i])

		// Ignore padding tokens in loss calculation
		if targetID == paddingID {
			continue
		}

		// Select the log probability of the correct class
		selectedLogProb, err := logSoftmax.Select(i*vocabSize + targetID)
		if err != nil {
			return nil, fmt.Errorf("failed to select log prob: %w", err)
		}
		losses = append(losses, selectedLogProb)
		numTokens++
	}

	if numTokens == 0 {
		return tensor.NewTensor([]int{1}, []float64{0.0}, false), nil // No valid tokens, loss is 0
	}

	// Sum all losses
	totalLoss := losses[0]
	for i := 1; i < len(losses); i++ {
		totalLoss, err = totalLoss.Add(losses[i])
		if err != nil {
			return nil, fmt.Errorf("failed to sum losses: %w", err)
		}
	}

	// Negate and calculate mean loss
	meanLoss, err := totalLoss.MulScalar(-1.0 / numTokens)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate mean loss: %w", err)
	}

	return meanLoss, nil
}
