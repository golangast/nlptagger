package bartsimple

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"os"

	"github.com/golangast/nlptagger/neural/nnu/word2vec"
	"github.com/golangast/nlptagger/tagger"
	"github.com/golangast/nlptagger/tagger/nertagger"
	"github.com/golangast/nlptagger/tagger/postagger"
)

// BARTTrainingData represents the structure of the training data JSON.
type BARTTrainingData struct {
	Sentences []struct {
		Input  string `json:"input"`
		Output string `json:"output"`
	} `json:"sentences"`
}

// LoadBARTTrainingData loads the BART training data from a JSON file.
func LoadBARTTrainingData(filePath string) (*BARTTrainingData, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open training data file %s: %w", filePath, err)
	}
	defer file.Close()

	bytes, err := io.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("failed to read training data file %s: %w", filePath, err)
	}

	var data BARTTrainingData
	err = json.Unmarshal(bytes, &data)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal training data JSON from %s: %w", filePath, err)
	}

	return &data, nil
}

// TrainBARTModel trains the SimplifiedBARTModel.
func TrainBARTModel(model *SimplifiedBARTModel, data *BARTTrainingData, epochs int, learningRate float64, batchSize int) error {
	if model == nil {
		return errors.New("cannot train a nil model")
	}
	if data == nil || len(data.Sentences) == 0 {
		return errors.New("no training data provided")
	}
	if model.Tokenizer == nil {
		return errors.New("model tokenizer is not initialized")
	}
	// Check if TokenEmbedding is already initialized. If not, initialize it.
	if model.TokenEmbedding == nil {
		// Load word2vec model
		word2vecModel, err := word2vec.LoadModel("gob_models/word2vec_model.gob")
		if err != nil {
			log.Printf("Warning: Could not load word2vec model: %v. Initializing BART embeddings randomly.\n", err)
			// If word2vec model fails to load, initialize BART embeddings randomly
			model.TokenEmbedding = NewEmbedding(model.VocabSize, model.Encoder.Layer.SelfAttention.DimModel)
		} else {
			log.Println("Word2vec model loaded successfully. Initializing BART embeddings with pretrained vectors.")
			// Convert word2vec.WordVectors to map[string][]float64 for NewEmbeddingWithPretrained
			pretrainedEmbeddings := word2vec.ConvertToMap(word2vecModel.WordVectors, word2vecModel.Vocabulary)
			model.TokenEmbedding = NewEmbeddingWithPretrained(model.VocabSize, model.Encoder.Layer.SelfAttention.DimModel, model.Vocabulary, pretrainedEmbeddings)
		}
	}
	if model.PosTagEmbedding == nil {
		model.PosTagEmbedding = NewEmbedding(len(postagger.PosTagToIDMap()), model.Encoder.Layer.SelfAttention.DimModel)
	}
	if model.NerTagEmbedding == nil {
		model.NerTagEmbedding = NewEmbedding(len(nertagger.NerTagToIDMap()), model.Encoder.Layer.SelfAttention.DimModel)
	}

	// Create an optimizer for the model's parameters. This relies on the Parameters() methods you added.
	optimizer := NewOptimizer(model.Parameters(), learningRate)

	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		numBatches := 0
		// Create batches for training
		for i := 0; i < len(data.Sentences); i += batchSize {
			end := i + batchSize
			if end > len(data.Sentences) {
				end = len(data.Sentences)
			}
			batch := data.Sentences[i:end]

			loss, err := trainBARTBatch(model, optimizer, batch)
			if err != nil {
				log.Printf("Error during training batch starting at index %d in epoch %d: %v\n", i, epoch, err)
				continue // Or handle error more strictly
			}
			totalLoss += loss
			numBatches++
		}
		avgLoss := totalLoss / float64(numBatches)
		fmt.Printf("Epoch %d/%d, Average Loss: %f\n", epoch+1, epochs, avgLoss)
	}

	log.Println("BART model training finished.")
	return nil
}

// trainBARTBatch performs a single training step on a batch of data.
func trainBARTBatch(model *SimplifiedBARTModel, optimizer *Optimizer, batch []struct {
	Input  string `json:"input"`
	Output string `json:"output"`
}) (float64, error) {
	// Zero out gradients from the previous step before starting a new one.
	optimizer.ZeroGrad()

	// 1. Prepare Input and Target Tensors
	batchSize := len(batch)
	seqLen := model.MaxSequenceLength

	flatInputData := make([]float64, batchSize*seqLen)
	flatTargetData := make([]float64, batchSize*seqLen)
	flatTargetIDsForLoss := make([]int, batchSize*seqLen)
	flatPosTagData := make([]float64, batchSize*seqLen)
	flatNerTagData := make([]float64, batchSize*seqLen)

	for i, pair := range batch {
		inputIDs, err := TokenizeAndConvertToIDs(pair.Input, model.Vocabulary, seqLen)
		if err != nil {
			return 0, fmt.Errorf("batch input tokenization failed for item %d: %w", i, err)
		}
		targetIDs, err := TokenizeAndConvertToIDs(pair.Output, model.Vocabulary, seqLen)
		if err != nil {
			return 0, fmt.Errorf("batch target tokenization failed for item %d: %w", i, err)
		}

		tags := tagger.Tagging(pair.Input)
		posTagToIDMap := postagger.PosTagToIDMap()
		nerTagToIDMap := nertagger.NerTagToIDMap()

		posTagIDs := make([]int, seqLen)
		nerTagIDs := make([]int, seqLen)

		for j := 0; j < seqLen; j++ {
			if j < len(tags.PosTag) {
				posTagIDs[j] = posTagToIDMap[tags.PosTag[j]]
			} else {
				posTagIDs[j] = 0 // Padding
			}
			if j < len(tags.NerTag) {
				nerTagIDs[j] = nerTagToIDMap[tags.NerTag[j]]
			} else {
				nerTagIDs[j] = 0 // Padding
			}
		}

		// Copy the tokenized data into the flat batch slices
		for j := 0; j < seqLen; j++ {
			flatIndex := i*seqLen + j
			flatInputData[flatIndex] = float64(inputIDs[j])
			flatTargetData[flatIndex] = float64(targetIDs[j])
			flatTargetIDsForLoss[flatIndex] = targetIDs[j]
			flatPosTagData[flatIndex] = float64(posTagIDs[j])
			flatNerTagData[flatIndex] = float64(nerTagIDs[j])
		}
	}

	// Create the final 2D tensors for the batch
	inputTensor := NewTensor(flatInputData, []int{batchSize, seqLen}, true)
	targetTensor := NewTensor(flatTargetData, []int{batchSize, seqLen}, true)
	posTagTensor := NewTensor(flatPosTagData, []int{batchSize, seqLen}, true)
	nerTagTensor := NewTensor(flatNerTagData, []int{batchSize, seqLen}, true)

	// 2. Forward Pass
	outputLogits, err := model.ForwardForTraining(inputTensor, targetTensor, posTagTensor, nerTagTensor)
	if err != nil {
		return 0, fmt.Errorf("model forward pass for training failed: %w", err)
	}

	// 3. Calculate Loss
	loss, err := CalculateCrossEntropyLoss(outputLogits, flatTargetIDsForLoss, model.Tokenizer.PadID)
	if err != nil {
		return 0, fmt.Errorf("loss calculation failed: %w", err)
	}

	// 4. Backpropagation
	outputGradient, err := CalculateCrossEntropyLossGradient(outputLogits, flatTargetIDsForLoss, model.Tokenizer.PadID)
	if err != nil {
		return 0, fmt.Errorf("loss gradient calculation failed: %w", err)
	}

	outputLogits.Grad = outputGradient

	outputLogits.Backward(outputLogits.Grad)

	// 5. Optimizer Step
	optimizer.Step()
	return loss, nil
}


// CalculateCrossEntropyLoss calculates the average cross-entropy loss between logits and target token IDs.
func CalculateCrossEntropyLoss(logits *Tensor, targetIDs []int, padID int) (float64, error) {
	if len(logits.Shape) != 2 {
		return 0, fmt.Errorf("logits tensor must be 2D (batch_size * seq_len, vocab_size), but got %dD", len(logits.Shape))
	}
	if logits.Shape[0] != len(targetIDs) {
		return 0, fmt.Errorf("mismatch between logits rows (%d) and target IDs (%d)", logits.Shape[0], len(targetIDs))
	}

	totalLoss := 0.0
	activeTokens := 0

	for i := 0; i < logits.Shape[0]; i++ {
		targetID := targetIDs[i]
		if targetID == padID {
			continue // Ignore padding tokens
		}
		activeTokens++

		// Softmax calculation for the current token
		maxLogit := math.Inf(-1)
		for j := 0; j < logits.Shape[1]; j++ {
			if logits.Data[i*logits.Shape[1]+j] > maxLogit {
				maxLogit = logits.Data[i*logits.Shape[1]+j]
			}
		}

		sumExp := 0.0
		for j := 0; j < logits.Shape[1]; j++ {
			sumExp += math.Exp(logits.Data[i*logits.Shape[1]+j] - maxLogit)
		}

		// Cross-entropy loss for the current token
		probOfCorrectToken := math.Exp(logits.Data[i*logits.Shape[1]+targetID] - maxLogit) / sumExp
		totalLoss -= math.Log(probOfCorrectToken)
	}

	if activeTokens == 0 {
		return 0.0, nil // Avoid division by zero
	}

	return totalLoss / float64(activeTokens), nil
}

// CalculateCrossEntropyLossGradient calculates the gradient of the cross-entropy loss with respect to the logits.
func CalculateCrossEntropyLossGradient(logits *Tensor, targetIDs []int, padID int) (*Tensor, error) {
	if len(logits.Shape) != 2 {
		return nil, fmt.Errorf("logits tensor must be 2D (batch_size * seq_len, vocab_size), but got %dD", len(logits.Shape))
	}
	if logits.Shape[0] != len(targetIDs) {
		return nil, fmt.Errorf("mismatch between logits rows (%d) and target IDs (%d)", logits.Shape[0], len(targetIDs))
	}

	grad := NewTensor(make([]float64, len(logits.Data)), logits.Shape, false)

	for i := 0; i < logits.Shape[0]; i++ {
		targetID := targetIDs[i]
		if targetID == padID {
			continue // No gradient for padding tokens
		}

		// Softmax calculation for the current token
		maxLogit := math.Inf(-1)
		for j := 0; j < logits.Shape[1]; j++ {
			if logits.Data[i*logits.Shape[1]+j] > maxLogit {
				maxLogit = logits.Data[i*logits.Shape[1]+j]
			}
		}
		sumExp := 0.0
		for j := 0; j < logits.Shape[1]; j++ {
			sumExp += math.Exp(logits.Data[i*logits.Shape[1]+j] - maxLogit)
		}

		// Gradient calculation
		for j := 0; j < logits.Shape[1]; j++ {
			prob := math.Exp(logits.Data[i*logits.Shape[1]+j] - maxLogit) / sumExp
			if j == targetID {
				grad.Data[i*logits.Shape[1]+j] = prob - 1
			} else {
				grad.Data[i*logits.Shape[1]+j] = prob
			}
		}
	}

	return grad, nil
}
