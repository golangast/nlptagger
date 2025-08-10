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
			model.TokenEmbedding = NewEmbeddingWithPretrained(model.VocabSize, model.TokenEmbedding.DimModel, model.Vocabulary, pretrainedEmbeddings)
		}
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

	for i, pair := range batch {
		inputIDs, err := TokenizeAndConvertToIDs(pair.Input, model.Vocabulary, seqLen)
		if err != nil {
			return 0, fmt.Errorf("batch input tokenization failed for item %d: %w", i, err)
		}
		targetIDs, err := TokenizeAndConvertToIDs(pair.Output, model.Vocabulary, seqLen)
		if err != nil {
			return 0, fmt.Errorf("batch target tokenization failed for item %d: %w", i, err)
		}

		// Copy the tokenized data into the flat batch slices
		for j := 0; j < seqLen; j++ {
			flatIndex := i*seqLen + j
			flatInputData[flatIndex] = float64(inputIDs[j])
			// For teacher forcing, the decoder input is the target sequence
			flatTargetData[flatIndex] = float64(targetIDs[j])
			// For loss calculation, we need the integer IDs
			flatTargetIDsForLoss[flatIndex] = targetIDs[j]
		}
	}

	// Create the final 2D tensors for the batch
	inputTensor := NewTensor(flatInputData, []int{batchSize, seqLen}, true)
	targetTensor := NewTensor(flatTargetData, []int{batchSize, seqLen}, true)

	// 2. Forward Pass
	// Pass both input and target tensors to the model for a full encoder-decoder pass.
	outputLogits, err := model.ForwardForTraining(inputTensor, targetTensor)
	if err != nil {
		return 0, fmt.Errorf("model forward pass for training failed: %w", err)
	}

	// 3. Calculate Loss
	// Compare the model's output logits with the target token IDs.
	loss, err := CalculateCrossEntropyLoss(outputLogits, flatTargetIDsForLoss, model.Tokenizer.PadID)
	if err != nil {
		return 0, fmt.Errorf("loss calculation failed: %w", err)
	}

	// 4. Backpropagation
	// Calculate the initial gradient of the loss with respect to the model's output logits.
	outputGradient, err := CalculateCrossEntropyLossGradient(outputLogits, flatTargetIDsForLoss, model.Tokenizer.PadID)
	if err != nil {
		return 0, fmt.Errorf("loss gradient calculation failed: %w", err)
	}

	// Initialize the gradient of the output logits with the calculated loss gradient.
	outputLogits.Grad = outputGradient

	fmt.Printf("\n--- trainBARTBatch: outputLogits.requiresGrad: %t ---\n", outputLogits.requiresGrad)
	fmt.Printf("--- trainBARTBatch: outputLogits.Grad is nil: %t ---\n", outputLogits.Grad == nil)
	if outputLogits.Grad != nil {
		fmt.Printf("--- trainBARTBatch: outputLogits.Grad.Shape: %v ---\n", outputLogits.Grad.Shape)
		for i := 0; i < int(math.Min(float64(len(outputLogits.Grad.Data)), 10)); i++ {
			fmt.Printf("--- trainBARTBatch: outputLogits.Grad.Data[%d]: %f ---\n", i, outputLogits.Grad.Data[i])
		}
	}

	// Start the backward pass from the output tensor. This will propagate gradients
	// through the entire computation graph.
	outputLogits.Backward(outputLogits.Grad)

	// 5. Optimizer Step
	optimizer.Step()
	return loss, nil
}

// CalculateCrossEntropyLoss calculates the average cross-entropy loss between logits and target token IDs.
func CalculateCrossEntropyLoss(logits *Tensor, targetIDs []int, padID int) (float64, error) {
	if logits == nil || logits.Data == nil {
		return 0, errors.New("logits tensor is nil or has no data")
	}
	if len(logits.Shape) != 3 {
		return 0, fmt.Errorf("expected logits shape [batch_size, sequence_length, vocab_size], but got %v", logits.Shape)
	}

	batchSize := logits.Shape[0]
	seqLength := logits.Shape[1]
	vocabSize := logits.Shape[2]

	if batchSize*seqLength != len(targetIDs) {
		return 0, fmt.Errorf("total number of logits (%d) does not match total number of target IDs (%d)", batchSize*seqLength, len(targetIDs))
	}

	// Apply Softmax to get probabilities
	probabilities, err := logits.Softmax(2) // Softmax along the vocab_size dimension
	if err != nil {
		return 0, fmt.Errorf("failed to apply softmax in loss calculation: %w", err)
	}

	totalLoss := 0.0
	numTokens := 0

	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLength; s++ {
			flatTargetIndex := b*seqLength + s
			targetID := targetIDs[flatTargetIndex]

			// Skip padding tokens in loss calculation
			if targetID == padID {
				continue
			}

			if targetID < 0 || targetID >= vocabSize {
				log.Printf("Warning: Invalid target token ID %d at batch %d, position %d. Skipping for loss calculation.\n", targetID, b, s)
				continue
			}

			// Calculate the flat index for the probability of the target token
			probFlatIndex := b*seqLength*vocabSize + s*vocabSize + targetID

			if probFlatIndex >= len(probabilities.Data) {
				return 0, fmt.Errorf("probability index out of bounds: %d for data length %d", probFlatIndex, len(probabilities.Data))
			}

			probability := probabilities.Data[probFlatIndex]

			// Calculate negative log likelihood. Add a small epsilon for numerical stability.
			if probability <= 0 {
				probability = 1e-9 // Add a small epsilon to prevent log(0)
			}
			totalLoss += -math.Log(probability)
			numTokens++
		}
	}

	if numTokens == 0 {
		return 0.0, nil // Avoid division by zero if no valid tokens were found
	}
	averageLoss := totalLoss / float64(numTokens)
	return averageLoss, nil
}

// CalculateCrossEntropyLossGradient calculates the gradient of the cross-entropy loss with respect to the logits.
// This is typically softmax(logits) - one_hot(target_token_ids).
func CalculateCrossEntropyLossGradient(logits *Tensor, targetIDs []int, padID int) (*Tensor, error) {
	if logits == nil || logits.Data == nil {
		return nil, errors.New("logits tensor is nil or has no data")
	}
	if len(logits.Shape) != 3 {
		return nil, fmt.Errorf("expected logits shape [batch_size, sequence_length, vocab_size], but got %v", logits.Shape)
	}

	batchSize := logits.Shape[0]
	seqLength := logits.Shape[1]
	vocabSize := logits.Shape[2]

	if batchSize*seqLength != len(targetIDs) {
		return nil, fmt.Errorf("total number of logits (%d) does not match total number of target IDs (%d)", batchSize*seqLength, len(targetIDs))
	}

	// Apply Softmax to get probabilities
	probabilities, err := logits.Softmax(2)
	if err != nil {
		return nil, fmt.Errorf("failed to apply softmax in gradient calculation: %w", err)
	}

	// The gradient of cross-entropy loss with softmax is `probabilities - one_hot_target`.
	// We can compute this directly without creating a large one-hot tensor.
	// The gradient is equal to the probabilities for all incorrect classes,
	// and `probability - 1` for the correct class.

	gradientData := make([]float64, len(logits.Data))
	copy(gradientData, probabilities.Data) // Start with gradient = probabilities

	// Now, subtract 1 from the positions corresponding to the target IDs.
	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLength; s++ {
			targetIndexInBatch := b*seqLength + s
			targetID := targetIDs[targetIndexInBatch]

			// Skip padding tokens in gradient calculation
			if targetID == padID {
				// Zero out gradients for all vocabulary tokens at this padded position
				for v := 0; v < vocabSize; v++ {
					gradFlatIndex := b*seqLength*vocabSize + s*vocabSize + v
					gradientData[gradFlatIndex] = 0
				}
				continue
			}

			if targetID < 0 || targetID >= vocabSize {
				log.Printf("Warning: Invalid target token ID %d at batch %d, position %d for gradient calculation. Skipping.\n", targetID, b, s)
				continue
			}

			// Calculate the flat index in the gradient tensor for the target ID.
			gradFlatIndex := b*seqLength*vocabSize + s*vocabSize + targetID
			gradientData[gradFlatIndex] -= 1.0
		}
	}

	outputGradient := NewTensor(gradientData, logits.Shape, true)

	return outputGradient, nil
}
