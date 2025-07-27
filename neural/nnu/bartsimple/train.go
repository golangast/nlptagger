package bartsimple

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"os"
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

	bytes, err := ioutil.ReadAll(file)
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
func TrainBARTModel(model *SimplifiedBARTModel, data *BARTTrainingData, epochs int, learningRate float64) error {
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
		return errors.New("model token embedding is not initialized")
	}
	if model.PositionalEmbedding == nil {
		return errors.New("model positional embedding is not initialized")
	}
	if model.Encoder == nil || model.Decoder == nil {
		return errors.New("model encoder or decoder is not initialized")
	}
	if model.OutputLinear == nil {
		return errors.New("model output linear layer is not initialized")
	}

	log.Printf("Starting BART model training for %d epochs with learning rate %f\n", epochs, learningRate)

	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		for i, sentencePair := range data.Sentences {
			loss, err := trainBARTStep(model, sentencePair.Input, sentencePair.Output)
			if err != nil {
				log.Printf("Error during training step %d in epoch %d: %v\n", i, epoch, err)
				continue // Or handle error more strictly
			}
			totalLoss += loss
		}
		avgLoss := totalLoss / float64(len(data.Sentences))
		log.Printf("Epoch %d/%d, Average Loss: %f\n", epoch+1, epochs, avgLoss)
	}

	log.Println("BART model training finished.")
	return nil
}

// CalculateCrossEntropyLoss calculates the cross-entropy loss between logits and target token IDs.
// This is a basic implementation and assumes the logits are flat [batch*seq_len, vocab_size]
// and targetIDs are flat [batch*seq_len].
// It returns the average loss per token.
// trainBARTStep performs a single training step (forward pass, loss calculation, placeholder for backprop).
func trainBARTStep(model *SimplifiedBARTModel, inputSentence, targetSentence string) (float64, error) {
	// 1. Prepare Input and Target Tensors
	// Use TokenizeAndConvertToIDs to handle tokenization, special tokens, and padding/truncation.
	inputTokenIDs, err := TokenizeAndConvertToIDs(inputSentence, model.Vocabulary, model.MaxSequenceLength)
	if err != nil {
		return 0, fmt.Errorf("input tokenization failed: %w", err)
	}
	targetTokenIDs, err := TokenizeAndConvertToIDs(targetSentence, model.Vocabulary, model.MaxSequenceLength)
	if err != nil {
		return 0, fmt.Errorf("target tokenization failed: %w", err)
	}

	// Convert token ID slices to 2D tensors [batch_size, sequence_length]
	inputData := make([]float64, len(inputTokenIDs))
	for i, id := range inputTokenIDs {
		inputData[i] = float64(id)
	}
	inputTensor := NewTensor(inputData, []int{1, len(inputTokenIDs)}, true)

	targetData := make([]float64, len(targetTokenIDs))
	for i, id := range targetTokenIDs {
		targetData[i] = float64(id)
	}
	// The target tensor is used as input to the decoder during teacher forcing.
	targetTensor := NewTensor(targetData, []int{1, len(targetTokenIDs)}, true)

	// 2. Forward Pass
	// Pass both input and target tensors to the model for a full encoder-decoder pass.
	outputLogits, err := model.ForwardForTraining(inputTensor, targetTensor)
	if err != nil {
		return 0, fmt.Errorf("model forward pass for training failed: %w", err)
	}

	// 3. Calculate Loss
	// Compare the model's output logits with the target token IDs.
	loss, err := CalculateCrossEntropyLoss(outputLogits, targetTokenIDs)
	if err != nil {
		return 0, fmt.Errorf("loss calculation failed: %w", err)
	}

	// 4. Backpropagation
	// Calculate the initial gradient of the loss with respect to the model's output logits.
	outputGradient, err := CalculateCrossEntropyLossGradient(outputLogits, targetTokenIDs)
	if err != nil {
		return 0, fmt.Errorf("loss gradient calculation failed: %w", err)
	}

	// Start the backward pass from the output tensor. This will propagate gradients
	// through the entire computation graph.
	outputLogits.Backward(outputGradient)

	// 5. Optimizer Step (Placeholder)
	// In a full implementation, you would have an optimizer instance that updates
	// all learnable parameters (weights and biases) in the model.
	// e.g., optimizer.Step()

	return loss, nil
}

// CalculateCrossEntropyLoss calculates the average cross-entropy loss between logits and target token IDs.
func CalculateCrossEntropyLoss(logits *Tensor, targetIDs []int) (float64, error) {
	if logits == nil || logits.Data == nil {
		return 0, errors.New("logits tensor is nil or has no data")
	}
	if len(logits.Shape) != 3 {
		return 0, fmt.Errorf("expected logits shape [batch_size, sequence_length, vocab_size], but got %v", logits.Shape)
	}

	batchSize := logits.Shape[0]
	seqLength := logits.Shape[1]
	vocabSize := logits.Shape[2]

	if batchSize != 1 {
		// This simplified implementation assumes a batch size of 1.
		// You'll need to extend this for batching.
		return 0, errors.New("CalculateCrossEntropyLoss currently only supports batch size 1")
	}

	if seqLength != len(targetIDs) {
		return 0, errors.New("logits sequence length (%d) does not match target IDs length (%d)")
	}

	// Apply Softmax to get probabilities
	// Assuming Softmax operates on the last dimension (vocab_size)
	probabilities, err := logits.Softmax(2) // Assuming Softmax method exists and takes axis
	if err != nil {
		return 0, fmt.Errorf("failed to apply softmax in loss calculation: %w", err)
	}

	totalLoss := 0.0
	// Iterate through the sequence length
	for s := 0; s < seqLength; s++ {
		targetID := targetIDs[s]
		if targetID < 0 || targetID >= vocabSize {
			// Handle invalid target IDs (e.g., unknown tokens) - might skip or use a special token
			log.Printf("Warning: Invalid target token ID %d at position %d. Skipping for loss calculation.\n", targetID, s)
			continue // Skip invalid target IDs
		}

		// Calculate the flat index for the probability of the target token
		// in the flattened probabilities data slice (assuming batch size 1)
		probFlatIndex := s*vocabSize + targetID

		if probFlatIndex >= len(probabilities.Data) {
			return 0, fmt.Errorf("probability index out of bounds: %d for data length %d", probFlatIndex, len(probabilities.Data))
		}

		probability := probabilities.Data[probFlatIndex]

		// Calculate negative log likelihood. Add a small epsilon for numerical stability if needed.
		if probability <= 0 {
			// Handle cases where probability is zero or negative (due to numerical issues)
			log.Printf("Warning: Probability is non-positive (%f) for target ID %d at position %d. Adding small value for log.", probability, targetID, s)
			probability = 1e-9 // Add a small epsilon
		}
		negativeLogLikelihood := -math.Log(probability)
		totalLoss += negativeLogLikelihood
	}

	// Calculate average loss over the sequence
	averageLoss := totalLoss / float64(seqLength)

	return averageLoss, nil
}

// CalculateCrossEntropyLossGradient calculates the gradient of the cross-entropy loss with respect to the logits.
// This is typically softmax(logits) - one_hot(target_token_ids).
func CalculateCrossEntropyLossGradient(logits *Tensor, targetIDs []int) (*Tensor, error) {
	if logits == nil || logits.Data == nil {
		return nil, errors.New("logits tensor is nil or has no data")
	}
	if len(logits.Shape) != 3 {
		return nil, fmt.Errorf("expected logits shape [batch_size, sequence_length, vocab_size], but got %v", logits.Shape)
	}

	batchSize := logits.Shape[0]
	seqLength := logits.Shape[1]
	vocabSize := logits.Shape[2]

	if batchSize != 1 {
		// This simplified implementation assumes a batch size of 1.
		// You'll need to extend this for batching.
		return nil, errors.New("CalculateCrossEntropyLossGradient currently only supports batch size 1")
	}

	if seqLength != len(targetIDs) {
		// For this simplified gradient calculation, assuming sequence lengths match.
		// You might need to handle padding and different sequence lengths.
		return nil, errors.New("logits sequence length (%d) does not match target IDs length (%d)")
	}

	// Apply Softmax to get probabilities
	// Assuming Softmax operates on the last dimension (vocab_size)
	probabilities, err := logits.Softmax(2) // Assuming Softmax method exists and takes axis
	if err != nil {
		return nil, fmt.Errorf("failed to apply softmax in gradient calculation: %w", err)
	}

	// Create a one-hot encoded tensor from target IDs
	// This tensor will have the same shape as the logits.
	// TODO: Implement CreateOneHotTensor function or similar logic
	// This requires creating a new tensor of shape [batch_size, sequence_length, vocab_size]
	// and setting the element to 1.0 at the target token ID position for each step in the sequence.
	// Example placeholder for one-hot tensor:
	oneHotTarget := NewTensor(make([]float64, len(logits.Data)), logits.Shape, false) // Gradients of target are not needed

	// Fill the one-hot tensor (manual for batch size 1)
	for s := 0; s < seqLength; s++ {
		targetID := targetIDs[s]
		if targetID < 0 || targetID >= vocabSize {
			// Handle invalid target IDs - might skip or use a special token
			log.Printf("Warning: Invalid target token ID %d at position %d for one-hot encoding.\n", targetID, s)
			continue // Skip invalid target IDs
		}
		// Set the element to 1.0 at the target ID position
		oneHotFlatIndex := s*vocabSize + targetID
		if oneHotFlatIndex >= len(oneHotTarget.Data) {
			return nil, fmt.Errorf("one-hot index out of bounds: %d for data length %d", oneHotFlatIndex, len(oneHotTarget.Data))
		}
		oneHotTarget.Data[oneHotFlatIndex] = 1.0
	}

	// Subtract the one-hot tensor from the softmax output element-wise
	// Assuming you have an element-wise subtraction operation for tensors.
	// If not, you'll need to implement one or perform subtraction manually.
	// Placeholder for subtraction:
	gradientData := make([]float64, len(logits.Data))
	for i := range gradientData {
		gradientData[i] = probabilities.Data[i] - oneHotTarget.Data[i]
	}
	outputGradient := NewTensor(gradientData, logits.Shape, false) // Gradient tensor itself does not require gradients

	return outputGradient, nil
}
