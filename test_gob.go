package main

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"log"

	"nlptagger/neural/moe"
	"nlptagger/neural/nn"
	"nlptagger/neural/tensor"
)

func main_test() {
	// Register types for gob
	gob.Register(&moe.IntentMoE{})
	gob.Register(&moe.MoELayer{})
	gob.Register(&moe.GatingNetwork{})
	gob.Register(&nn.Linear{})
	gob.Register((*moe.FeedForwardExpert)(nil)) // Register the concrete type for the Expert interface

	// Create a dummy IntentMoE model
	vocabSize := 100
	embeddingDim := 64
	numExperts := 2
	parentVocabSize := 10
	childVocabSize := 5
	sentenceVocabSize := 20
	maxAttentionHeads := 1

	model, err := moe.NewIntentMoE(vocabSize, embeddingDim, numExperts, parentVocabSize, childVocabSize, sentenceVocabSize, maxAttentionHeads, nil)
	if err != nil {
		log.Fatalf("Failed to create new IntentMoE model: %v", err)
	}

	fmt.Printf("Original model Encoder: %p\n", model.Encoder)
	if model.Encoder != nil {
		fmt.Printf("Original model Encoder GatingNetwork: %p\n", model.Encoder.GatingNetwork)
		fmt.Printf("Original model Encoder Experts count: %d\n", len(model.Encoder.Experts))
		if len(model.Encoder.Experts) > 0 {
			fmt.Printf("Original model Encoder Expert 0 type: %T\n", model.Encoder.Experts[0])
		}
	}

	// Encode the model
	var buffer bytes.Buffer
	encoder := gob.NewEncoder(&buffer)
	err = encoder.Encode(model)
	if err != nil {
		log.Fatalf("Failed to encode model: %v", err)
	}
	fmt.Println("Model encoded successfully.")

	// Decode the model
	decoder := gob.NewDecoder(&buffer)
	var loadedModel moe.IntentMoE
	err = decoder.Decode(&loadedModel)
	if err != nil {
		log.Fatalf("Failed to decode model: %v", err)
	}
	fmt.Println("Model decoded successfully.")

	fmt.Printf("Loaded model Encoder: %p\n", loadedModel.Encoder)
	if loadedModel.Encoder == nil {
		log.Fatal("Loaded IntentMoE model has a nil Encoder after decoding")
	}
	fmt.Printf("Loaded model Encoder GatingNetwork: %p\n", loadedModel.Encoder.GatingNetwork)
	if loadedModel.Encoder.GatingNetwork == nil {
		log.Fatal("Loaded IntentMoE model's Encoder has a nil GatingNetwork after decoding")
	}
	fmt.Printf("Loaded model Encoder Experts count: %d\n", len(loadedModel.Encoder.Experts))
	if len(loadedModel.Encoder.Experts) == 0 {
		log.Fatal("Loaded IntentMoE model's Encoder has no Experts after decoding")
	}
	fmt.Printf("Loaded model Encoder Expert 0 type: %T\n", loadedModel.Encoder.Experts[0])

	// Verify some parameters (optional)
	// For example, check if weights are not nil
	if loadedModel.Encoder.GatingNetwork.Linear.Weights == nil {
		log.Fatal("Loaded GatingNetwork Linear Weights are nil")
	}
	fmt.Println("Verification successful: GatingNetwork Linear Weights are not nil.")

	// Test forward pass with dummy input
	dummyInput := tensor.NewTensor([]int{1, 32}, make([]float64, 1*32), false)
	dummyInput2 := tensor.NewTensor([]int{1, 32}, make([]float64, 1*32), false)
	_, _, err = loadedModel.Forward(dummyInput, dummyInput2)
	if err != nil {
		log.Fatalf("Forward pass on loaded model failed: %v", err)
	}
	fmt.Println("Forward pass on loaded model successful.")
}
