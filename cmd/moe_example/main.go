package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	
	. "github.com/zendrulat/nlptagger/neural/moe"
	. "github.com/zendrulat/nlptagger/neural/tensor"
)



func main() {
	rand.Seed(time.Now().UnixNano())

	// Hyperparameters
	inputDim := 10
	hiddenDim := 20
	outputDim := 10 // Output dimension of experts, should match inputDim for simplicity
	numExperts := 4
	k := 2 // Select top 2 experts
	batchSize := 2

	// Create an expert builder function
	expertBuilder := func(expertIdx int) (Expert, error) {
		// Each expert can be a simple feed-forward network
		return NewFeedForwardExpert(inputDim, hiddenDim, outputDim)
	}

	// Create the MoE Layer
	moeLayer, err := NewMoELayer(inputDim, numExperts, k, expertBuilder)
	if err != nil {
		log.Fatalf("Failed to create MoE Layer: %v", err)
	}

	// Create dummy input tensor
	inputData := make([]float64, batchSize*inputDim)
	for i := range inputData {
		inputData[i] = rand.Float64() * 10 // Random input values
	}
	inputTensor := NewTensor([]int{batchSize, inputDim}, inputData, true) // requiresGrad = true for backprop

	fmt.Printf("Input Tensor Shape: %v\n", inputTensor.Shape)

	// Forward pass
	output, err := moeLayer.Forward(inputTensor)
	if err != nil {
		log.Fatalf("MoE Layer forward pass failed: %v", err)
	}

	fmt.Printf("Output Tensor Shape: %v\n", output.Shape)
	fmt.Printf("Output Tensor Data (first 5): %v\n", output.Data[:min(5, len(output.Data))])

	// Create a dummy gradient for backward pass
	gradData := make([]float64, len(output.Data))
	for i := range gradData {
		gradData[i] = 1.0 // Simple gradient of ones
	}
	gradTensor := NewTensor(output.Shape, gradData, false)

	// Backward pass
	fmt.Println("\nPerforming backward pass...")
	err = moeLayer.Backward(gradTensor)
	if err != nil {
		log.Fatalf("MoE Layer backward pass failed: %v", err)
	}

	fmt.Println("Backward pass completed.")

	// Check gradients of some parameters (e.g., gating network weights)
	fmt.Printf("\nGating Network Weights Grad (first 5): %v\n", moeLayer.GatingNetwork.Linear.Weights.Grad.Data[:min(5, len(moeLayer.GatingNetwork.Linear.Weights.Grad.Data))])

	// Check gradients of an expert's weights
	if len(moeLayer.Experts) > 0 {
		firstExpert := moeLayer.Experts[0].(*FeedForwardExpert)
		fmt.Printf("First Expert Linear1 Weights Grad (first 5): %v\n", firstExpert.Layer1.Weights.Grad.Data[:min(5, len(firstExpert.Layer1.Weights.Grad.Data))])
	}

	// Check input tensor gradient
	fmt.Printf("Input Tensor Grad (first 5): %v\n", inputTensor.Grad.Data[:min(5, len(inputTensor.Grad.Data))])

	fmt.Println("MoE Layer example finished successfully.")
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
