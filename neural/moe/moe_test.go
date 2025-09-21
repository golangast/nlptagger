package moe_test

import (
	"fmt"
	"testing"

	. "nlptagger/neural/moe"
	. "nlptagger/neural/tensor"
)

func TestMoELayer(t *testing.T) {
	inputDim := 10
	hiddenDim := 20
	outputDim := 10 // Output dimension of each expert should match input dimension for simple MoE
	numExperts := 4
	k := 2 // Select top 2 experts

	// Expert builder function
	expertBuilder := func(index int) (Expert, error) {
		return NewFeedForwardExpert(inputDim, hiddenDim, outputDim)
	}

	// Create MoE Layer
	moeLayer, err := NewMoELayer(inputDim, numExperts, k, expertBuilder)
	if err != nil {
		t.Fatalf("Failed to create MoE Layer: %v", err)
	}

	// Create a dummy input tensor
	batchSize := 2
	inputData := make([]float64, batchSize*inputDim)
	for i := range inputData {
		inputData[i] = float64(i) + 0.1 // Some dummy data
	}
	inputTensor := NewTensor(inputData, []int{batchSize, inputDim}, true) // requiresGrad = true for backprop

	// Forward pass
	output, err := moeLayer.Forward(inputTensor)
	if err != nil {
		t.Fatalf("MoE Layer forward pass failed: %v", err)
	}

	fmt.Printf("MoE Layer Output Shape: %v\n", output.Shape)
	fmt.Printf("MoE Layer Output Data (first 5): %v\n", output.Data[:min(5, len(output.Data))])

	// Create a dummy gradient for backward pass
	gradData := make([]float64, batchSize*outputDim)
	for i := range gradData {
		gradData[i] = 1.0 // Simple gradient of ones
	}
	gradTensor := NewTensor(gradData, []int{batchSize, outputDim}, false)

	// Backward pass
	err = moeLayer.Backward(gradTensor)
	if err != nil {
		t.Fatalf("MoE Layer backward pass failed: %v", err)
	}

	// Check if input gradients are populated
	if inputTensor.Grad == nil {
		t.Errorf("Input tensor gradient is nil after backward pass")
	}

	fmt.Printf("Input Tensor Gradient Shape: %v\n", inputTensor.Grad.Shape)
	fmt.Printf("Input Tensor Gradient Data (first 5): %v\n", inputTensor.Grad.Data[:min(5, len(inputTensor.Grad.Data))])

	// Optionally, check gradients of experts and gating network
	for i, expert := range moeLayer.Experts {
		params := expert.Parameters()
		for j, param := range params {
			if param.Grad == nil {
				// t.Errorf("Expert %d parameter %d gradient is nil", i, j)
			} else {
				fmt.Printf("Expert %d Parameter %d Gradient Data (first 5): %v\n", i, j, param.Grad.Data[:min(5, len(param.Grad.Data))])
			}
		}
	}

	gnParams := moeLayer.GatingNetwork.Parameters()
	for i, param := range gnParams {
		if param.Grad == nil {
			// t.Errorf("Gating Network parameter %d gradient is nil", i)
		} else {
			fmt.Printf("Gating Network Parameter %d Gradient Data (first 5): %v\n", i, param.Grad.Data[:min(5, len(param.Grad.Data))])
		}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
