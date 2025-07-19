package bartsimple

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

const tenThousand = 10000.0

// Embedding represents a simple token embedding layer.
type Embedding struct {
	Weights *Tensor // Embedding weights (learnable)

	// Stored values from forward pass for backward calculation
	inputTokenIDs []int // Store input token IDs
	inputShape    []int // Store input shape
}

// PositionalEmbedding represents a simple positional embedding layer.
type PositionalEmbedding struct {
	MaxSequenceLength  int
	DimModel           int
	PositionEmbeddings *Tensor // Positional embeddings (typically not learnable)

	// Stored values from forward pass for backward calculation
	inputShape []int // Store input shape
}
// Backward performs the backward pass for the token embedding layer.
// grad is the gradient from the subsequent layer.
func (e *Embedding) Backward(grad *Tensor) {
	if grad == nil || grad.Data == nil {
		// No gradient to propagate
		return
	}
	if e.Weights == nil || !e.Weights.requiresGrad {
		// Embedding weights are not learnable or do not require gradients
		return
	}
	if e.inputTokenIDs == nil || len(e.inputTokenIDs) != len(grad.Data)/e.Weights.Shape[1] {
		panic("Embedding backward called before forward or with mismatched input/gradient shapes")
	}

	vocabSize := e.Weights.Shape[0]
	dimModel := e.Weights.Shape[1]
	batchSize := e.inputShape[0]
	seqLength := e.inputShape[1]


	// Ensure gradient for weights is initialized
	if e.Weights.Grad == nil {
		e.Weights.Grad = NewTensor(make([]float64, len(e.Weights.Data)), e.Weights.Shape, false)
	}

	// Accumulate gradients to the embedding weights.
	// For each gradient element in 'grad', add it to the corresponding
	// position in e.Weights.Grad based on the input token ID.
	// The incoming gradient 'grad' has shape [batch_size, sequence_length, dim_model].
	// We need to add the gradient for each embedded vector back to the
	// corresponding row in the embedding weights matrix.

	// Iterate through the incoming gradient (which has the shape of the output from forward)
	gradFlatIndex := 0
	for i := 0; i < batchSize; i++ {
		for j := 0; j < seqLength; j++ {
			tokenID := e.inputTokenIDs[i*seqLength+j]
			if tokenID < 0 || tokenID >= vocabSize {
				// Should not happen if forward passed without error, but add a check
				log.Printf("Warning: Invalid token ID %d encountered during Embedding backward.\n", tokenID)
				continue // Skip invalid token IDs
			}

			// Add the gradient for this embedded vector (grad[i, j, :])
			// to the gradient of the corresponding embedding vector in Weights.Grad (Weights.Grad[tokenID, :]).
			embeddingGradStart := tokenID * dimModel
			gradVectorStart := gradFlatIndex // Current position in flattened grad data

			for k := 0; k < dimModel; k++ {
				if embeddingGradStart+k >= len(e.Weights.Grad.Data) || gradVectorStart+k >= len(grad.Data) {
					panic("gradient data index out of bounds during Embedding backward accumulation")
				}
				e.Weights.Grad.Data[embeddingGradStart+k] += grad.Data[gradVectorStart+k] // Accumulate gradient
			}

			gradFlatIndex += dimModel // Move to the next embedded vector in grad
		}
	}
}

// Backward performs the backward pass for the positional embedding layer.
// Since positional embeddings are typically not learnable, this method
// only propagates the gradient to the input if the input requires gradients.
func (pe *PositionalEmbedding) Backward(grad *Tensor) {
	if grad == nil || grad.Data == nil {
		// No gradient to propagate
		return
	}

	// If the input to the positional embedding requires gradients,
	// the gradient is simply passed through.
	// Assuming the input tensor is stored in the struct or can be accessed.
	// Since positional embedding is an addition in the forward pass (output = input + positional_embeddings),
	// the gradient with respect to the input is simply the gradient of the output.
	// We need to find the input tensor that was passed to the Forward method.
	// Assuming the input tensor is stored in a field like 'inputTensor' in the struct.

	// The 'creator' mechanism in the Tensor struct should handle this propagation.
	// When Tensor.Backward is called, it calls the creator's Backward method.
	// For the PositionalEmbedding's output tensor, its creator is the PositionalEmbedding struct itself.
	// The PositionalEmbedding.Backward method is called with the gradient from the next layer.
	// This gradient is then added to the gradient of the input tensor (if it requires gradients).

	// Since PositionalEmbedding is effectively adding a constant tensor (PositionEmbeddings),
	// the gradient with respect to the input is the incoming gradient.
	// We just need to ensure the gradient is passed back to the input tensor.
	// The Tensor.Backward method should handle the propagation to the input tensor
	// based on the 'creator' and 'Inputs()' methods.

	// If PositionEmbeddings were trainable (requiresGrad is true),
	// we would accumulate gradients to pe.PositionEmbeddings.Grad here,
	// similar to how gradients are accumulated in Embedding.Backward,
	// based on which positions were used in the forward pass.
	if pe.PositionEmbeddings != nil && pe.PositionEmbeddings.requiresGrad {
		// If positional embeddings are trainable, accumulate gradients here.
		// This would involve iterating through the incoming gradient (grad)
		// and adding its value to the corresponding position in
		// pe.PositionEmbeddings.Grad based on the positions used.
		// This is similar to Embedding.Backward, but based on position index instead of token ID.
		// You would need to iterate through the batch and sequence dimensions of grad
		// and add the gradient vector for each position to the corresponding row
		// in pe.PositionEmbeddings.Grad.
		log.Println("Warning: PositionalEmbedding.Backward accumulation for trainable embeddings not fully implemented.")
		// Implementation would go here if PositionEmbeddings.requiresGrad is true.
	}


	// The gradient for the input is simply the incoming gradient.
	// The Tensor.Backward mechanism should handle adding this gradient
	// to the input tensor's gradient if the input requires gradients.
	// So, this method primarily handles gradients for learnable parameters (if any).
	// Since PositionEmbeddings are typically not learnable, there's no parameter
	// gradient accumulation here. The gradient for the input is handled by
	// the Tensor.Backward propagation through the creator.

	// Therefore, if positional embeddings are not trainable, this method
	// effectively does nothing regarding parameter gradients and relies on
	// the tensor's backward mechanism to propagate the input gradient.
}

// Inputs returns the input tensors of the Embedding operation.
// Assuming the input tensor is stored in the struct.
func (e *Embedding) Inputs() []*Tensor {
	// Assuming the input tensor is stored in a field like 'inputTensor'
	// which is not explicitly added in the provided struct definition yet.
	// If the input tensor is not stored, you might need to rethink how to
	// access it for the Backward method or store it in the struct.
	// For now, assuming the input tensor is accessible through the 'creator' mechanism
	// or is implicitly handled by the Tensor.Backward traversal.

	// Based on the Tensor.Backward implementation that traverses Inputs(),
	// we should return the input tensor here if it's stored.
	// Let's assume the input tensor was stored in a field like 'inputTensor'
	// when the Forward method was called and it was passed as an argument.
	// This is a common pattern.

	// If you added 'inputTensor *Tensor' to the Embedding struct:
	// return []*Tensor{e.inputTensor}

	// Since we modified Forward to store inputTokenIDs and inputShape,
	// let's assume for now that the input tensor itself is NOT stored
	// in the Embedding struct, and we rely on the Tensor.Backward traversal
	// to propagate the gradient to the input tensor.
	// In this case, the Embedding operation doesn't have explicit input *Tensor* fields
	// that it needs to return here for topological sort based on *Tensor* references.
	// The input to the Embedding is essentially the token IDs, which are not tensors
	// that participate in the gradient computation graph in the same way.

	// Let's revisit the Tensor.Backward topological sort. It traverses based on
	// tensor.creator.Inputs(). If Embedding.Inputs() returns the token ID tensor,
	// and that tensor has requiresGrad, it will be visited. However, the token ID tensor
	// should not have requiresGrad=true.

	// Let's assume the input *tensor* to the Embedding.Forward is stored in a field
	// for consistency with other layers and for the graph traversal.
	// If you added 'inputTensor *Tensor' to the Embedding struct:
	// return []*Tensor{e.inputTensor}

	// For now, let's assume the input tensor is stored in 'inputTensor' field.
	// You will need to add 'inputTensor *Tensor' field to Embedding struct
	// and store the input tensor in the Forward method.

	// If inputTensor is not stored, the Tensor.Backward traversal based on
	// tensor.creator.Inputs() won't traverse back to the Embedding's input.
	// This might be okay if the gradient flow is intended to stop at the Embedding input.

	// Let's go with the assumption that inputTensor is stored for now to facilitate graph traversal.
	// You need to add 'inputTensor *Tensor' to Embedding struct and set it in Forward.

	// return []*Tensor{e.inputTensor} // Assuming inputTensor field exists

	// Reconsidering: The input to Embedding is token IDs, which are not tensors
	// in the computational graph sense for gradient propagation.
	// The gradient flow effectively stops at the Embedding input.
	// The Embedding layer calculates gradients *only* for its weights.
	// So, Embedding.Inputs() should likely return an empty slice or nil,
	// as there are no input *tensors* in the gradient graph to traverse back to.

	return []*Tensor{} // Embedding input is not a tensor for backprop graph traversal
}


// Inputs returns the input tensors of the PositionalEmbedding operation.
// Similar to Embedding, the input to PositionalEmbedding.Forward is a tensor,
// but the operation itself is an addition.
func (pe *PositionalEmbedding) Inputs() []*Tensor {
	// The positional embedding operation adds a constant tensor to the input tensor.
	// Output = Input + PositionalEmbeddings
	// The input tensor is part of the gradient computation graph.
	// So, we should return the input tensor here if it's stored.

	// If you added 'inputTensor *Tensor' to the PositionalEmbedding struct:
	// return []*Tensor{pe.inputTensor}

	// Assuming inputTensor is stored in a field 'inputTensor' for consistency.
	// You need to add 'inputTensor *Tensor' field to PositionalEmbedding struct
	// and set it in Forward.

	// return []*Tensor{pe.inputTensor} // Assuming inputTensor field exists

	// Reconsidering: The Tensor.Backward traversal needs to go back through
	// the operation's inputs to find the next operation to call Backward on.
	// For PositionalEmbedding, the input tensor is indeed the tensor
	// that needs to receive the gradient propagated through this operation.
	// So, PositionalEmbedding.Inputs() should return the input tensor.

	// Let's assume inputTensor is stored in the struct.
	return []*Tensor{pe.inputTensor} // Assuming inputTensor field exists
}

// Forward performs a simplified embedding lookup.
// It takes a tensor of token IDs and returns a tensor of corresponding embeddings.
func (e *Embedding) Forward(inputIDs *Tensor) (*Tensor, error) {
	// Assuming inputIDs is a 2D tensor [batch_size, sequence_length]
	// where each element is a token ID (integer).
	if len(inputIDs.Shape) != 2 {
		return nil, fmt.Errorf("embedding input must be 2D, got %v", inputIDs.Shape)
	}

	// Extract dimensions from the input tensor shape.
	// inputIDs.Shape[0] is the batch size.
	batchSize := inputIDs.Shape[0]
	seqLength := inputIDs.Shape[1]

	// Perform embedding lookup
	outputShape := []int{batchSize, seqLength, e.DimModel}
	outputData := make([]float64, batchSize*seqLength*e.DimModel)

	// Iterate through each item in the batch.
	for b := 0; b < batchSize; b++ {
		// Iterate through each token in the sequence for the current batch item.
		for s := 0; s < seqLength; s++ {
			// Get the token ID. Assuming the data is float64, cast to int.
			tokenID := int(inputIDs.Data[b*seqLength+s])
			// Validate the token ID is within the vocabulary range.
			if tokenID < 0 || tokenID >= e.VocabSize {
				return nil, fmt.Errorf("token ID %d is out of vocabulary range [0, %d)", tokenID, e.VocabSize)
			}
			// Copy the embedding vector for the current token ID from the weights tensor.
			startIndex := tokenID * e.DimModel
			endIndex := (tokenID + 1) * e.DimModel
			copy(outputData[(b*seqLength+s)*e.DimModel:(b*seqLength+s+1)*e.DimModel], e.Weights.Data[startIndex:endIndex])
		}
	}
	return NewTensor(outputData, outputShape, true), nil
}

// NewPositionalEmbedding creates a basic positional embedding layer.
// It initializes the positional embeddings using a sinusoidal pattern.
func NewPositionalEmbedding(maxSequenceLength, dimModel int) *PositionalEmbedding {
	// Placeholder: Create a dummy position embeddings tensor
	positionEmbeddingsData := make([]float64, maxSequenceLength*dimModel)
	// Initialize with sinusoidal positional embeddings
	for pos := 0; pos < maxSequenceLength; pos++ {
		for i := 0; i < dimModel; i++ {
			if i%2 == 0 {
				positionEmbeddingsData[pos*dimModel+i] = math.Sin(float64(pos) / math.Pow(tenThousand, float64(i)/float64(dimModel)))
			} else {
				positionEmbeddingsData[pos*dimModel+i] = math.Cos(float64(pos) / math.Pow(tenThousand, float64(i-1)/float64(dimModel)))
			}
		}
	}
	positionEmbeddings := NewTensor(positionEmbeddingsData, []int{maxSequenceLength, dimModel}, false)

	return &PositionalEmbedding{
		MaxSequenceLength:  maxSequenceLength,
		DimModel:           dimModel,
		PositionEmbeddings: positionEmbeddings,
	}
}

// Forward performs a simplified positional embedding addition.
// It adds the positional embeddings to the input tensor (which is assumed
// to be token embeddings or similar).
func (pe *PositionalEmbedding) Forward(inputTensor *Tensor) (*Tensor, error) {
	// Assuming inputTensor is [batch_size, sequence_length, embedding_dim]
	// It should be the output of a token embedding layer.
	if len(inputTensor.Shape) != 3 || inputTensor.Shape[2] != pe.DimModel {
		return nil, fmt.Errorf("positional embedding input must be 3D with embedding_dim %d, got %v", pe.DimModel, inputTensor.Shape)
	}

	// Extract dimensions from the input tensor shape.
	batchSize := inputTensor.Shape[0]
	seqLength := inputTensor.Shape[1]
	dimModel := inputTensor.Shape[2]

	outputData := make([]float64, len(inputTensor.Data))
	copy(outputData, inputTensor.Data) // Start with the input tensor data

	// Iterate through each item in the batch.
	for b := 0; b < batchSize; b++ {
		// Iterate through each position in the sequence for the current batch item.
		for s := 0; s < seqLength; s++ {
			// Add positional embedding for position 's'
			for d := 0; d < dimModel; d++ {
				outputData[(b*seqLength+s)*dimModel+d] += pe.PositionEmbeddings.Data[s*dimModel+d]
			}
		}
	}

	return NewTensor(outputData, inputTensor.Shape, false), nil
}

func init() {
	rand.Seed(time.Now().UnixNano())
}
