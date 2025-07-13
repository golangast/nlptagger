package bartsimple

import (
	"errors"
	"fmt"
	"math"
)

// Linear represents a linear layer (fully connected layer).
type Linear struct {
	Weights *Tensor
	Biases  *Tensor
}

// NewLinear creates a new Linear layer with random weights and zero biases.
func NewLinear(inputDim, outputDim int) (*Linear, error) {
	// Initialize weights with random values (you might want a better initialization later)
	weightsData := make([]float64, inputDim*outputDim)
	for i := range weightsData {
		weightsData[i] = float64(i) * 0.01 // Simple non-zero initialization
	}
	weights := NewTensor(weightsData, []int{inputDim, outputDim})

	// Initialize biases with zeros
	biasesData := make([]float64, outputDim)
	biases := NewTensor(biasesData, []int{outputDim})

	return &Linear{Weights: weights, Biases: biases}, nil
}

// Forward performs the forward pass of the Linear layer.
func (l *Linear) Forward(input *Tensor) (*Tensor, error) {
	// Assuming input is 2D [batch_size, input_dim] or 3D [batch_size, sequence_length, input_dim]
	var output *Tensor
	var err error

	switch len(input.Shape) {
	case 2:
		// Handle 2D input: [batch_size, input_dim]
		// Perform matrix multiplication: [batch_size, input_dim] @ [input_dim, output_dim]
		output, err = input.MatMul(l.Weights)
		if err != nil {
			return nil, fmt.Errorf("linear layer 2D matrix multiplication failed: %w", err)
		}

	case 3:
		// Handle 3D input: [batch_size, sequence_length, input_dim]
		batchSize := input.Shape[0]
		seqLength := input.Shape[1]
		inputDim := input.Shape[2]
		outputDim := l.Weights.Shape[1]

		// Reshape input for batch matrix multiplication
		reshapedInputData := make([]float64, batchSize*seqLength*inputDim)
		copy(reshapedInputData, input.Data) // Create a copy to avoid modifying original
		reshapedInput := NewTensor(reshapedInputData, []int{batchSize * seqLength, inputDim})

		// Perform matrix multiplication: [batch_size * sequence_length, input_dim] @ [input_dim, output_dim]
		output2D, err := reshapedInput.MatMul(l.Weights)
		if err != nil {
			return nil, fmt.Errorf("linear layer 3D matrix multiplication failed: %w", err)
		}

		// Reshape output back to 3D
		outputData := make([]float64, batchSize*seqLength*outputDim)
		copy(outputData, output2D.Data) // Create a copy
		output = NewTensor(outputData, []int{batchSize, seqLength, outputDim})

	default:
		return nil, fmt.Errorf("linear layer only supports 2D or 3D input, got %d dimensions", len(input.Shape))
	}

	// Add biases if they exist
	if l.Biases != nil {
		// AddWithBroadcast handles broadcasting biases
		output, err = output.AddWithBroadcast(l.Biases)
		if err != nil {
			return nil, fmt.Errorf("linear layer bias addition failed: %w", err)
		}
	}

	return output, nil
}

// LayerNormalization represents a layer normalization layer.
type LayerNormalization struct {
	Gamma   *Tensor // Scale parameter
	Beta    *Tensor // Shift parameter
	Epsilon float64 // Small value to prevent division by zero
}

// NewLayerNormalization creates a new LayerNormalization layer.
func NewLayerNormalization(dimModel int) *LayerNormalization {
	// Initialize gamma to ones and beta to zeros
	gammaData := make([]float64, dimModel)
	betaData := make([]float64, dimModel)
	for i := range gammaData {
		gammaData[i] = 1.0 // Initialize gamma to 1s
		betaData[i] = 0.0  // Initialize beta to 0s
	}
	gamma := NewTensor(gammaData, []int{dimModel})
	beta := NewTensor(betaData, []int{dimModel})

	return &LayerNormalization{
		Gamma:   gamma,
		Beta:    beta,
		Epsilon: 1e-6, // Small epsilon value
	}
}

// Forward performs the forward pass of the LayerNormalization layer.
// It normalizes the input tensor along the last dimension.
func (ln *LayerNormalization) Forward(input *Tensor) (*Tensor, error) {
	if len(input.Shape) < 1 {
		return nil, fmt.Errorf("layer normalization input must have at least one dimension")
	}

	// Assuming normalization is performed along the last dimension (embedding_dim)
	axis := len(input.Shape) - 1
	axisSize := input.Shape[axis]
	batchSize := 1
	for i := 0; i < axis; i++ {
		batchSize *= input.Shape[i]
	}

	// Ensure input data size is correct
	if len(input.Data) != batchSize*axisSize {
		return nil, fmt.Errorf("layer normalization input data size mismatch: expected %d, got %d", batchSize*axisSize, len(input.Data))
	}

	normalizedData := make([]float64, len(input.Data))

	for i := 0; i < batchSize; i++ {
		// Calculate mean and variance for the current batch element
		sum := 0.0
		for j := 0; j < axisSize; j++ {
			sum += input.Data[i*axisSize+j]
		}
		mean := sum / float64(axisSize)

		varianceSum := 0.0
		for j := 0; j < axisSize; j++ {
			diff := input.Data[i*axisSize+j] - mean
			varianceSum += diff * diff
		}
		variance := varianceSum / float64(axisSize)

		// Normalize and apply gamma and beta
		stdDev := math.Sqrt(variance + ln.Epsilon)
		for j := 0; j < axisSize; j++ {
			normalizedData[i*axisSize+j] = (input.Data[i*axisSize+j]-mean)/stdDev*ln.Gamma.Data[j] + ln.Beta.Data[j]
		}

	}
	// Debug print after normalization

	return NewTensor(normalizedData, input.Shape), nil
}

// MultiHeadAttention represents a multi-head attention layer.
type MultiHeadAttention struct {
	NumHeads     int
	DimModel     int
	HeadDim      int
	QueryLinear  *Linear
	KeyLinear    *Linear
	ValueLinear  *Linear
	OutputLinear *Linear
}

func NewMultiHeadAttention(dimModel, numHeads, numKVHeads int) (*MultiHeadAttention, error) {
	if dimModel%numHeads != 0 {
		return nil, fmt.Errorf("dimModel (%d) must be divisible by numHeads (%d)", dimModel, numHeads)
	}
	headDim := dimModel / numHeads

	queryLinear, err := NewLinear(dimModel, dimModel) // Output dim is dimModel (numHeads * headDim)
	if err != nil {
		return nil, fmt.Errorf("failed to create query linear layer: %w", err)
	}
	keyLinear, err := NewLinear(dimModel, dimModel) // Output dim is dimModel (numKVHeads * headDim in general, but here assuming numHeads == numKVHeads)
	if err != nil {
		return nil, fmt.Errorf("failed to create key linear layer: %w", err)
	}
	valueLinear, err := NewLinear(dimModel, dimModel) // Output dim is dimModel (numKVHeads * headDim in general, but here assuming numHeads == numKVHeads)
	if err != nil {
		return nil, fmt.Errorf("failed to create value linear layer: %w", err)
	}
	outputLinear, err := NewLinear(dimModel, dimModel) // Output dim is dimModel
	if err != nil {
		return nil, fmt.Errorf("failed to create output linear layer: %w", err)
	}

	return &MultiHeadAttention{
		NumHeads:     numHeads,
		DimModel:     dimModel,
		HeadDim:      headDim, // Store headDim
		QueryLinear:  queryLinear,
		KeyLinear:    keyLinear,
		ValueLinear:  valueLinear,
		OutputLinear: outputLinear,
	}, nil
}

// Forward performs the forward pass of the MultiHeadAttention layer.
// This is a simplified version without caching or masks.
func (mha *MultiHeadAttention) Forward(query, key, value *Tensor, mask *Tensor) (*Tensor, error) {
	// Assume input shapes are [batch_size, sequence_length, dim_model]

	batchSize := query.Shape[0]
	qSeqLength := query.Shape[1]
	kvSeqLength := key.Shape[1] // Key and Value should have the same sequence length

	// Apply linear transformations to get Q, K, V
	q, err := mha.QueryLinear.Forward(query)
	if err != nil {
		return nil, fmt.Errorf("multihead attention query linear failed: %w", err)
	}
	k, err := mha.KeyLinear.Forward(key)
	if err != nil {
		return nil, fmt.Errorf("multihead attention key linear failed: %w", err)
	}

	v, err := mha.ValueLinear.Forward(value)
	if err != nil {
		return nil, fmt.Errorf("multihead attention value linear failed: %w", err)
	}

	// Reshape Q, K, V for multi-head attention
	// [batch_size, sequence_length, dim_model] -> [batch_size, num_heads, sequence_length, head_dim]
	qReshaped, err := q.Reshape([]int{batchSize, qSeqLength, mha.NumHeads, mha.HeadDim})
	if err != nil {
		return nil, fmt.Errorf("failed to reshape query tensor: %w", err)
	}
	qTransposed, err := qReshaped.Transpose(1, 2) // Transpose to [batch_size, num_heads, q_seq_length, head_dim]
	if err != nil {
		return nil, fmt.Errorf("failed to transpose query tensor: %w", err)
	}

	kReshaped, err := k.Reshape([]int{batchSize, kvSeqLength, mha.NumHeads, mha.HeadDim})
	if err != nil {
		return nil, fmt.Errorf("failed to reshape key tensor: %w", err)
	}

	kTransposed, err := kReshaped.Transpose(1, 2) // Transpose to [batch_size, num_heads, kv_seq_length, head_dim]
	if err != nil {
		return nil, fmt.Errorf("failed to transpose key tensor: %w", err)
	}

	vReshaped, err := v.Reshape([]int{batchSize, kvSeqLength, mha.NumHeads, mha.HeadDim})
	if err != nil {
		return nil, fmt.Errorf("failed to reshape value tensor: %w", err)
	}

	vTransposed, err := vReshaped.Transpose(1, 2) // Transpose to [batch_size, num_heads, kv_seq_length, head_dim]
	if err != nil {
		return nil, fmt.Errorf("failed to transpose value tensor: %w", err)
	}

	// Calculate attention scores: Q @ K^T
	// K^T will have shape [batch_size, num_heads, head_dim, kv_seq_length]
	kT_Transposed, err := kTransposed.Transpose(2, 3)
	if err != nil {
		return nil, fmt.Errorf("failed to transpose key tensor for multiplication: %w", err)
	}

	attentionScores, err := qTransposed.MatMul(kT_Transposed)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate attention scores (Q@K^T): %w", err)
	}

	// Temporary check for attentionScores data
	isAttentionScoresAllZeros := true
	for _, val := range attentionScores.Data {
		if val != 0 {
			isAttentionScoresAllZeros = false
			break
		}
	}
	if isAttentionScoresAllZeros {
		fmt.Println("Debug: attentionScores are all zeros before ScalarMul")
	}
	// End of temporary check

	// Scale attention scores
	scale := 1.0 / math.Sqrt(float64(mha.HeadDim))
	scaledAttentionScores, err := attentionScores.ScalarMul(scale)
	if err != nil {
		return nil, fmt.Errorf("failed to scale attention scores: %w", err)
	}

	// Apply mask (if provided) - Simplified, just add large negative to masked positions
	if mask != nil {
		maskedAttentionScores, err := scaledAttentionScores.AddWithBroadcast(mask)
		if err != nil {
			return nil, fmt.Errorf("failed to apply mask to attention scores: %w", err)
		}
		scaledAttentionScores = maskedAttentionScores
	}

	// Apply Softmax to get attention weights
	attentionWeights, err := scaledAttentionScores.Softmax(len(scaledAttentionScores.Shape) - 1) // Softmax along the last dimension
	if err != nil {
		return nil, fmt.Errorf("failed to apply softmax to attention scores: %w", err)
	}

	// Apply attention weights to Value: Attention Weights @ V
	contextLayer, err := attentionWeights.MatMul(vTransposed)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate context layer (Attention@V): %w", err)
	}

	// Concatenate heads and apply final linear layer
	// [batch_size, num_heads, q_seq_length, head_dim] -> [batch_size, q_seq_length, num_heads * head_dim] (which is dim_model)
	contextLayerTransposed, err := contextLayer.Transpose(1, 2) // Transpose back to [batch_size, q_seq_length, num_heads, head_dim]
	if err != nil {
		return nil, fmt.Errorf("failed to transpose context layer: %w", err)
	}

	// Reshape to [batch_size, q_seq_length, dim_model]
	outputShape := []int{batchSize, qSeqLength, mha.DimModel}
	contextLayerReshaped, err := contextLayerTransposed.Reshape(outputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to reshape context layer: %w", err)
	}

	// Apply output linear layer
	output, err := mha.OutputLinear.Forward(contextLayerReshaped)
	if err != nil {
		return nil, fmt.Errorf("multihead attention output linear failed: %w", err)
	}

	return output, nil
}

// FeedForward represents a feed-forward network.
type FeedForward struct {
	Linear1 *Linear
	Linear2 *Linear
}

// NewFeedForward creates a new FeedForward layer.
func NewFeedForward(dimModel int) (*FeedForward, error) {

	// Inner dimension is typically 4 times dimModel
	innerDim := dimModel * 4

	linear1, err := NewLinear(dimModel, innerDim)
	if err != nil {
		return nil, fmt.Errorf("failed to create feed-forward linear1 layer: %w", err)
	}
	linear2, err := NewLinear(innerDim, dimModel)
	if err != nil {
		return nil, fmt.Errorf("failed to create feed-forward linear2 layer: %w", err)
	}

	return &FeedForward{
		Linear1: linear1,
		Linear2: linear2,
	}, nil
}

// Forward performs the forward pass of the FeedForward layer.
func (ff *FeedForward) Forward(input *Tensor) (*Tensor, error) {

	// Apply the first linear transformation
	hidden, err := ff.Linear1.Forward(input)
	if err != nil {
		return nil, fmt.Errorf("feed-forward first linear failed: %w", err)
	}

	// Apply ReLU activation manually
	activatedHiddenData := make([]float64, len(hidden.Data))
	for i, val := range hidden.Data {
		activatedHiddenData[i] = math.Max(0, val)
	}
	activatedHidden := NewTensor(activatedHiddenData, hidden.Shape)

	// Apply the second linear transformation
	output, err := ff.Linear2.Forward(activatedHidden)
	if err != nil {
		return nil, fmt.Errorf("feed-forward second linear failed: %w", err)
	}

	return output, nil
}

// MultiHeadCrossAttention represents a multi-head cross-attention layer.
type MultiHeadCrossAttention struct {
	NumQHeads  int // Number of query heads
	NumKVHeads int // Number of key/value heads
	DimModel   int
	DimKVHeads int // Dimension per key/value head

	QueryLinear  *Linear
	KeyLinear    *Linear // Linear layer for keys from encoder output
	ValueLinear  *Linear // Linear layer for values from encoder output
	OutputLinear *Linear
}

// NewMultiHeadCrossAttention creates a new MultiHeadCrossAttention layer.
func NewMultiHeadCrossAttention(dimModel, numQHeads, numKVHeads int) (*MultiHeadCrossAttention, error) {
	if dimModel%numQHeads != 0 {
		return nil, fmt.Errorf("dimModel (%d) must be divisible by numQHeads (%d)", dimModel, numQHeads)
	}
	if dimModel%numKVHeads != 0 {
		return nil, fmt.Errorf("dimModel (%d) must be divisible by numKVHeads (%d)", dimModel, numKVHeads)
	}
	dimKVHeads := dimModel / numKVHeads

	queryLinear, err := NewLinear(dimModel, dimModel) // Input dim is dimModel (from decoder), output dim is dimModel (for q)
	if err != nil {
		return nil, fmt.Errorf("failed to create cross-attention query linear layer: %w", err)
	}
	keyLinear, err := NewLinear(dimModel, dimModel) // Input dim is dimModel (from encoder), output dim is dimModel (for k)
	if err != nil {
		return nil, fmt.Errorf("failed to create cross-attention key linear layer: %w", err)
	}
	valueLinear, err := NewLinear(dimModel, dimModel) // Input dim is dimModel (from encoder), output dim is dimModel (for v)
	if err != nil {
		return nil, fmt.Errorf("failed to create cross-attention value linear layer: %w", err)
	}
	outputLinear, err := NewLinear(dimModel, dimModel) // Input dim is dimModel, output dim is dimModel
	if err != nil {
		return nil, fmt.Errorf("failed to create cross-attention output linear layer: %w", err)
	}

	return &MultiHeadCrossAttention{
		NumQHeads:    numQHeads,
		NumKVHeads:   numKVHeads,
		DimModel:     dimModel,
		DimKVHeads:   dimKVHeads, // Store dim per KV head
		QueryLinear:  queryLinear,
		KeyLinear:    keyLinear,
		ValueLinear:  valueLinear,
		OutputLinear: outputLinear,
	}, nil
}

// Forward performs the forward pass of the MultiHeadCrossAttention layer.
// query: Input from the decoder layer (shape: [batch_size, target_sequence_length, dim_model]).
// key/value: Input from the encoder output (shape: [batch_size, source_sequence_length, dim_model]).
// mask: Optional mask for attention (e.g., padding mask for encoder output).
func (mha *MultiHeadCrossAttention) Forward(query, key, value *Tensor, mask *Tensor) (*Tensor, error) {
	batchSize := query.Shape[0]
	qSeqLength := query.Shape[1]
	kvSeqLength := key.Shape[1] // Key and Value should have the same sequence length

	// Apply linear transformations to get Q, K, V
	q, err := mha.QueryLinear.Forward(query) // Q from decoder input
	if err != nil {
		return nil, fmt.Errorf("cross-attention query linear failed: %w", err)
	}
	k, err := mha.KeyLinear.Forward(key) // K from encoder output
	if err != nil {
		return nil, fmt.Errorf("cross-attention key linear failed: %w", err)
	}
	v, err := mha.ValueLinear.Forward(value) // V from encoder output
	if err != nil {
		return nil, fmt.Errorf("cross-attention value linear failed: %w", err)
	}
	// Reshape Q, K, V for multi-head attention
	// Q shape: [batch_size, num_q_heads, q_seq_length, head_dim]
	qReshaped, err := q.Reshape([]int{batchSize, qSeqLength, mha.NumQHeads, mha.DimModel / mha.NumQHeads}) // Use DimModel/NumQHeads
	if err != nil {
		return nil, fmt.Errorf("failed to reshape cross-attention query tensor: %w", err)
	}

	qTransposed, err := qReshaped.Transpose(1, 2)
	if err != nil {
		return nil, fmt.Errorf("failed to transpose cross-attention query tensor: %w", err)
	}

	// K, V shapes: [batch_size, num_kv_heads, kv_seq_length, head_dim]
	kReshaped, err := k.Reshape([]int{batchSize, kvSeqLength, mha.NumKVHeads, mha.DimKVHeads}) // Use DimKVHeads
	if err != nil {
		return nil, fmt.Errorf("failed to reshape cross-attention key tensor: %w", err)
	}
	kTransposed, err := kReshaped.Transpose(1, 2)
	if err != nil {
		return nil, fmt.Errorf("failed to transpose cross-attention key tensor: %w", err)
	}

	vReshaped, err := v.Reshape([]int{batchSize, kvSeqLength, mha.NumKVHeads, mha.DimKVHeads}) // Use DimKVHeads
	if err != nil {
		return nil, fmt.Errorf("failed to reshape cross-attention value tensor: %w", err)
	}
	vTransposed, err := vReshaped.Transpose(1, 2)
	if err != nil {
		return nil, fmt.Errorf("failed to transpose cross-attention value tensor: %w", err)
	}

	// Calculate attention scores: Q @ K^T
	// K^T shape: [batch_size, num_kv_heads, head_dim, kv_seq_length]
	kT_Transposed, err := kTransposed.Transpose(2, 3)
	if err != nil {
		return nil, fmt.Errorf("failed to transpose key tensor for cross-attention multiplication: %w", err)
	}

	// MatMul Q [batch, num_q_heads, q_seq_len, head_dim] @ K^T [batch, num_kv_heads, head_dim, kv_seq_len]
	// This requires broadcasting or repeating K^T along the num_q_heads dimension if num_q_heads != num_kv_heads.
	// Assuming num_q_heads == num_kv_heads for simplicity in this simplified model.
	if mha.NumQHeads != mha.NumKVHeads {
		return nil, errors.New("cross-attention simplified model requires NumQHeads == NumKVHeads")
	}

	attentionScores, err := qTransposed.MatMul(kT_Transposed)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate cross-attention scores (Q@K^T): %w", err)
	}

	// Scale attention scores
	scale := 1.0 / math.Sqrt(float64(mha.DimModel/mha.NumQHeads)) // Scale by sqrt of head dim
	scaledAttentionScores, err := attentionScores.ScalarMul(scale)
	if err != nil {
		return nil, fmt.Errorf("failed to scale cross-attention scores: %w", err)
	}

	// Apply mask (if provided) - Simplified, just add large negative to masked positions
	if mask != nil {
		maskedAttentionScores, err := scaledAttentionScores.AddWithBroadcast(mask)
		if err != nil {
			return nil, fmt.Errorf("failed to apply mask to cross-attention scores: %w", err)
		}
		scaledAttentionScores = maskedAttentionScores
	}

	// Apply Softmax to get attention weights
	attentionWeights, err := scaledAttentionScores.Softmax(len(scaledAttentionScores.Shape) - 1) // Softmax along the last dimension
	if err != nil {
		return nil, fmt.Errorf("failed to apply softmax to cross-attention scores: %w", err)
	}

	// Apply attention weights to Value: Attention Weights @ V
	contextLayer, err := attentionWeights.MatMul(vTransposed)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate cross-attention context layer (Attention@V): %w", err)
	}

	// Concatenate heads and apply final linear layer
	// [batch_size, num_heads, q_seq_length, head_dim] -> [batch_size, q_seq_length, num_heads * head_dim] (which is dim_model)
	contextLayerTransposed, err := contextLayer.Transpose(1, 2) // Transpose back to [batch_size, q_seq_length, num_heads, head_dim]
	if err != nil {
		return nil, fmt.Errorf("failed to transpose context layer: %w", err)
	}

	// Reshape to [batch_size, q_seq_length, dim_model]
	outputShape := []int{batchSize, qSeqLength, mha.DimModel}
	contextLayerReshaped, err := contextLayerTransposed.Reshape(outputShape)
	if err != nil {
		return nil, fmt.Errorf("failed to reshape cross-attention context layer: %w", err)
	}

	// Apply output linear layer
	output, err := mha.OutputLinear.Forward(contextLayerReshaped)
	if err != nil {
		return nil, fmt.Errorf("cross-attention output linear failed: %w", err)
	}

	return output, nil
}
