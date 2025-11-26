package nn

import (
	"encoding/gob"
	"errors"
	"fmt"
	"math"
	"math/rand"

	. "github.com/zendrulat/nlptagger/neural/tensor"
)

func init() {
	gob.Register(&Linear{})
	gob.Register(&MultiHeadCrossAttention{})

}

// Linear represents a linear layer (fully connected layer).
type Linear struct {
	Weights *Tensor
	Biases  *Tensor
	input   *Tensor // Store input for backward pass
}

// NewLinear creates a new Linear layer with random weights and zero biases.
func NewLinear(inputDim, outputDim int) (*Linear, error) {
	// He initialization
	stdDev := math.Sqrt(2.0 / float64(inputDim))
	weightsData := make([]float64, inputDim*outputDim)
	for i := range weightsData {
		weightsData[i] = rand.NormFloat64() * stdDev
	}
	weights := NewTensor([]int{inputDim, outputDim}, weightsData, true)
	weights.RequiresGrad = true

	// Biases are usually initialized to zero
	biasesData := make([]float64, outputDim)
	biases := NewTensor([]int{outputDim}, biasesData, true)
	biases.RequiresGrad = true

	return &Linear{Weights: weights, Biases: biases}, nil
}

// Parameters returns all learnable parameters of the layer.
func (l *Linear) Parameters() []*Tensor {
	params := []*Tensor{l.Weights}
	if l.Biases != nil {
		params = append(params, l.Biases)
	}
	return params
}

// Input returns the input tensor of the Linear operation.
func (l *Linear) Input() *Tensor {
	return l.input
}

// Forward performs the forward pass of the Linear layer.
func (l *Linear) Forward(inputs ...*Tensor) (*Tensor, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("Linear.Forward expects 1 input, got %d", len(inputs))
	}
	input := inputs[0]
	if input == nil { // Add this check
		return nil, fmt.Errorf("Linear.Forward received a nil input tensor")
	}
	// Store input tensor for backward pass
	l.input = input

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
		reshapedInput := NewTensor([]int{batchSize * seqLength, inputDim}, reshapedInputData, false)

		// Perform matrix multiplication: [batch_size * sequence_length, input_dim] @ [input_dim, output_dim]
		output2D, err := reshapedInput.MatMul(l.Weights)
		if err != nil {
			return nil, fmt.Errorf("linear layer 3D matrix multiplication failed: %w", err)
		}
		for i, val := range output2D.Data {
			if math.IsNaN(val) || math.IsInf(val, 0) {
				return nil, fmt.Errorf("NaN or Inf detected in Linear.Forward output2D at index %d", i)
			}
		}

		// Reshape output back to 3D
		outputData := make([]float64, batchSize*seqLength*outputDim)
		copy(outputData, output2D.Data) // Create a copy
		output = NewTensor([]int{batchSize, seqLength, outputDim}, outputData, false)
		output.RequiresGrad = true

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

	// Set creator and RequiresGrad for the output tensor
	output.RequiresGrad = input.RequiresGrad || l.Weights.RequiresGrad || (l.Biases != nil && l.Biases.RequiresGrad)
	if output.RequiresGrad {
		output.Creator = l
	}

	return output, nil
}

// Backward performs the backward pass for the Linear layer.
// grad is the gradient from the output (dLoss/dOutput).
func (l *Linear) Backward(grad *Tensor) error {
	if grad == nil || grad.Data == nil {
		// No gradient to propagate
		return nil
	}
	if l.input == nil {
		return errors.New("linear layer backward called before forward (input is nil)")
	}

	// Ensure gradients are initialized for parameters that require them
	if l.Weights.RequiresGrad && l.Weights.Grad == nil {
		l.Weights.Grad = NewTensor(l.Weights.Shape, make([]float64, len(l.Weights.Data)), false)
	}
	if l.Biases != nil && l.Biases.RequiresGrad {
		if l.Biases.Grad == nil {
			l.Biases.Grad = NewTensor(l.Biases.Shape, make([]float64, len(l.Biases.Data)), false)
		}
	}

	// --- Calculate Gradient with respect to Weights (dLoss/dWeights) ---
	// dLoss/dWeights = Input^T @ dLoss/dOutput (grad)
	var inputTranspose *Tensor
	var err error

	switch len(l.input.Shape) {
	case 2:
		// Input is 2D [batch_size, input_dim]
		inputTranspose, err = l.input.Transpose(0, 1)
		if err != nil {
			return fmt.Errorf("linear layer backward: failed to transpose input (2D): %w", err)
		}
		dWeights, err := inputTranspose.MatMul(grad)
		if err != nil {
			return fmt.Errorf("linear layer backward: failed to calculate dLoss/dWeights (2D): %w", err)
		}
		for i := range l.Weights.Grad.Data {
			l.Weights.Grad.Data[i] += dWeights.Data[i]
		}

	case 3:
		// Input is 3D [batch_size, sequence_length, input_dim]
		batchSize := l.input.Shape[0]
		seqLength := l.input.Shape[1]
		inputDim := l.input.Shape[2]
		outputDim := l.Weights.Shape[1]

		reshapedInput := NewTensor([]int{batchSize * seqLength, inputDim}, l.input.Data, false)
		reshapedGrad := NewTensor([]int{batchSize * seqLength, outputDim}, grad.Data, false)

		inputTranspose, err = reshapedInput.Transpose(0, 1)
		if err != nil {
			return fmt.Errorf("linear layer backward: failed to transpose reshaped input (3D): %w", err)
		}

		dWeights, err := inputTranspose.MatMul(reshapedGrad)
		if err != nil {
			return fmt.Errorf("linear layer backward: failed to calculate dLoss/dWeights (3D): %w", err)
		}
		for i := range l.Weights.Grad.Data {
			l.Weights.Grad.Data[i] += dWeights.Data[i]
		}

	default:
		return fmt.Errorf("linear layer backward only supports 2D or 3D input, got %d dimensions", len(l.input.Shape))
	}

	// --- Calculate Gradient with respect to Bias (dLoss/dBias) ---
	if l.Biases != nil && l.Biases.RequiresGrad {
		lastDimSize := grad.Shape[len(grad.Shape)-1]
		for i := 0; i < len(grad.Data); i++ {
			biasIndex := i % lastDimSize
			l.Biases.Grad.Data[biasIndex] += grad.Data[i]
		}
	}

	// --- Calculate Gradient with respect to Input (dLoss/dInput) ---
	if l.input.RequiresGrad {
		if l.input.Grad == nil {
			l.input.Grad = NewTensor(l.input.Shape, make([]float64, len(l.input.Data)), false)
		}

		weightsTranspose, err := l.Weights.Transpose(0, 1)
		if err != nil {
			return fmt.Errorf("linear layer backward: failed to transpose weights: %w", err)
		}

		var dInput *Tensor
		switch len(grad.Shape) {
		case 2:
			dInput, err = grad.MatMul(weightsTranspose)
			if err != nil {
				return fmt.Errorf("linear layer backward: failed to calculate dLoss/dInput (2D): %w", err)
			}
		case 3:
			batchSize := grad.Shape[0]
			seqLength := grad.Shape[1]
			outputDim := grad.Shape[2]
			inputDim := l.Weights.Shape[0]

			reshapedGrad := NewTensor([]int{batchSize * seqLength, outputDim}, grad.Data, false)

			dInput2D, err := reshapedGrad.MatMul(weightsTranspose)
			if err != nil {
				return fmt.Errorf("linear layer backward: failed to calculate dLoss/dInput (3D): %w", err)
			}
			dInput = NewTensor([]int{batchSize, seqLength, inputDim}, dInput2D.Data, false)

		default:
			return fmt.Errorf("linear layer backward only supports 2D or 3D gradient, got %d dimensions", len(grad.Shape))
		}

		for i := range l.input.Grad.Data {
			l.input.Grad.Data[i] += dInput.Data[i]
		}
	}

	return nil
}

// Inputs returns the input tensors of the Linear operation.
func (l *Linear) Inputs() []*Tensor {
	if l.input != nil {
		return []*Tensor{l.input}
	}
	return []*Tensor{}
}

// LayerNormalization represents a layer normalization layer.
type LayerNormalization struct {
	Gamma           *Tensor // Scale parameter
	Beta            *Tensor // Shift parameter
	Epsilon         float64 // Small value to prevent division by zero
	mean            *Tensor
	invStdDev       *Tensor // Inverse standard deviation (1 / sqrt(variance + epsilon))
	normalizedInput *Tensor // Input after normalization, before scaling and shifting
	inputTensor     *Tensor // Add this field
	inputShape      []int   // Store input shape for backward
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
	gamma := NewTensor([]int{dimModel}, gammaData, true)
	beta := NewTensor([]int{dimModel}, betaData, true)

	return &LayerNormalization{
		Gamma:   gamma,
		Beta:    beta,
		Epsilon: 1e-5, // Small epsilon value
	}
}

// Parameters returns all learnable parameters of the layer.
func (l *LayerNormalization) Parameters() []*Tensor {
	return []*Tensor{l.Gamma, l.Beta}
}

// Backward performs the backward pass for layer normalization.
// grad is the gradient from the output (dLoss/dOutput).
func (l *LayerNormalization) Backward(grad *Tensor) error {
	if grad == nil || grad.Data == nil {
		// No gradient to propagate
		return nil
	}

	if l.normalizedInput == nil || l.invStdDev == nil || l.mean == nil {
		panic("LayerNormalization backward called before forward (intermediate values are nil)")
	}
	if l.Gamma == nil || l.Beta == nil {
		panic("LayerNormalization scale or bias is nil in backward")
	}

	lastDimSize := l.inputShape[len(l.inputShape)-1]
	numElementsToNormalize := len(l.normalizedInput.Data) / lastDimSize

	// Ensure gradients are initialized
	if l.inputTensor != nil && l.inputTensor.RequiresGrad { // Assuming inputTensor is stored in the struct
		if l.inputTensor.Grad == nil {
			l.inputTensor.Grad = NewTensor(l.inputTensor.Shape, make([]float64, len(l.inputTensor.Data)), false)
		}
	}
	if l.Gamma.RequiresGrad {
		if l.Gamma.Grad == nil {
			l.Gamma.Grad = NewTensor(l.Gamma.Shape, make([]float64, len(l.Gamma.Data)), false)
		}
	}
	if l.Beta.RequiresGrad {
		if l.Beta.Grad == nil {
			l.Beta.Grad = NewTensor(l.Beta.Shape, make([]float64, len(l.Beta.Data)), false)
		}
	}

	// --- Calculate Gradient with respect to Beta (Bias) ---
	// dLoss/dBeta = Sum(grad) over all dims except the last one.
	if l.Beta.RequiresGrad {
		for i := 0; i < len(grad.Data); i++ {
			biasIndex := i % lastDimSize
			l.Beta.Grad.Data[biasIndex] += grad.Data[i]
		}
	}

	// --- Calculate Gradient with respect to Gamma (Scale) ---
	// dLoss/dGamma = Sum(grad * normalized_input) over all dims except the last one.
	if l.Gamma.RequiresGrad {
		if l.Gamma.Grad == nil {
			l.Gamma.Grad = NewTensor(l.Gamma.Shape, make([]float64, len(l.Gamma.Data)), false)
		}
		// Sum (grad * normalizedInput) over all dimensions except the last one
		for i := 0; i < numElementsToNormalize; i++ {
			for j := 0; j < lastDimSize; j++ {
				// Index in flattened data
				flatIndex := i*lastDimSize + j
				l.Gamma.Grad.Data[j] += grad.Data[flatIndex] * l.normalizedInput.Data[flatIndex]
			}
		}
	}

	// Calculate gradient with respect to normalized input
	// dLoss/dNormalizedInput = grad * gamma (Scale)
	dLoss_dNormalizedInputData := make([]float64, len(grad.Data))
	for i := 0; i < numElementsToNormalize; i++ {
		for j := 0; j < lastDimSize; j++ {
			flatIndex := i*lastDimSize + j
			dLoss_dNormalizedInputData[flatIndex] = grad.Data[flatIndex] * l.Gamma.Data[j]
		}
	}

	// Propagate gradient backward through normalization (mean and variance)
	// This is the most complex part. The formula for the gradient with respect to the input 'x' is:
	// dLoss/dx = dLoss/dNormalizedInput * (1 / std_dev)
	//           + dLoss/dStdDev * (x - mean) / (std_dev^2 * N)
	//           + dLoss/dMean / N
	// where N is the size of the last dimension (lastDimSize).

	// dLoss/dStdDev = Sum(dLoss/dNormalizedInput * (x - mean)) over the last dimension.
	// dLoss/dMean = Sum(dLoss/dNormalizedInput * (-1 / std_dev)) over the last dimension.

	// Let's calculate dLoss/dStdDev and dLoss/dMean first.

	dLoss_dStdDevData := make([]float64, numElementsToNormalize) // Gradients for std dev of each feature set
	dLoss_dMeanData := make([]float64, numElementsToNormalize)   // Gradients for mean of each feature set

	for i := 0; i < numElementsToNormalize; i++ {
		sum_dL_dNorm_x_minus_mean := 0.0
		sum_dL_dNorm := 0.0 // Needed for dLoss/dMean calculation

		for j := 0; j < lastDimSize; j++ {
			flatIndex := i*lastDimSize + j
			x_minus_mean := l.inputTensor.Data[flatIndex] - l.mean.Data[i] // Assuming inputTensor is stored
			sum_dL_dNorm_x_minus_mean += dLoss_dNormalizedInputData[flatIndex] * x_minus_mean
			sum_dL_dNorm += dLoss_dNormalizedInputData[flatIndex]
		}

		// dLoss/dStdDev
		stdDev := 1.0 / l.invStdDev.Data[i]
		if math.IsNaN(stdDev) || math.IsInf(stdDev, 0) {
		}
		dLoss_dStdDevData[i] = sum_dL_dNorm_x_minus_mean * (-1.0 / (stdDev * stdDev)) // Derivative of 1/std_dev is -1/std_dev^2
		dLoss_dMeanData[i] = sum_dL_dNorm * (-l.invStdDev.Data[i])

		if math.IsNaN(dLoss_dStdDevData[i]) || math.IsInf(dLoss_dStdDevData[i], 0) {
		}
		if math.IsNaN(dLoss_dMeanData[i]) || math.IsInf(dLoss_dMeanData[i], 0) {
		}
	}

	if l.inputTensor.RequiresGrad {
		if l.inputTensor.Grad == nil {
			l.inputTensor.Grad = NewTensor(l.inputTensor.Shape, make([]float64, len(l.inputTensor.Data)), false)
		}
		// Iterate over each feature vector (e.g., each token embedding in a sequence)
		for i := 0; i < numElementsToNormalize; i++ {
			// Pre-calculate sums for the current feature vector to avoid redundant computation.
			sum_dL_dNorm := 0.0
			sum_dL_dNorm_x_minus_mean := 0.0
			for k := 0; k < lastDimSize; k++ {
				flatIndex_k := i*lastDimSize + k
				sum_dL_dNorm += dLoss_dNormalizedInputData[flatIndex_k]
				sum_dL_dNorm_x_minus_mean += dLoss_dNormalizedInputData[flatIndex_k] * (l.inputTensor.Data[flatIndex_k] - l.mean.Data[i])
			}

			invStdDev_i := l.invStdDev.Data[i]

			// Now, calculate the gradient for each element within the feature vector using the pre-calculated sums.
			for j := 0; j < lastDimSize; j++ {
				flatIndex := i*lastDimSize + j
				x_j_minus_mean_i := l.inputTensor.Data[flatIndex] - l.mean.Data[i]

				// Calculate dLoss/dx_j
				dL_dx_j := invStdDev_i * (dLoss_dNormalizedInputData[flatIndex] - sum_dL_dNorm/float64(lastDimSize) - x_j_minus_mean_i*invStdDev_i*invStdDev_i*sum_dL_dNorm_x_minus_mean/float64(lastDimSize))
				if math.IsNaN(dL_dx_j) || math.IsInf(dL_dx_j, 0) {
				}

				l.inputTensor.Grad.Data[flatIndex] += dL_dx_j // Accumulate gradient
			}
		}
	}
	return nil
}

// Inputs returns the input tensors of the LayerNormalization operation.
// Assuming the input tensor is stored in the struct.
func (l *LayerNormalization) Inputs() []*Tensor {
	if l.inputTensor != nil {
		return []*Tensor{l.inputTensor}
	}
	return []*Tensor{} // Return empty slice if inputTensor is nil
}

// Forward performs the forward pass of layer normalization.
func (l *LayerNormalization) Forward(inputs ...*Tensor) (*Tensor, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("LayerNormalization.Forward expects 1 input, got %d", len(inputs))
	}
	input := inputs[0]
	l.inputTensor = input      // Store input for backward pass
	l.inputShape = input.Shape // Store input shape

	for i, val := range input.Data {
		if math.IsNaN(val) || math.IsInf(val, 0) {
			return nil, fmt.Errorf("input to LayerNormalization contains NaN or Inf at index %d", i)
		}
	}

	l.inputTensor = input // Add this line
	if input == nil || input.Data == nil {
		return nil, errors.New("input tensor is nil or has no data")
	}
	if len(input.Shape) == 0 {
		return nil, errors.New("input tensor cannot be a scalar for layer normalization")
	}
	if l.Gamma == nil || l.Beta == nil {
		return nil, errors.New("layer normalization Gamma or Beta is nil")
	}
	if len(l.Gamma.Shape) != 1 || l.Gamma.Shape[0] != input.Shape[len(input.Shape)-1] ||
		len(l.Beta.Shape) != 1 || l.Beta.Shape[0] != input.Shape[len(input.Shape)-1] {
		return nil, fmt.Errorf("layer normalization Gamma/bias shape mismatch with input last dimension: %v vs %d", l.Gamma.Shape, input.Shape[len(input.Shape)-1])
	}

	// Store input shape for backward pass
	l.inputShape = input.Shape

	// Calculate mean and variance across the last dimension
	lastDimSize := input.Shape[len(input.Shape)-1]
	numElementsToNormalize := len(input.Data) / lastDimSize // Number of elements to calculate mean/variance over for each feature

	meanData := make([]float64, numElementsToNormalize)
	varianceData := make([]float64, numElementsToNormalize)
	normalizedInputData := make([]float64, len(input.Data))
	meanShape := make([]int, len(input.Shape)-1)
	copy(meanShape, input.Shape[:len(input.Shape)-1])

	// Create the Tensor structs for intermediate values
	l.mean = NewTensor(meanShape, meanData, false)                                     // Pass the data slice
	l.invStdDev = NewTensor(meanShape, make([]float64, numElementsToNormalize), false) // Create data slice here
	l.normalizedInput = NewTensor(input.Shape, normalizedInputData, false)             // Pass the data slice
	for i := 0; i < numElementsToNormalize; i++ {
		// Calculate mean
		sum := 0.0
		for j := 0; j < lastDimSize; j++ {
			sum += input.Data[i*lastDimSize+j]
		}
		l.mean.Data[i] = sum / float64(lastDimSize) // Store mean in the tensor's data

		// Calculate variance
		sumSqDiff := 0.0
		for j := 0; j < lastDimSize; j++ {
			diff := input.Data[i*lastDimSize+j] - l.mean.Data[i] // Use l.mean.Data
			sumSqDiff += diff * diff
		}
		varianceData[i] = sumSqDiff / float64(lastDimSize)

		// Calculate normalized input
		variance := math.Max(0, varianceData[i]) // Ensure variance is non-negative
		invStdDev := 1.0 / math.Sqrt(variance+l.Epsilon)
		l.invStdDev.Data[i] = invStdDev // Store inverse standard deviation in the tensor's data

		for j := 0; j < lastDimSize; j++ {
			l.normalizedInput.Data[i*lastDimSize+j] = (input.Data[i*lastDimSize+j] - l.mean.Data[i]) * invStdDev // Store normalized input
		}
	}

	// Scale and shift
	outputData := make([]float64, len(input.Data))
	for i := 0; i < numElementsToNormalize; i++ {
		for j := 0; j < lastDimSize; j++ {
			outputData[i*lastDimSize+j] = l.Gamma.Data[j]*l.normalizedInput.Data[i*lastDimSize+j] + l.Beta.Data[j] // Use l.normalizedInput.Data
		}
	}

	outputTensor := NewTensor(input.Shape, outputData, false)
	outputTensor.RequiresGrad = input.RequiresGrad || l.Gamma.RequiresGrad || l.Beta.RequiresGrad
	if outputTensor.RequiresGrad {
		outputTensor.Creator = l
	}

	return outputTensor, nil
}

// MultiHeadAttention represents a multi-head attention layer.
type MultiHeadAttention struct {
	Wq, Wk, Wv      *Tensor // Linear layers for Q, K, V (learnable weights)
	Wo              *Tensor // Output linear layer (learnable weights)
	NumHeads        int
	DimModel        int
	HeadDim         int // dimModel / numHeads
	Depth           int
	attentionOutput *Tensor
	// Stored intermediate tensors for backward pass
	inputTensor                 *Tensor // Original input (Q, K, V are the same for self-attention)
	q, k, v                     *Tensor // Q, K, V after linear projection and splitting heads
	attentionScores             *Tensor // Q @ K^T
	attentionWeights            *Tensor // Softmax(attentionScores) + Mask
	attentionOutputBeforeConcat *Tensor // attentionWeights @ V (before concatenating heads)
	QueryLinear                 *Linear
	KeyLinear                   *Linear
	ValueLinear                 *Linear
	OutputLinear                *Linear
	// Output of the final linear layer is returned by Forward
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
		Depth:        headDim, // Initialize Depth
		QueryLinear:  queryLinear,
		KeyLinear:    keyLinear,
		ValueLinear:  valueLinear,
		OutputLinear: outputLinear,
	}, nil
}

// Parameters returns all learnable parameters of the layer.
func (mha *MultiHeadAttention) Parameters() []*Tensor {
	params := mha.QueryLinear.Parameters()
	params = append(params, mha.KeyLinear.Parameters()...)
	params = append(params, mha.ValueLinear.Parameters()...)
	params = append(params, mha.OutputLinear.Parameters()...)
	return params
}

// Backward performs the backward pass for multi-head self-attention.
// grad is the gradient from the output of the attention layer (after the final linear layer).
func (mha *MultiHeadAttention) Backward(grad *Tensor) error {
	if grad == nil || grad.Data == nil {
		return nil
	}

	// Initialize mha.attentionOutput.Grad with the incoming gradient
	// This is the gradient of the loss with respect to the output of this layer.
	if mha.attentionOutput == nil {
		return errors.New("mha.attentionOutput is nil in backward pass")
	}
	if mha.attentionOutput.Grad == nil {
		mha.attentionOutput.Grad = NewTensor(grad.Shape, make([]float64, len(grad.Data)), false)
	}
	// Accumulate the incoming gradient to mha.attentionOutput.Grad
	for i := range grad.Data {
		mha.attentionOutput.Grad.Data[i] += grad.Data[i]
	}

	if mha.OutputLinear != nil {
		err := mha.OutputLinear.Backward(mha.attentionOutput.Grad) // Pass the accumulated gradient
		if err != nil {
			return err
		}
	}

	batchSize := mha.attentionOutput.Grad.Shape[0]
	seqLength := mha.attentionOutput.Grad.Shape[1]
	dimModel := mha.attentionOutput.Grad.Shape[2]
	numHeads := mha.NumHeads // Assuming NumHeads is stored in MHA struct
	depth := mha.Depth       // Assuming Depth is stored in MHA struct (dimModel / numHeads)

	if dimModel != numHeads*depth {
		panic(fmt.Sprintf("dimModel (%d) does not match numHeads (%d) * depth (%d) in reshape step", dimModel, numHeads, depth))
	}

	// The reshaped gradient will have shape [batch_size, num_heads, seq_len, depth]
	reshapedShape := []int{batchSize, numHeads, seqLength, depth}
	gradBeforeConcatData := make([]float64, len(mha.attentionOutput.Grad.Data)) // Same size data
	gradBeforeConcat := NewTensor(reshapedShape, gradBeforeConcatData, false)   // Gradient tensor does not require gradients

	// Manually reshape the gradient by mapping flattened indices.
	// Original flat index for [b, s, d_model_idx]: b * seqLength * dimModel + s * dimModel + d_model_idx
	// Reshaped flat index for [b, h, s, d]: b * num_heads * seqLength * depth + h * seqLength * depth + s * depth + d

	originalFlatIndex := 0
	for b := 0; b < batchSize; b++ {
		for s := 0; s < seqLength; s++ {
			for d_model_idx := 0; d_model_idx < dimModel; d_model_idx++ {
				// Calculate original flat index
				originalFlatIndex = b*seqLength*dimModel + s*dimModel + d_model_idx

				// Calculate h and d from d_model_idx
				h := d_model_idx / depth
				d := d_model_idx % depth

				// Calculate reshaped flat index
				reshapedFlatIndex := b*numHeads*seqLength*depth + h*seqLength*depth + s*depth + d

				// Ensure indices are within bounds
				if originalFlatIndex >= len(mha.attentionOutput.Grad.Data) || reshapedFlatIndex >= len(gradBeforeConcat.Data) {
					panic("index out of bounds during gradient reshape before concat")
				}

				// Copy the gradient value
				gradBeforeConcat.Data[reshapedFlatIndex] = mha.attentionOutput.Grad.Data[originalFlatIndex]
			}
		}
	}

	// --- 3. Backpropagate through MatMul(attentionWeights @ V) ---
	// Inputs: mha.attentionWeights [b, h, s, s], mha.v [b, h, s, d]
	// Output gradient: gradBeforeConcat [b, h, s, d]
	// Calculate gradients w.r.t. inputs:
	// dLoss/dattentionWeights [b, h, s, s] = gradBeforeConcat [b, h, s, d] @ mha.v^T [b, h, d, s]
	// dLoss/dv [b, h, s, d] = mha.attentionWeights^T [b, h, s, s] @ gradBeforeConcat [b, h, s, d]
	// Accumulate these gradients to mha.attentionWeights.Grad and mha.v.Grad.

	if mha.attentionWeights != nil && mha.v != nil && gradBeforeConcat != nil {
		// dLoss/dattentionWeights = gradBeforeConcat @ mha.v^T
		// Transpose mha.v: swap last two dimensions (seq_len and depth)
		vTransposed, err := mha.v.Transpose(1, 2)

		if err != nil {
			panic(fmt.Sprintf("failed to transpose v for MHA backward: %v", err))
		}

		// Need a batched MatMul operation: (b, h, s, d) @ (b, h, d, s) -> (b, h, s, s)
		gradAttentionWeights, err := gradBeforeConcat.MatMul(vTransposed) // Assuming MatMul handles batches
		if err != nil {
			panic(fmt.Sprintf("failed to calculate gradAttentionWeights in MHA backward: %v", err))
		}

		// Add to existing gradient for attention weights
		if mha.attentionWeights.RequiresGrad {
			if mha.attentionWeights.Grad == nil {
				mha.attentionWeights.Grad = NewTensor(mha.attentionWeights.Shape, make([]float64, len(mha.attentionWeights.Data)), false)
			}
			// Accumulate gradAttentionWeights to mha.attentionWeights.Grad element-wise
			for i := range mha.attentionWeights.Grad.Data {
				mha.attentionWeights.Grad.Data[i] += gradAttentionWeights.Data[i]
			}
		}

		// dLoss/dv = mha.attentionWeights^T @ gradBeforeConcat
		// Transpose mha.attentionWeights: swap last two dimensions (seq_len and seq_len)
		attentionWeightsTransposed, err := mha.attentionWeights.Transpose(2, 3)
		if err != nil {
			panic(fmt.Sprintf("failed to transpose attentionWeights for MHA backward: %v", err))
		}

		// Need a batched MatMul operation: (b, h, s, s) @ (b, h, s, d) -> (b, h, s, d)
		gradV, err := attentionWeightsTransposed.MatMul(gradBeforeConcat) // Assuming MatMul handles batches
		if err != nil {
			panic(fmt.Sprintf("failed to calculate gradV in MHA backward: %v", err))
		}

		// Add to existing gradient for v
		if mha.v != nil && mha.v.RequiresGrad {
			if mha.v.Grad == nil {
				mha.v.Grad = NewTensor(mha.v.Shape, make([]float64, len(mha.v.Data)), false)
			}
			for i := range mha.v.Grad.Data {
				mha.v.Grad.Data[i] += gradV.Data[i]
			}
		}
	}

	// --- 4. Backpropagate through Softmax (and Masking) ---
	// Input: mha.attentionScores
	// Output: mha.attentionWeights
	// Output gradient: mha.attentionWeights.Grad (accumulated in step 3)
	// Calculate gradient w.r.t. attentionScores and accumulate in mha.attentionScores.Grad.
	// Assuming your Softmax operation has a Backward method and mha.attentionWeights is its output tensor created by the Softmax operation:

	if mha.attentionWeights != nil && mha.attentionWeights.Grad != nil {
		// Call Backward on the output tensor of the Softmax operation.
		// This triggers the Softmax operation's Backward method.
		if mha.attentionWeights.Creator != nil {
			err := mha.attentionWeights.Creator.Backward(mha.attentionWeights.Grad)
			if err != nil {
				return fmt.Errorf("softmax backward failed: %w", err)
			}
		} else {
			return errors.New("attentionWeights.Creator is nil, cannot backpropagate through Softmax")
		}
	}
	// After this, the gradient w.r.t. mha.attentionScores is accumulated in mha.attentionScores.Grad.

	// --- 5. Backpropagate through MatMul(Q @ K^T) ---
	// Inputs: mha.q [b, h, s, d], mha.k [b, h, s, d] (K^T in forward)
	// Output: mha.attentionScores [b, h, s, s]
	// Output gradient: mha.attentionScores.Grad (accumulated in step 4)
	// Calculate gradients w.r.t. inputs:
	// dLoss/dq [b, h, s, d] = mha.attentionScores.Grad [b, h, s, s] @ mha.k [b, h, s, d] (untransposed K)
	// dLoss/dk [b, h, s, d] = mha.q^T [b, h, d, s] @ mha.attentionScores.Grad [b, h, s, s]
	// Accumulate these gradients to mha.q.Grad and mha.k.Grad.

	if mha.q != nil && mha.k != nil && mha.attentionScores != nil && mha.attentionScores.Grad != nil {
		// dLoss/dq = dLoss/dAttentionScores @ k (untransposed)
		// Transpose mha.k to get the original K shape if needed for MatMul (depends on your MatMul implementation)
		// Assuming your MatMul works with [b, h, s, s] @ [b, h, s, d] -> [b, h, s, d]
		gradQ_per_head, err := mha.attentionScores.Grad.MatMul(mha.k) // Batched MatMul
		if err != nil {
			panic(fmt.Sprintf("failed to calculate gradQ in MHA backward: %v", err))
		}
		// Accumulate gradQ_per_head to mha.q.Grad
		if mha.q.RequiresGrad {
			if mha.q.Grad == nil {
				mha.q.Grad = NewTensor(mha.q.Shape, make([]float64, len(mha.q.Data)), false)
			}
			for i := range mha.q.Grad.Data {
				mha.q.Grad.Data[i] += gradQ_per_head.Data[i]
			}
		}

		// dLoss/dk = dLoss/dAttentionScores^T @ q
		attentionScoresGradTransposed, err := mha.attentionScores.Grad.Transpose(2, 3)
		if err != nil {
			panic(fmt.Sprintf("failed to transpose attentionScores.Grad for MHA backward: %v", err))
		}

		gradK_per_head, err := attentionScoresGradTransposed.MatMul(mha.q)
		if err != nil {
			panic(fmt.Sprintf("failed to calculate gradK in MHA backward: %v", err))
		}
		// Accumulate gradK_per_head to mha.k.Grad
		if mha.k != nil && mha.k.RequiresGrad {
			if mha.k.Grad == nil {
				mha.k.Grad = NewTensor(mha.k.Shape, make([]float64, len(mha.k.Data)), false)
			}
			for i := range mha.k.Grad.Data {
				mha.k.Grad.Data[i] += gradK_per_head.Data[i]
			}
		}
	}

	// --- 6. Combine gradients from heads ---
	// Gradients w.r.t. q, k, v are in mha.q.Grad [b, h, s, d], mha.k.Grad [b, h, s, d], mha.v.Grad [b, h, s, d].
	// Sum these gradients over the 'num_heads' dimension (axis 1) to get combined gradients
	// with shape [batch_size, seq_len, dim_model].
	// Accumulate these combined gradients to the input tensor's gradient (mha.inputTensor.Grad).

	if mha.q != nil && mha.k != nil && mha.v != nil && mha.q.Grad != nil && mha.k.Grad != nil && mha.v.Grad != nil && mha.inputTensor != nil && mha.inputTensor.RequiresGrad {

		batchSize := mha.q.Grad.Shape[0]
		numHeads := mha.NumHeads // Assuming NumHeads is stored in MHA struct
		seqLength := mha.q.Grad.Shape[2]
		depth := mha.Depth       // Assuming Depth is stored in MHA struct
		dimModel := mha.DimModel // Assuming DimModel is stored in MHA struct

		// Implement the summation over heads for mha.q.Grad, mha.k.Grad, mha.v.Grad
		// to get gradQCombined, gradKCombined, gradVCombined [b, s, dim_model].
		// See previous explanation on how to manually sum over heads.

		// Sum mha.q.Grad over heads
		combinedShape := []int{batchSize, seqLength, dimModel}
		gradQCombinedData := make([]float64, batchSize*seqLength*dimModel)
		gradQCombined := NewTensor(combinedShape, gradQCombinedData, false)

		qGradFlatIndex := 0
		for b := 0; b < batchSize; b++ {
			for h := 0; h < numHeads; h++ {
				for s := 0; s < seqLength; s++ {
					for d := 0; d < depth; d++ {
						qGradFlatIndex = b*numHeads*seqLength*depth + h*seqLength*depth + s*depth + d
						combinedFlatIndex := b*seqLength*dimModel + s*dimModel + (h*depth + d)
						if qGradFlatIndex >= len(mha.q.Grad.Data) || combinedFlatIndex >= len(gradQCombined.Data) {
							panic("index out of bounds during gradient summation over heads for Q")
						}
						gradQCombined.Data[combinedFlatIndex] += mha.q.Grad.Data[qGradFlatIndex]
					}
				}
			}
		}

		// Sum mha.k.Grad over heads
		gradKCombinedData := make([]float64, batchSize*seqLength*dimModel)
		gradKCombined := NewTensor(combinedShape, gradKCombinedData, false)

		kGradFlatIndex := 0
		for b := 0; b < batchSize; b++ {
			for h := 0; h < numHeads; h++ {
				for s := 0; s < seqLength; s++ {
					for d := 0; d < depth; d++ {
						kGradFlatIndex = b*numHeads*seqLength*depth + h*seqLength*depth + s*depth + d
						combinedFlatIndex := b*seqLength*dimModel + s*dimModel + (h*depth + d)
						if kGradFlatIndex >= len(mha.k.Grad.Data) || combinedFlatIndex >= len(gradKCombined.Data) {
							panic("index out of bounds during gradient summation over heads for K")
						}
						gradKCombined.Data[combinedFlatIndex] += mha.k.Grad.Data[kGradFlatIndex]
					}
				}
			}
		}

		// Sum mha.v.Grad over heads
		gradVCombinedData := make([]float64, batchSize*seqLength*dimModel)
		gradVCombined := NewTensor(combinedShape, gradVCombinedData, false)

		vGradFlatIndex := 0
		for b := 0; b < batchSize; b++ {
			for h := 0; h < numHeads; h++ {
				for s := 0; s < seqLength; s++ {
					for d := 0; d < depth; d++ {
						vGradFlatIndex = b*numHeads*seqLength*depth + h*seqLength*depth + s*depth + d
						combinedFlatIndex := b*seqLength*dimModel + s*dimModel + (h*depth + d)
						if vGradFlatIndex >= len(mha.v.Grad.Data) || combinedFlatIndex >= len(gradVCombined.Data) {
							panic("index out of bounds during gradient summation over heads for V")
						}
						gradVCombined.Data[combinedFlatIndex] += mha.v.Grad.Data[vGradFlatIndex]
					}
				}
			}
		}

		// Accumulate combined gradients to the input tensor's gradient
		// Add gradQCombined.Data to mha.inputTensor.Grad.Data element-wise
		if mha.inputTensor.Grad == nil {
			mha.inputTensor.Grad = NewTensor(mha.inputTensor.Shape, make([]float64, len(mha.inputTensor.Data)), false)
		}
		for i := range mha.inputTensor.Grad.Data {
			mha.inputTensor.Grad.Data[i] += gradQCombined.Data[i]
		}

		// Add gradKCombined.Data to mha.inputTensor.Grad.Data element-wise
		for i := range mha.inputTensor.Grad.Data {
			mha.inputTensor.Grad.Data[i] += gradKCombined.Data[i]
		}

		// Add gradVCombined.Data to mha.inputTensor.Grad.Data element-wise
		for i := range mha.inputTensor.Grad.Data {
			mha.inputTensor.Grad.Data[i] += gradVCombined.Data[i]
		}

		if mha.inputTensor != nil && gradQCombined != nil && gradKCombined != nil && gradVCombined != nil {
			// Transpose input tensor for matrix multiplication with combined gradients
			inputTransposedForWeights, err := mha.inputTensor.Transpose(1, 2)
			if err != nil {
				panic(fmt.Sprintf("failed to transpose inputTensor for MHA backward weights: %v", err))
			}

			// dLoss/dWq = inputTensor^T @ gradQCombined
			if mha.Wq != nil && mha.Wq.RequiresGrad {
				if mha.Wq.Grad == nil {
					mha.Wq.Grad = NewTensor(mha.Wq.Shape, make([]float64, len(mha.Wq.Data)), false)
				}
				// Need a batched MatMul: [b, s, d_m] @ [b, s, d_m] -> [d_m, d_m] ? No, [s, d_m] @ [s, d_m] -> [d_m, d_m] summed over batch and sequence
				// The dimensions for the matrix multiplication should be [dim_model, seq_len] @ [seq_len, dim_model] -> [dim_model, dim_model]
				// We need to perform matrix multiplication of inputTensor^T [b, d_m, s] and gradQCombined [b, s, d_m]
				// and sum the results over the batch dimension.

				// This requires a batched matrix multiplication where the batch dimension is preserved
				// or a way to perform matrix multiplication and sum over batch.

				// Let's assume your MatMul handles batching such that [b, d_m, s] @ [b, s, d_m] results in [b, d_m, d_m]
				// and then we sum over the batch dimension.
				// Or, if your MatMul is designed for [..., M, K] @ [..., K, N], we need to reshape.

				// Let's perform the matrix multiplication and then sum over the batch dimension manually.

				// Reshape inputTransposedForWeights to [b, d_m, s] and gradQCombined to [b, s, d_m] (already are)
				// Perform MatMul: [b, d_m, s] @ [b, s, d_m] -> [b, d_m, d_m]
				// Sum over batch dimension (axis 0) to get [d_m, d_m]

				batchSize := mha.inputTensor.Shape[0]
				dimModel := mha.DimModel
				seqLength := mha.inputTensor.Shape[1]

				// Create a temporary tensor for the result of batched MatMul before summation
				tempGradWqShape := []int{batchSize, dimModel, dimModel}
				tempGradWqData := make([]float64, batchSize*dimModel*dimModel)
				tempGradWq := NewTensor(tempGradWqShape, tempGradWqData, false)

				// Perform batched matrix multiplication: inputTransposedForWeights @ gradQCombined
				// Iterate through batch
				for b := 0; b < batchSize; b++ {
					// Perform matrix multiplication for each batch: [d_m, s] @ [s, d_m] -> [d_m, d_m]
					// Input slices for MatMul:
					inputSlice := inputTransposedForWeights.Data[b*dimModel*seqLength : (b+1)*dimModel*seqLength]
					gradQSlice := gradQCombined.Data[b*seqLength*dimModel : (b+1)*seqLength*dimModel]
					outputSlice := tempGradWq.Data[b*dimModel*dimModel : (b+1)*dimModel*dimModel]

					// Assuming a 2D matrix multiplication function is available
					// You might need to implement a function like MatMul2D(A, B, result, M, K, N)
					// where A, B, result are flattened slices, and M, K, N are dimensions.
					// Let's simulate the 2D MatMul within the loop:
					M := dimModel
					K := seqLength
					N := dimModel
					for i := 0; i < M; i++ { // rows of output
						for j := 0; j < N; j++ { // columns of output
							sum := 0.0
							for k := 0; k < K; k++ {
								sum += inputSlice[i*K+k] * gradQSlice[k*N+j]
							}
							// OutputSlice index: i*N + j
							outputSlice[i*N+j] = sum
						}
					}
				}

				// Sum tempGradWq over the batch dimension (axis 0)
				// The result shape should be [dim_model, dim_model]
				finalGradWqData := make([]float64, dimModel*dimModel)
				for b := 0; b < batchSize; b++ {
					batchStart := b * dimModel * dimModel
					for i := 0; i < dimModel*dimModel; i++ {
						finalGradWqData[i] += tempGradWq.Data[batchStart+i]
					}
				}
				finalGradWq := NewTensor([]int{dimModel, dimModel}, finalGradWqData, false)

				// Accumulate finalGradWq to mha.Wq.Grad
				for i := range mha.Wq.Grad.Data {
					mha.Wq.Grad.Data[i] += finalGradWq.Data[i]
				}
			}

			// dLoss/dWk = inputTensor^T @ gradKCombined
			if mha.Wk != nil && mha.Wk.RequiresGrad {
				if mha.Wk.Grad == nil {
					mha.Wk.Grad = NewTensor(mha.Wk.Shape, make([]float64, len(mha.Wk.Data)), false)
				}
				// Implement calculation and accumulation for gradWk similar to gradWq
				// MatMul: inputTransposedForWeights [b, d_m, s] @ gradKCombined [b, s, d_m] -> [b, d_m, d_m]
				// Sum over batch dimension.

				batchSize := mha.inputTensor.Shape[0]
				dimModel := mha.DimModel
				seqLength := mha.inputTensor.Shape[1]

				tempGradWkShape := []int{batchSize, dimModel, dimModel}
				tempGradWkData := make([]float64, batchSize*dimModel*dimModel)
				tempGradWk := NewTensor(tempGradWkShape, tempGradWkData, false)

				for b := 0; b < batchSize; b++ {
					inputSlice := inputTransposedForWeights.Data[b*dimModel*seqLength : (b+1)*dimModel*seqLength]
					gradKSlice := gradKCombined.Data[b*seqLength*dimModel : (b+1)*seqLength*dimModel]
					outputSlice := tempGradWk.Data[b*dimModel*dimModel : (b+1)*dimModel*dimModel]

					M := dimModel
					K := seqLength
					N := dimModel
					for i := 0; i < M; i++ {
						for j := 0; j < N; j++ {
							sum := 0.0
							for k := 0; k < K; k++ {
								sum += inputSlice[i*K+k] * gradKSlice[k*N+j]
							}
							outputSlice[i*N+j] = sum
						}
					}
				}

				finalGradWkData := make([]float64, dimModel*dimModel)
				for b := 0; b < batchSize; b++ {
					batchStart := b * dimModel * dimModel
					for i := 0; i < dimModel*dimModel; i++ {
						finalGradWkData[i] += tempGradWk.Data[batchStart+i]
					}
				}
				finalGradWk := NewTensor([]int{dimModel, dimModel}, finalGradWkData, false)

				for i := range mha.Wk.Grad.Data {
					mha.Wk.Grad.Data[i] += finalGradWk.Data[i]
				}
			}

			// dLoss/dWv = inputTensor^T @ gradVCombined
			if mha.Wv != nil && mha.Wv.RequiresGrad {
				if mha.Wv.Grad == nil {
					mha.Wv.Grad = NewTensor(mha.Wv.Shape, make([]float64, len(mha.Wv.Data)), false)
				}
				// Implement calculation and accumulation for gradWv similar to gradWq
				// MatMul: inputTransposedForWeights [b, d_m, s] @ gradVCombined [b, s, d_m] -> [b, d_m, d_m]
				// Sum over batch dimension.

				batchSize := mha.inputTensor.Shape[0]
				dimModel := mha.DimModel
				seqLength := mha.inputTensor.Shape[1]

				tempGradWvShape := []int{batchSize, dimModel, dimModel}
				tempGradWvData := make([]float64, batchSize*dimModel*dimModel)
				tempGradWv := NewTensor(tempGradWvShape, tempGradWvData, false)

				for b := 0; b < batchSize; b++ {
					inputSlice := inputTransposedForWeights.Data[b*dimModel*seqLength : (b+1)*dimModel*seqLength]
					gradVSlice := gradVCombined.Data[b*seqLength*dimModel : (b+1)*seqLength*dimModel]
					outputSlice := tempGradWv.Data[b*dimModel*dimModel : (b+1)*dimModel*dimModel]

					M := dimModel
					K := seqLength
					N := dimModel
					for i := 0; i < M; i++ {
						for j := 0; j < N; j++ {
							sum := 0.0
							for k := 0; k < K; k++ {
								sum += inputSlice[i*K+k] * gradVSlice[k*N+j]
							}
							outputSlice[i*N+j] = sum
						}
					}
				}

				finalGradWvData := make([]float64, dimModel*dimModel)
				for b := 0; b < batchSize; b++ {
					batchStart := b * dimModel * dimModel
					for i := 0; i < dimModel*dimModel; i++ {
						finalGradWvData[i] += tempGradWv.Data[batchStart+i]
					}
				}
				finalGradWv := NewTensor([]int{dimModel, dimModel}, finalGradWvData, false)

				for i := range mha.Wv.Grad.Data {
					mha.Wv.Grad.Data[i] += finalGradWv.Data[i]
				}
			}
		}
	}
	return nil
}

// Forward performs the forward pass of the MultiHeadAttention layer.
// This is a simplified version without caching or masks.
func (mha *MultiHeadAttention) Forward(inputs ...*Tensor) (*Tensor, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("MultiHeadAttention.Forward expects 1 input, got %d", len(inputs))
	}
	value := inputs[0] // Extract the single input

	query := value
	key := value
	mask := value.Mask // Assuming mask is a field of the input tensor

	// Store the original input tensor (for self-attention, Q, K, V are the same)
	mha.inputTensor = query

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
	mha.q = qTransposed // Store q after transpose

	kReshaped, err := k.Reshape([]int{batchSize, kvSeqLength, mha.NumHeads, mha.HeadDim})
	if err != nil {
		return nil, fmt.Errorf("failed to reshape key tensor: %w", err)
	}

	kTransposed, err := kReshaped.Transpose(1, 2) // Transpose to [batch_size, num_heads, kv_seq_length, head_dim]
	if err != nil {
		return nil, fmt.Errorf("failed to transpose key tensor: %w", err)
	}
	mha.k = kTransposed // Store k after transpose

	vReshaped, err := v.Reshape([]int{batchSize, kvSeqLength, mha.NumHeads, mha.HeadDim})
	if err != nil {
		return nil, fmt.Errorf("failed to reshape value tensor: %w", err)
	}

	vTransposed, err := vReshaped.Transpose(1, 2) // Transpose to [batch_size, num_heads, kv_seq_length, head_dim]
	if err != nil {
		return nil, fmt.Errorf("failed to transpose value tensor: %w", err)
	}
	mha.v = vTransposed // Store v after transpose

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
	for i, val := range attentionScores.Data {
		if math.IsNaN(val) || math.IsInf(val, 0) {
			return nil, fmt.Errorf("attentionScores contains NaN or Inf at index %d", i)
		}
	}
	mha.attentionScores = attentionScores // Store attention scores

	// Gamma attention scores
	scale := 1.0 / math.Sqrt(float64(mha.HeadDim))
	scaledAttentionScores, err := attentionScores.MulScalar(scale)
	if err != nil {
		return nil, fmt.Errorf("failed to scale attention scores: %w", err)
	}
	for i, val := range scaledAttentionScores.Data {
		if math.IsNaN(val) || math.IsInf(val, 0) {
			return nil, fmt.Errorf("scaledAttentionScores contains NaN or Inf at index %d", i)
		}
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
	mha.attentionWeights = attentionWeights // Store attention weights

	// Apply attention weights to Value: Attention Weights @ V
	contextLayer, err := attentionWeights.MatMul(vTransposed)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate context layer (Attention@V): %w", err)
	}

	mha.attentionOutputBeforeConcat = contextLayer // Store attention output before concatenation

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

	// Apply the final linear layer to the reshaped context layer
	output, err := mha.OutputLinear.Forward(contextLayerReshaped) // Use contextLayerReshaped here
	if err != nil {
		return nil, fmt.Errorf("multihead attention output linear failed: %w", err)
	}

	// Set creator and RequiresGrad for the output tensor
	outputRequiresGrad := query.RequiresGrad || key.RequiresGrad || value.RequiresGrad ||
		mha.QueryLinear.Weights.RequiresGrad || (mha.QueryLinear.Biases != nil && mha.QueryLinear.Biases.RequiresGrad) ||
		mha.KeyLinear.Weights.RequiresGrad || (mha.KeyLinear.Biases != nil && mha.KeyLinear.Biases.RequiresGrad) ||
		mha.ValueLinear.Weights.RequiresGrad || (mha.ValueLinear.Biases != nil && mha.ValueLinear.Biases.RequiresGrad) ||
		mha.OutputLinear.Weights.RequiresGrad || (mha.OutputLinear.Biases != nil && mha.OutputLinear.Biases.RequiresGrad)

	output.RequiresGrad = outputRequiresGrad
	if output.RequiresGrad {
		output.Creator = mha // Set the creator to the MultiHeadAttention layer itself
	}

	mha.attentionOutput = output

	return output, nil // Return the output tensor and nil error
}

// Inputs returns the input tensors of the MultiHeadAttention operation.
func (mha *MultiHeadAttention) Inputs() []*Tensor {
	// For self-attention, the input is stored in inputTensor
	if mha.inputTensor != nil {
		return []*Tensor{mha.inputTensor}
	}
	// If no inputs are stored (e.g., before forward pass), return an empty slice
	return []*Tensor{}
}

// MultiHeadCrossAttention represents a multi-head cross-attention layer.
type MultiHeadCrossAttention struct {
	NumQHeads  int // Number of query heads
	NumKVHeads int // Number of key/value heads
	DimModel   int
	DimKVHeads int // Dimension per key/value head (head_dim for keys/values)
	Depth      int // Dimension per query head (head_dim for queries)

	// Stored intermediate tensors for backward pass
	queryTensor                 *Tensor // Original query input from decoder
	keyTensor                   *Tensor // Original key input from encoder
	valueTensor                 *Tensor // Original value input from encoder
	q, k, v                     *Tensor // Q, K, V after linear projection and splitting heads
	attentionScores             *Tensor // Q @ K^T
	attentionWeights            *Tensor // Softmax(attentionScores) + Mask
	attentionOutputBeforeConcat *Tensor // attentionWeights @ V (before concatenating heads)
	// Add the following field to store the final output of the layer
	attentionOutput *Tensor
	QueryLinear     *Linear
	KeyLinear       *Linear // Linear layer for keys from encoder output
	ValueLinear     *Linear // Linear layer for values from encoder output
	OutputLinear    *Linear
	Wo              *Linear // This seems to be a duplicate of OutputLinear based on its usage
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
		DimKVHeads:   dimKVHeads,           // Store dim per KV head
		Depth:        dimModel / numQHeads, // Initialize Depth
		QueryLinear:  queryLinear,
		KeyLinear:    keyLinear,
		ValueLinear:  valueLinear,
		OutputLinear: outputLinear,
	}, nil
}

// Parameters returns all learnable parameters of the layer.
func (mha *MultiHeadCrossAttention) Parameters() []*Tensor {
	params := mha.QueryLinear.Parameters()
	params = append(params, mha.KeyLinear.Parameters()...)
	params = append(params, mha.ValueLinear.Parameters()...)
	params = append(params, mha.OutputLinear.Parameters()...)
	return params
}

// Inputs returns the input tensors of the MultiHeadCrossAttention operation.
func (mha *MultiHeadCrossAttention) Inputs() []*Tensor {
	// Return the original input tensors: query, key, and value
	// The mask is not typically considered a tensor that requires gradients in the same way,
	// so it's usually not included in the Inputs for backpropagation.
	if mha.queryTensor != nil && mha.keyTensor != nil && mha.valueTensor != nil {
		return []*Tensor{mha.queryTensor, mha.keyTensor, mha.valueTensor}
	}
	// If inputs are not stored (e.g., before forward pass), return an empty slice
	return []*Tensor{}
}

func (mha *MultiHeadCrossAttention) Backward(grad *Tensor) error {
	if grad == nil || grad.Data == nil {
		return nil
	}

	// Initialize mha.attentionOutput.Grad with the incoming gradient
	// This is the gradient of the loss with respect to the output of this layer.
	if mha.attentionOutput == nil {
		return errors.New("mha.attentionOutput is nil in backward pass")
	}
	if mha.attentionOutput.Grad == nil {
		mha.attentionOutput.Grad = NewTensor(grad.Shape, make([]float64, len(grad.Data)), false)
	}
	// Accumulate the incoming gradient to mha.attentionOutput.Grad
	for i := range grad.Data {
		mha.attentionOutput.Grad.Data[i] += grad.Data[i]
	}

	// Backpropagate through the final linear layer (OutputLinear)
	if mha.OutputLinear != nil {
		err := mha.OutputLinear.Backward(mha.attentionOutput.Grad) // Pass the accumulated gradient
		if err != nil {
			return err
		}
	}

	batchSize := mha.attentionOutput.Grad.Shape[0]
	querySeqLen := mha.attentionOutput.Grad.Shape[1]
	dimModel := mha.attentionOutput.Grad.Shape[2]
	numQHeads := mha.NumQHeads
	depth := mha.Depth // Use mha.Depth for the dimension per Q head

	if dimModel != numQHeads*depth {
		panic(fmt.Sprintf("dimModel (%d) does not match numQHeads (%d) * depth (%d) in reshape step", dimModel, numQHeads, depth))
	}

	// The reshaped gradient will have shape [batch_size, num_q_heads, query_seq_len, depth]
	reshapedShape := []int{batchSize, numQHeads, querySeqLen, depth}
	gradBeforeConcatData := make([]float64, batchSize*numQHeads*querySeqLen*depth)
	gradBeforeConcat := NewTensor(reshapedShape, gradBeforeConcatData, false)

	originalFlatIndex := 0
	for b := 0; b < batchSize; b++ {
		for s := 0; s < querySeqLen; s++ {
			for d_model_idx := 0; d_model_idx < dimModel; d_model_idx++ {
				originalFlatIndex = b*querySeqLen*dimModel + s*dimModel + d_model_idx

				h := d_model_idx / depth
				d := d_model_idx % depth

				reshapedFlatIndex := b*numQHeads*querySeqLen*depth + h*querySeqLen*depth + s*depth + d

				if originalFlatIndex >= len(mha.attentionOutput.Grad.Data) || reshapedFlatIndex >= len(gradBeforeConcat.Data) {
					panic("index out of bounds during gradient reshape before concat")
				}

				gradBeforeConcat.Data[reshapedFlatIndex] = mha.attentionOutput.Grad.Data[originalFlatIndex]
			}
		}
	}

	if mha.attentionWeights != nil && mha.v != nil && gradBeforeConcat != nil {
		// dLoss/dattentionWeights = gradBeforeConcat @ mha.v^T
		// dLoss/dattentionWeights = gradBeforeConcat @ mha.v^T
		// Transpose mha.v to get shape [batch_size, num_kv_heads, head_dim, kv_seq_length]
		// mha.v has shape [batch_size, kv_seq_length, num_kv_heads, head_dim]
		// Transpose mha.v: swap last two dimensions (kv_seq_length and head_dim)
		vTransposed, err := mha.v.Transpose(2, 3)
		if err != nil {
			panic(fmt.Sprintf("failed to transpose v for MHCA backward: %v", err))
		}

		gradAttentionWeights, err := gradBeforeConcat.MatMul(vTransposed)
		if err != nil {
			panic(fmt.Sprintf("failed to calculate gradAttentionWeights in MHCA backward: %v", err))
		}

		if mha.attentionWeights.RequiresGrad {
			if mha.attentionWeights.Grad == nil {
				mha.attentionWeights.Grad = NewTensor(mha.attentionWeights.Shape, make([]float64, len(mha.attentionWeights.Data)), false)
			}
			for i := range mha.attentionWeights.Grad.Data {
				mha.attentionWeights.Grad.Data[i] += gradAttentionWeights.Data[i]
			}
		}

		// dLoss/dv = mha.attentionWeights^T @ gradBeforeConcat
		attentionWeightsTransposed, err := mha.attentionWeights.Transpose(len(mha.attentionWeights.Shape)-2, len(mha.attentionWeights.Shape)-1)
		if err != nil {
			panic(fmt.Sprintf("failed to transpose attentionWeights for MHCA backward: %v", err))
		}

		gradV, err := attentionWeightsTransposed.MatMul(gradBeforeConcat)
		if err != nil {
			panic(fmt.Sprintf("failed to calculate gradV in MHCA backward: %v", err))
		}

		if mha.v != nil && mha.v.RequiresGrad {
			if mha.v.Grad == nil {
				mha.v.Grad = NewTensor(mha.v.Shape, make([]float64, len(mha.v.Data)), false)
			}
			for i := range mha.v.Grad.Data {
				mha.v.Grad.Data[i] += gradV.Data[i]
			}
		}
	}

	if mha.attentionWeights != nil && mha.attentionWeights.Grad != nil {
		// Assuming Softmax output tensor (mha.attentionWeights) has its creator set to the Softmax operation,
		// calling Backward on it will trigger the Softmax backward pass.
		if mha.attentionWeights.Creator != nil {
			err := mha.attentionWeights.Creator.Backward(mha.attentionWeights.Grad)
			if err != nil {
				return fmt.Errorf("softmax backward failed: %w", err)
			}
		} else {
			return errors.New("attentionWeights.Creator is nil, cannot backpropagate through Softmax")
		}
	}

	if mha.q != nil && mha.k != nil && mha.attentionScores != nil && mha.attentionScores.Grad != nil {
		// dLoss/dq = dLoss/dAttentionScores @ k (untransposed)
		gradQ_per_head, err := mha.attentionScores.Grad.MatMul(mha.k)
		if err != nil {
			panic(fmt.Sprintf("failed to calculate gradQ in MHCA backward: %v", err))
		}
		if mha.q.RequiresGrad {
			if mha.q.Grad == nil {
				mha.q.Grad = NewTensor(mha.q.Shape, make([]float64, len(mha.q.Data)), false)
			}
			for i := range mha.q.Grad.Data {
				mha.q.Grad.Data[i] += gradQ_per_head.Data[i]
			}
		}

		// dLoss/dk = dLoss/dAttentionScores^T @ q (untransposed)
		attentionScoresGradTransposed, err := mha.attentionScores.Grad.Transpose(len(mha.attentionScores.Grad.Shape)-2, len(mha.attentionScores.Grad.Shape)-1)
		if err != nil {
			panic(fmt.Sprintf("failed to transpose attentionScores.Grad for MHCA backward: %v", err))
		}

		gradK_per_head, err := attentionScoresGradTransposed.MatMul(mha.q)
		if err != nil {
			panic(fmt.Sprintf("failed to calculate gradK in MHCA backward: %v", err))
		}
		if mha.k != nil && mha.k.RequiresGrad {
			if mha.k.Grad == nil {
				mha.k.Grad = NewTensor(mha.k.Shape, make([]float64, len(mha.k.Data)), false)
			}
			for i := range mha.k.Grad.Data {
				mha.k.Grad.Data[i] += gradK_per_head.Data[i]
			}
		}
	}

	// --- 6. Combine gradients from heads ---
	// Sum mha.q.Grad over heads
	var gradQCombined *Tensor
	if mha.q != nil && mha.q.Grad != nil && mha.queryTensor != nil {
		batchSizeQ := mha.q.Grad.Shape[0]
		numHeadsQ := mha.NumQHeads
		querySeqLenQ := mha.q.Grad.Shape[2]
		depthQ := mha.Depth
		dimModelQ := mha.DimModel

		combinedShapeQ := []int{batchSizeQ, querySeqLenQ, dimModelQ}
		gradQCombinedData := make([]float64, batchSizeQ*querySeqLenQ*dimModelQ)
		gradQCombined = NewTensor(combinedShapeQ, gradQCombinedData, false)

		qGradFlatIndex := 0
		for b := 0; b < batchSizeQ; b++ {
			for h := 0; h < numHeadsQ; h++ {
				for s := 0; s < querySeqLenQ; s++ {
					for d := 0; d < depthQ; d++ {
						qGradFlatIndex = b*numHeadsQ*querySeqLenQ*depthQ + h*querySeqLenQ*depthQ + s*depthQ + d
						combinedFlatIndex := b*querySeqLenQ*dimModelQ + s*dimModelQ + (h*depthQ + d)
						if qGradFlatIndex >= len(mha.q.Grad.Data) || combinedFlatIndex >= len(gradQCombined.Data) {
							panic("index out of bounds during gradient summation over heads for Q")
						}
						gradQCombined.Data[combinedFlatIndex] += mha.q.Grad.Data[qGradFlatIndex]
					}
				}
			}
		}
		// Accumulate gradQCombined to queryTensor.Grad if queryTensor requires grad
		if mha.queryTensor.RequiresGrad {
			if mha.queryTensor.Grad == nil {
				mha.queryTensor.Grad = NewTensor(mha.queryTensor.Shape, make([]float64, len(mha.queryTensor.Data)), false)
			}
			for i := range mha.queryTensor.Grad.Data {
				mha.queryTensor.Grad.Data[i] += gradQCombined.Data[i]
			}
		}
	}

	// Sum mha.k.Grad over heads
	var gradKCombined *Tensor
	if mha.k != nil && mha.k.Grad != nil && mha.keyTensor != nil {
		batchSizeK := mha.k.Grad.Shape[0]
		numHeadsK := mha.NumKVHeads
		keySeqLenK := mha.k.Grad.Shape[2] // Assuming k.Grad has the same sequence length as k
		depthK := mha.DimKVHeads
		dimModelK := mha.DimModel // Combined gradient is dimModel

		combinedShapeK := []int{batchSizeK, keySeqLenK, dimModelK}
		gradKCombinedData := make([]float64, batchSizeK*keySeqLenK*dimModelK)
		gradKCombined = NewTensor(combinedShapeK, gradKCombinedData, false)

		kGradFlatIndex := 0
		for b := 0; b < batchSizeK; b++ {
			for h := 0; h < numHeadsK; h++ {
				for s := 0; s < keySeqLenK; s++ {
					for d := 0; d < depthK; d++ {
						kGradFlatIndex = b*numHeadsK*keySeqLenK*depthK + h*keySeqLenK*depthK + s*depthK + d
						combinedFlatIndexK := b*keySeqLenK*dimModelK + s*dimModelK + (h*depthK + d)
						if kGradFlatIndex >= len(mha.k.Grad.Data) || combinedFlatIndexK >= len(gradKCombined.Data) {
							panic("index out of bounds during gradient summation over heads for K")
						}
						gradKCombined.Data[combinedFlatIndexK] += mha.k.Grad.Data[kGradFlatIndex]
					}
				}
			}
		}
		// Accumulate gradKCombined to keyTensor.Grad if keyTensor requires grad
		if mha.keyTensor.RequiresGrad {
			if mha.keyTensor.Grad == nil {
				mha.keyTensor.Grad = NewTensor(mha.keyTensor.Shape, make([]float64, len(mha.keyTensor.Data)), false)
			}
			for i := range mha.keyTensor.Grad.Data {
				mha.keyTensor.Grad.Data[i] += gradKCombined.Data[i]
			}
		}
	}

	// Sum mha.v.Grad over heads
	var gradVCombined *Tensor
	if mha.v != nil && mha.v.Grad != nil && mha.valueTensor != nil {
		batchSizeV := mha.v.Grad.Shape[0]
		numHeadsV := mha.NumKVHeads
		keySeqLenV := mha.v.Grad.Shape[2] // Assuming v.Grad has the same sequence length as v
		depthV := mha.DimKVHeads
		dimModelV := mha.DimModel // Combined gradient is dimModel

		combinedShapeV := []int{batchSizeV, keySeqLenV, dimModelV}
		gradVCombinedData := make([]float64, batchSizeV*keySeqLenV*dimModelV)
		gradVCombined = NewTensor(combinedShapeV, gradVCombinedData, false)

		vGradFlatIndex := 0
		for b := 0; b < batchSizeV; b++ {
			for h := 0; h < numHeadsV; h++ {
				for s := 0; s < keySeqLenV; s++ {
					for d := 0; d < depthV; d++ {
						vGradFlatIndex = b*numHeadsV*keySeqLenV*depthV + h*keySeqLenV*depthV + s*depthV + d
						combinedFlatIndexV := b*keySeqLenV*dimModelV + s*dimModelV + (h*depthV + d)
						if vGradFlatIndex >= len(mha.v.Grad.Data) || combinedFlatIndexV >= len(gradVCombined.Data) {
							panic("index out of bounds during gradient summation over heads for V")
						}
						gradVCombined.Data[combinedFlatIndexV] += mha.v.Grad.Data[vGradFlatIndex]
					}
				}
			}
		}
		// Accumulate gradVCombined to valueTensor.Grad if valueTensor requires grad
		if mha.valueTensor.RequiresGrad {
			if mha.valueTensor.Grad == nil {
				mha.valueTensor.Grad = NewTensor(mha.valueTensor.Shape, make([]float64, len(mha.valueTensor.Data)), false)
			}
			for i := range mha.valueTensor.Grad.Data {
				mha.valueTensor.Grad.Data[i] += gradVCombined.Data[i]
			}
		}
	}

	// --- 7. Backpropagate through the initial linear projections (QueryLinear, KeyLinear, ValueLinear) ---
	// Backpropagate gradQCombined to Weights of QueryLinear
	// dLoss/dWq = queryTensor^T @ gradQCombined
	if mha.queryTensor != nil && mha.QueryLinear != nil && mha.QueryLinear.Weights.RequiresGrad && gradQCombined != nil {
		if mha.QueryLinear.Weights.Grad == nil {
			mha.QueryLinear.Weights.Grad = NewTensor(mha.QueryLinear.Weights.Shape, make([]float64, len(mha.QueryLinear.Weights.Data)), false)
		}
		queryTransposedForWq, err := mha.queryTensor.Transpose(1, 2)
		if err != nil {
			panic(fmt.Sprintf("failed to transpose queryTensor for MHCA Wq backward: %v", err))
		}

		batchSize := mha.queryTensor.Shape[0]
		dimModel := mha.DimModel
		querySeqLen := mha.queryTensor.Shape[1]

		tempGradWqShape := []int{batchSize, dimModel, dimModel}
		tempGradWqData := make([]float64, batchSize*dimModel*dimModel)
		tempGradWq := NewTensor(tempGradWqShape, tempGradWqData, false)

		for b := 0; b < batchSize; b++ {
			querySlice := queryTransposedForWq.Data[b*dimModel*querySeqLen : (b+1)*dimModel*querySeqLen]
			gradQSlice := gradQCombined.Data[b*querySeqLen*dimModel : (b+1)*querySeqLen*dimModel]
			outputSlice := tempGradWq.Data[b*dimModel*dimModel : (b+1)*dimModel*dimModel]

			M := dimModel
			K := querySeqLen
			N := dimModel
			for i := 0; i < M; i++ {
				for j := 0; j < N; j++ {
					sum := 0.0
					for k := 0; k < K; k++ {
						sum += querySlice[i*K+k] * gradQSlice[k*N+j]
					}
					outputSlice[i*N+j] = sum
				}
			}
		}

		finalGradWqData := make([]float64, dimModel*dimModel)
		for b := 0; b < batchSize; b++ {
			batchStart := b * dimModel * dimModel
			for i := 0; i < dimModel*dimModel; i++ {
				finalGradWqData[i] += tempGradWq.Data[batchStart+i]
			}
		}
		finalGradWq := NewTensor([]int{dimModel, dimModel}, finalGradWqData, false)

		for i := range mha.QueryLinear.Weights.Grad.Data {
			mha.QueryLinear.Weights.Grad.Data[i] += finalGradWq.Data[i]
		}
	}

	// Backpropagate gradKCombined to Weights of KeyLinear
	// dLoss/dWk = keyTensor^T @ gradKCombined
	if mha.keyTensor != nil && mha.KeyLinear != nil && mha.KeyLinear.Weights.RequiresGrad && gradKCombined != nil {
		if mha.KeyLinear.Weights.Grad == nil {
			mha.KeyLinear.Weights.Grad = NewTensor(mha.KeyLinear.Weights.Shape, make([]float64, len(mha.KeyLinear.Weights.Data)), false)
		}
		keyTransposedForWk, err := mha.keyTensor.Transpose(1, 2)
		if err != nil {
			panic(fmt.Sprintf("failed to transpose keyTensor for MHCA Wk backward: %v", err))
		}

		batchSize := mha.keyTensor.Shape[0]
		dimModel := mha.DimModel
		keySeqLen := mha.keyTensor.Shape[1]

		tempGradWkShape := []int{batchSize, dimModel, dimModel}
		tempGradWkData := make([]float64, batchSize*dimModel*dimModel)
		tempGradWk := NewTensor(tempGradWkShape, tempGradWkData, false)

		for b := 0; b < batchSize; b++ {
			keySlice := keyTransposedForWk.Data[b*dimModel*keySeqLen : (b+1)*dimModel*keySeqLen]
			gradKSlice := gradKCombined.Data[b*keySeqLen*dimModel : (b+1)*keySeqLen*dimModel]
			outputSlice := tempGradWk.Data[b*dimModel*dimModel : (b+1)*dimModel*dimModel]

			M := dimModel
			K := keySeqLen
			N := dimModel
			for i := 0; i < M; i++ {
				for j := 0; j < N; j++ {
					sum := 0.0
					for k := 0; k < K; k++ {
						sum += keySlice[i*K+k] * gradKSlice[k*N+j]
					}
					outputSlice[i*N+j] = sum
				}
			}
		}
		finalGradWkData := make([]float64, dimModel*dimModel)
		for b := 0; b < batchSize; b++ {
			batchStart := b * dimModel * dimModel
			for i := 0; i < dimModel*dimModel; i++ {
				finalGradWkData[i] += tempGradWk.Data[batchStart+i]
			}
		}
		finalGradWk := NewTensor([]int{dimModel, dimModel}, finalGradWkData, false)

		for i := range mha.KeyLinear.Weights.Grad.Data {
			mha.KeyLinear.Weights.Grad.Data[i] += finalGradWk.Data[i]
		}
	}

	// Backpropagate gradVCombined to Weights of ValueLinear
	// dLoss/dWv = valueTensor^T @ gradVCombined
	if mha.valueTensor != nil && mha.ValueLinear != nil && mha.ValueLinear.Weights.RequiresGrad && gradVCombined != nil {
		if mha.ValueLinear.Weights.Grad == nil {
			mha.ValueLinear.Weights.Grad = NewTensor(mha.ValueLinear.Weights.Shape, make([]float64, len(mha.ValueLinear.Weights.Data)), false)
		}
		valueTransposedForWv, err := mha.valueTensor.Transpose(1, 2)
		if err != nil {
			panic(fmt.Sprintf("failed to transpose valueTensor for MHCA Wv backward: %v", err))
		}

		batchSize := mha.valueTensor.Shape[0]
		dimModel := mha.DimModel
		keySeqLen := mha.valueTensor.Shape[1] // Use keySeqLen for the sequence length of Key/Value tensors

		tempGradWvShape := []int{batchSize, dimModel, dimModel}
		tempGradWvData := make([]float64, batchSize*dimModel*dimModel)
		tempGradWv := NewTensor(tempGradWvShape, tempGradWvData, false)

		for b := 0; b < batchSize; b++ {
			valueSlice := valueTransposedForWv.Data[b*dimModel*keySeqLen : (b+1)*dimModel*keySeqLen]
			gradVSlice := gradVCombined.Data[b*keySeqLen*dimModel : (b+1)*keySeqLen*dimModel]
			outputSlice := tempGradWv.Data[b*dimModel*dimModel : (b+1)*dimModel*dimModel]

			M := dimModel
			K := keySeqLen
			N := dimModel
			for i := 0; i < M; i++ {
				for j := 0; j < N; j++ {
					sum := 0.0
					for k := 0; k < K; k++ {
						sum += valueSlice[i*K+k] * gradVSlice[k*N+j]
					}
					outputSlice[i*N+j] = sum
				}
			}
		}

		finalGradWvData := make([]float64, dimModel*dimModel)
		for b := 0; b < batchSize; b++ {
			batchStart := b * dimModel * dimModel
			for i := 0; i < dimModel*dimModel; i++ {
				finalGradWvData[i] += tempGradWv.Data[batchStart+i]
			}
		}
		finalGradWv := NewTensor([]int{dimModel, dimModel}, finalGradWvData, false)

		for i := range mha.ValueLinear.Weights.Grad.Data {
			mha.ValueLinear.Weights.Grad.Data[i] += finalGradWv.Data[i]
		}
	}

	// Backpropagate gradients to the original input tensors (queryTensor, keyTensor, valueTensor) ---
	// These gradients are already accumulated in queryTensor.Grad, keyTensor.Grad, valueTensor.Grad
	// by the Backward calls on mha.QueryLinear, mha.KeyLinear, mha.ValueLinear.
	// You don't need to manually calculate and accumulate them again here if you rely on the
	// Tensor.Backward traversal via the creator chain.
	return nil
}

// Forward performs the forward pass of the MultiHeadCrossAttention layer.
// query: Input from the decoder layer (shape: [batch_size, target_sequence_length, dim_model]).
// key/value: Input from the encoder output (shape: [batch_size, source_sequence_length, dim_model]).
// mask: Optional mask for attention (e.g., padding mask for encoder output).
func (mha *MultiHeadCrossAttention) Forward(inputs ...*Tensor) (*Tensor, error) { // Changed to accept variadic inputs
	// Expect 3 or 4 inputs (query, key, value, and optional mask)
	if len(inputs) < 3 || len(inputs) > 4 {
		return nil, fmt.Errorf("MultiHeadCrossAttention.Forward expects 3 or 4 inputs (query, key, value, optional mask), got %d", len(inputs))
	}

	query := inputs[0]
	key := inputs[1]
	value := inputs[2]
	var mask *Tensor // Declare mask as optional

	if len(inputs) == 4 {
		mask = inputs[3] // Extract mask if provided
	}

	// Store the original input tensors
	mha.queryTensor = query
	mha.keyTensor = key
	mha.valueTensor = value

	// ... (rest of your forward pass logic, which uses query, key, value, and mask) ...

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

	qTransposed, err := qReshaped.Transpose(1, 2) // Transpose to [batch_size, num_heads, q_seq_length, head_dim]
	if err != nil {
		return nil, fmt.Errorf("failed to transpose cross-attention query tensor: %w", err)
	}
	mha.q = qTransposed

	// K, V shapes: [batch_size, num_kv_heads, kv_seq_length, head_dim]
	kReshaped, err := k.Reshape([]int{batchSize, kvSeqLength, mha.NumKVHeads, mha.DimKVHeads}) // Use DimKVHeads
	if err != nil {
		return nil, fmt.Errorf("failed to reshape cross-attention key tensor: %w", err)
	}
	kTransposed, err := kReshaped.Transpose(1, 2) // Transpose to [batch_size, num_heads, kv_seq_length, head_dim]
	if err != nil {
		return nil, fmt.Errorf("failed to transpose cross-attention key tensor: %w", err)
	}
	mha.k = kTransposed

	vReshaped, err := v.Reshape([]int{batchSize, kvSeqLength, mha.NumKVHeads, mha.DimKVHeads}) // Use DimKVHeads
	if err != nil {
		return nil, fmt.Errorf("failed to reshape cross-attention value tensor: %w", err)
	}
	vTransposed, err := vReshaped.Transpose(1, 2) // Transpose to [batch_size, num_heads, kv_seq_length, head_dim]
	if err != nil {
		return nil, fmt.Errorf("failed to transpose cross-attention value tensor: %w", err)
	}
	mha.v = vTransposed

	// Calculate attention scores: Q @ K^T
	// K^T shape: [batch_size, num_kv_heads, head_dim, kv_seq_length]
	kT_Transposed, err := kTransposed.Transpose(len(kTransposed.Shape)-2, len(kTransposed.Shape)-1)
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
	mha.attentionScores = attentionScores // Store attention scores

	// Scale attention scores
	scale := 1.0 / math.Sqrt(float64(mha.DimModel/mha.NumQHeads)) // Scale by sqrt of head dim
	scaledAttentionScores, err := attentionScores.MulScalar(scale)
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
		return nil, fmt.Errorf("failed to apply softmax to attention scores: %w", err)
	}
	mha.attentionWeights = attentionWeights // Store attention weights

	// Apply attention weights to Value: Attention Weights @ V
	contextLayer, err := attentionWeights.MatMul(vTransposed)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate context layer (Attention@V): %w", err)
	}

	mha.attentionOutputBeforeConcat = contextLayer // Store attention output before concatenation

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

	// Store the final output tensor
	mha.attentionOutput = output // Store the final output

	// Set creator and RequiresGrad for the output tensor
	outputRequiresGrad := query.RequiresGrad || key.RequiresGrad || value.RequiresGrad ||
		mha.QueryLinear.Weights.RequiresGrad || (mha.QueryLinear.Biases != nil && mha.QueryLinear.Biases.RequiresGrad) ||
		mha.KeyLinear.Weights.RequiresGrad || (mha.KeyLinear.Biases != nil && mha.KeyLinear.Biases.RequiresGrad) ||
		mha.ValueLinear.Weights.RequiresGrad || (mha.ValueLinear.Biases != nil && mha.ValueLinear.Biases.RequiresGrad) ||
		mha.OutputLinear.Weights.RequiresGrad || (mha.OutputLinear.Biases != nil && mha.OutputLinear.Biases.RequiresGrad)

	output.RequiresGrad = outputRequiresGrad
	if output.RequiresGrad {
		output.Creator = mha // Set the creator to the MultiHeadCrossAttention layer itself
	}

	mha.attentionOutput = output

	return output, nil
}
