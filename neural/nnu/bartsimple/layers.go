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
	input   *Tensor // Store input for backward pass
}

// NewLinear creates a new Linear layer with random weights and zero biases.
func NewLinear(inputDim, outputDim int) (*Linear, error) {
	// Initialize weights with random values (you might want a better initialization later)
	weightsData := make([]float64, inputDim*outputDim)
	for i := range weightsData {
		weightsData[i] = float64(i) * 0.01 // Simple non-zero initialization
	}
	weights := NewTensor(weightsData, []int{inputDim, outputDim}, true)

	// Initialize biases with zeros
	biasesData := make([]float64, outputDim)
	biases := NewTensor(biasesData, []int{outputDim}, true)

	return &Linear{Weights: weights, Biases: biases}, nil
}

// Forward performs the forward pass of the Linear layer.
func (l *Linear) Forward(input *Tensor) (*Tensor, error) {
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
		reshapedInput := NewTensor(reshapedInputData, []int{batchSize * seqLength, inputDim}, true)

		// Perform matrix multiplication: [batch_size * sequence_length, input_dim] @ [input_dim, output_dim]
		output2D, err := reshapedInput.MatMul(l.Weights)
		if err != nil {
			return nil, fmt.Errorf("linear layer 3D matrix multiplication failed: %w", err)
		}

		// Reshape output back to 3D
		outputData := make([]float64, batchSize*seqLength*outputDim)
		copy(outputData, output2D.Data) // Create a copy
		output = NewTensor(outputData, []int{batchSize, seqLength, outputDim}, true)

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

	// Set creator and requiresGrad for the output tensor
	output.requiresGrad = input.requiresGrad || l.Weights.requiresGrad || (l.Biases != nil && l.Biases.requiresGrad)
	if output.requiresGrad {
		// In a real scenario, you would create an operation struct
		// that encapsulates the MatMul and AddWithBroadcast operations
		// and set that operation as the creator.
		// For simplicity here, we'll just set the creator to nil, meaning this is
		// a composite operation whose backward pass is handled by the layer itself.
		// A more robust design would define LinearOperation struct.
		// Setting creator to nil prevents automatic backward traversal
		// beyond this point via Tensor.Backward().
		// The layer's Backward method must manually handle the gradients.
		output.creator = nil // This layer is the 'creator' conceptually, but it's a composite op
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
	if l.Weights.requiresGrad {
		if l.Weights.Grad == nil {
			l.Weights.Grad = NewTensor(make([]float64, len(l.Weights.Data)), l.Weights.Shape, false)
		}
	}
	if l.Biases != nil && l.Biases.requiresGrad {
		if l.Biases.Grad == nil {
			l.Biases.Grad = NewTensor(make([]float64, len(l.Biases.Data)), l.Biases.Shape, false)
		}
	}

	// --- Calculate Gradient with respect to Weights (dLoss/dWeights) ---
	// dLoss/dWeights = Input^T @ dLoss/dOutput (grad)
	// Input shape: [batch_size, sequence_length, input_dim] or [batch_size, input_dim]
	// grad shape: [batch_size, sequence_length, output_dim] or [batch_size, output_dim]
	var inputTranspose *Tensor
	var err error

	switch len(l.input.Shape) {
	case 2:
		// Input is 2D [batch_size, input_dim]
		// Transpose input: [input_dim, batch_size]
		inputTranspose, err = l.input.Transpose(0, 1)
		if err != nil {
			return fmt.Errorf("linear layer backward: failed to transpose input (2D): %w", err)
		}
		// dLoss/dWeights: [input_dim, batch_size] @ [batch_size, output_dim] -> [input_dim, output_dim]
		dWeights, err := inputTranspose.MatMul(grad)
		if err != nil {
			return fmt.Errorf("linear layer backward: failed to calculate dLoss/dWeights (2D): %w", err)
		}
		// Accumulate gradient
		for i := range l.Weights.Grad.Data {
			l.Weights.Grad.Data[i] += dWeights.Data[i]
		}

	case 3:
		// Input is 3D [batch_size, sequence_length, input_dim]
		// Reshape input and grad for batch matrix multiplication
		batchSize := l.input.Shape[0]
		seqLength := l.input.Shape[1]
		inputDim := l.input.Shape[2]
		outputDim := l.Weights.Shape[1]

		reshapedInputData := make([]float64, batchSize*seqLength*inputDim)
		copy(reshapedInputData, l.input.Data)
		reshapedInput := NewTensor(reshapedInputData, []int{batchSize * seqLength, inputDim}, false) // Gradient calculation, no need for grad here

		reshapedGradData := make([]float64, batchSize*seqLength*outputDim)
		copy(reshapedGradData, grad.Data)
		reshapedGrad := NewTensor(reshapedGradData, []int{batchSize * seqLength, outputDim}, false) // Gradient calculation, no need for grad here

		// Transpose reshaped input: [input_dim, batch_size * sequence_length]
		inputTranspose, err = reshapedInput.Transpose(0, 1)
		if err != nil {
			return fmt.Errorf("linear layer backward: failed to transpose reshaped input (3D): %w", err)
		}

		// dLoss/dWeights: [input_dim, batch_size * sequence_length] @ [batch_size * sequence_length, output_dim] -> [input_dim, output_dim]
		dWeights, err := inputTranspose.MatMul(reshapedGrad)
		if err != nil {
			return fmt.Errorf("linear layer backward: failed to calculate dLoss/dWeights (3D): %w", err)
		}
		// Accumulate gradient
		for i := range l.Weights.Grad.Data {
			l.Weights.Grad.Data[i] += dWeights.Data[i]
		}

	default:
		return fmt.Errorf("linear layer backward only supports 2D or 3D input, got %d dimensions", len(l.input.Shape))
	}

	// --- Calculate Gradient with respect to Bias (dLoss/dBias) ---
	// dLoss/dBias = Sum(dLoss/dOutput) (grad) over batch and sequence dimensions
	// dLoss/dBias = Sum(dLoss/dOutput) over batch and sequence dimensions
	if l.Biases != nil && l.Bias.requiresGrad && grad != nil {
		if l.Biases.Grad == nil { // Corrected: Check l.Biases.Grad instead of l.Bias.Grad
			// Initialize Biases gradient tensor with zeros
			l.Biases.Grad = NewTensor(make([]float64, len(l.Biases.Data)), l.Biases.Shape, false) // Gradient itself does not require gradients
		}

		// Assuming grad has shape [batch_size, ..., last_dim] and Biases has shape [last_dim]
		// We need to sum over all dimensions of grad except the last one.

		// Calculate the size of the last dimension
		lastDimSize := grad.Shape[len(grad.Shape)-1]

		// Iterate through the flattened gradient data and sum for each bias term
		for i := 0; i < len(grad.Data); i++ {
			// Calculate the index in the bias gradient
			// This is the flattened index modulo the size of the last dimension
			biasIndex := i % lastDimSize
			l.Biases.Grad.Data[biasIndex] += grad.Data[i] // Accumulate gradient
		}
	}

	// --- Calculate Gradient with respect to Input (dLoss/dInput) ---
	// dLoss/dInput = dLoss/dOutput (grad) @ Weights^T
	// grad shape: [batch_size, sequence_length, output_dim] or [batch_size, output_dim]
	// Weights shape: [input_dim, output_dim]
	// Weights^T shape: [output_dim, input_dim]

	if l.input.requiresGrad {
		if l.input.Grad == nil {
			l.input.Grad = NewTensor(make([]float64, len(l.input.Data)), l.input.Shape, false)
		}

		weightsTranspose, err := l.Weights.Transpose(0, 1)
		if err != nil {
			return fmt.Errorf("linear layer backward: failed to transpose weights: %w", err)
		}

		var dInput *Tensor
		switch len(grad.Shape) {
		case 2:
			// grad is 2D [batch_size, output_dim]
			// dLoss/dInput: [batch_size, output_dim] @ [output_dim, input_dim] -> [batch_size, input_dim]
			dInput, err = grad.MatMul(weightsTranspose)
			if err != nil {
				return fmt.Errorf("linear layer backward: failed to calculate dLoss/dInput (2D): %w", err)
			}
		case 3:
			// grad is 3D [batch_size, sequence_length, output_dim]
			// Reshape grad for batch matrix multiplication
			batchSize := grad.Shape[0]
			seqLength := grad.Shape[1]
			outputDim := grad.Shape[2]
			inputDim := l.Weights.Shape[0]

			reshapedGradData := make([]float64, batchSize*seqLength*outputDim)
			copy(reshapedGradData, grad.Data)
			reshapedGrad := NewTensor(reshapedGradData, []int{batchSize * seqLength, outputDim}, false)

			// dLoss/dInput: [batch_size * sequence_length, output_dim] @ [output_dim, input_dim] -> [batch_size * sequence_length, input_dim]
			dInput2D, err := reshapedGrad.MatMul(weightsTranspose)
			if err != nil {
				return fmt.Errorf("linear layer backward: failed to calculate dLoss/dInput (3D): %w", err)
			}
			// Reshape dInput back to 3D
			dInputData := make([]float64, batchSize*seqLength*inputDim)
			copy(dInputData, dInput2D.Data)
			dInput = NewTensor(dInputData, []int{batchSize, seqLength, inputDim}, false)

		default:
			return fmt.Errorf("linear layer backward only supports 2D or 3D gradient, got %d dimensions", len(grad.Shape))
		}

		// Accumulate gradient for input
		for i := range l.input.Grad.Data {
			l.input.Grad.Data[i] += dInput.Data[i]
		}
	}

	return output, nil
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

	inputShape []int // Store input shape for backward
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
	gamma := NewTensor(gammaData, []int{dimModel}, true)
	beta := NewTensor(betaData, []int{dimModel}, true)

	return &LayerNormalization{
		Gamma:   gamma,
		Beta:    beta,
		Epsilon: 1e-6, // Small epsilon value
	}
}

// Backward performs the backward pass for layer normalization.
// grad is the gradient from the output (dLoss/dOutput).
func (l *LayerNormalization) Backward(grad *Tensor) {
	if grad == nil || grad.Data == nil {
		// No gradient to propagate
		return
	}
	if l.normalizedInput == nil || l.invStdDev == nil || l.mean == nil {
		panic("LayerNormalization backward called before forward (intermediate values are nil)")
	}
	if l.Scale == nil || l.Bias == nil {
		panic("LayerNormalization scale or bias is nil in backward")
	}

	lastDimSize := l.inputShape[len(l.inputShape)-1]
	numElementsToNormalize := len(l.normalizedInput.Data) / lastDimSize

	// Ensure gradients are initialized
	if l.inputTensor != nil && l.inputTensor.requiresGrad { // Assuming inputTensor is stored in the struct
		if l.inputTensor.Grad == nil {
			l.inputTensor.Grad = NewTensor(make([]float64, len(l.inputTensor.Data)), l.inputTensor.Shape, false)
		}
	}
	if l.Scale.requiresGrad {
		if l.Scale.Grad == nil {
			l.Scale.Grad = NewTensor(make([]float64, len(l.Scale.Data)), l.Scale.Shape, false)
		}
	}
	if l.Bias.requiresGrad {
		if l.Bias.Grad == nil {
			l.Bias.Grad = NewTensor(make([]float64, len(l.Bias.Data)), l.Bias.Shape, false)
		}
	}

	// Calculate gradients for gamma (Scale) and beta (Bias)
	// dLoss/dBeta = Sum(grad) over all dims except the last one
	if l.Bias.requiresGrad {
		// Sum grad over all dimensions except the last one and add to l.Bias.Grad.Data
		// This is the same summation logic as in Linear.Backward for bias.
		for i := 0; i < len(grad.Data); i++ {
			biasIndex := i % lastDimSize
			l.Bias.Grad.Data[biasIndex] += grad.Data[i]
		}
	}

	// dLoss/dGamma = Sum(grad * normalized_input) over all dims except the last one
	if l.Scale.requiresGrad {
		if l.Scale.Grad == nil {
			l.Scale.Grad = NewTensor(make([]float64, len(l.Scale.Data)), l.Scale.Shape, false)
		}
		// Sum (grad * normalizedInput) over all dimensions except the last one
		for i := 0; i < numElementsToNormalize; i++ {
			for j := 0; j < lastDimSize; j++ {
				// Index in flattened data
				flatIndex := i*lastDimSize + j
				l.Scale.Grad.Data[j] += grad.Data[flatIndex] * l.normalizedInput.Data[flatIndex]
			}
		}
	}

	// Calculate gradient with respect to normalized input
	// dLoss/dNormalizedInput = grad * gamma (Scale)
	dLoss_dNormalizedInputData := make([]float64, len(grad.Data))
	for i := 0; i < numElementsToNormalize; i++ {
		for j := 0; j < lastDimSize; j++ {
			flatIndex := i*lastDimSize + j
			dLoss_dNormalizedInputData[flatIndex] = grad.Data[flatIndex] * l.Scale.Data[j]
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
		dLoss_dStdDevData[i] = sum_dL_dNorm_x_minus_mean * (-1.0 / (stdDev * stdDev)) // Derivative of 1/std_dev is -1/std_dev^2

		// dLoss/dMean
		// This involves two parts: from the (x - mean) term and from the mean in std_dev calculation.
		// Part 1: from (x - mean)
		part1_dLoss_dMean := sum_dL_dNorm * (-l.invStdDev.Data[i]) // Sum(dLoss/dNormalizedInput * -1/std_dev)

		// Part 2: from mean in std_dev calculation (more complex)
		// d(std_dev)/dMean = -Sum(x - mean) / (N * std_dev) = 0 because Sum(x - mean) = 0
		// so the derivative of std_dev with respect to mean is 0.
		// However, the variance calculation depends on the mean, and the derivative of sqrt(variance) wrt mean is not zero.
		// Let's use the standard formula for dLoss/dMean:
		// dLoss/dMean = Sum(dLoss/dNormalizedInput * -1/std_dev) + dLoss/dStdDev * dStdDev/dMean
		// dStdDev/dMean = -Sum(x - mean) / (N * std_dev) = 0

		// The correct dLoss/dMean comes from:
		// dLoss/dMean = Sum(dLoss/dOutput * gamma * (-1/std_dev))  (from dLoss/dNormalizedInput part)
		//             + dLoss/dVariance * dVariance/dMean         (from the variance part)
		// dVariance/dMean = Sum(2 * (x - mean) * -1) / N = -2 * Sum(x - mean) / N = 0

		// Let's use the simplified dLoss/dMean formula that combines terms:
		// dLoss/dMean = Sum(dLoss/dNormalizedInput * -1/std_dev) - (dLoss/dStdDev * Sum(x - mean)) / (N * std_dev^2)

		// The sum(x - mean) is zero, so the second term is zero.
		// dLoss/dMean = Sum(dLoss/dNormalizedInput * -1/std_dev)

		// Let's recalculate dLoss/dMean more directly from common formulas:
		// dLoss/dMean = Sum(dLoss/dNormalizedInput) * (-invStdDev) + dLoss/dInvStdDev * (-1/N) * invStdDev^3 * Sum(x-mean)
		// The second term is zero.
		// dLoss/dMean = Sum(dLoss/dNormalizedInput) * (-invStdDev)

		dLoss_dMeanData[i] = sum_dL_dNorm * (-l.invStdDev.Data[i])

	}

	// Calculate gradient with respect to input
	// dLoss/dx = dLoss/dNormalizedInput * (1 / std_dev)
	//           + dLoss/dStdDev * 2 * (x - mean) / (N * std_dev)
	//           + dLoss/dMean / N

	if l.inputTensor.requiresGrad {
		if l.inputTensor.Grad == nil {
			l.inputTensor.Grad = NewTensor(make([]float64, len(l.inputTensor.Data)), l.inputTensor.Shape, false)
		}
		for i := 0; i < numElementsToNormalize; i++ {
			stdDev := 1.0 / l.invStdDev.Data[i]
			for j := 0; j < lastDimSize; j++ {
				flatIndex := i*lastDimSize + j
				x_minus_mean := l.inputTensor.Data[flatIndex] - l.mean.Data[i]

				term1 := dLoss_dNormalizedInputData[flatIndex] * l.invStdDev.Data[i] // dLoss/dNormalizedInput * (1 / std_dev)

				term2 := dLoss_dStdDevData[i] * 2.0 * x_minus_mean * l.invStdDev.Data[i] * l.invStdDev.Data[i] * l.invStdDev.Data[i] / float64(lastDimSize) // dLoss/dStdDev * 2 * (x - mean) / (N * std_dev^3) ? No, the formula is different.

				// Let's use the combined formula for dLoss/dx:
				// dLoss/dx = (dLoss/dNormalizedInput * N * invStdDev + dLoss/dStdDev * (x - mean) * invStdDev^2 + dLoss/dMean * invStdDev * N) / N
				// dLoss/dx = dLoss/dNormalizedInput * invStdDev + dLoss/dStdDev * (x - mean) * invStdDev^2 / N + dLoss/dMean * invStdDev

				// Rechecking standard LayerNorm backward formulas:
				// dLoss/dx = (dLoss/dy * gamma) * (N * invStdDev * (N - 1) - (x - mean)^2 * invStdDev^3) / N^2  + Sum(dLoss/dy * gamma * (x - mean)) * 2 * (x - mean) * invStdDev^3 / N^2 + Sum(dLoss/dy * gamma) * (-invStdDev / N)

				// Simpler form:
				// dLoss/dx = dLoss/dNormalizedInput * invStdDev
				//           + dLoss/dVariance * 2 * (x - mean) / N
				//           + dLoss/dMean / N

				// Where dLoss/dVariance and dLoss/dMean are calculated from higher gradients.

				// dLoss/dVariance = Sum(dLoss/dOutput * gamma * (x - mean) * (-0.5) * (variance + epsilon)^(-1.5))
				// dLoss/dMean = Sum(dLoss/dOutput * gamma * (-invStdDev))

				// Let's use the common and relatively simpler backward pass derivation directly:
				// dLoss/dx_i = (1/N) * inv_std_dev * [N * dL/dnormalized_i - Sum(dL/dnormalized) - (x_i - mean) * inv_std_dev^2 * Sum(dL/dnormalized * (x - mean))]

				sum_dL_dNorm := 0.0              // Sum(dLoss/dNormalizedInput)
				sum_dL_dNorm_x_minus_mean := 0.0 // Sum(dLoss/dNormalizedInput * (x - mean))

				for k := 0; k < lastDimSize; k++ {
					flatIndex_k := i*lastDimSize + k
					sum_dL_dNorm += dLoss_dNormalizedInputData[flatIndex_k]
					sum_dL_dNorm_x_minus_mean += dLoss_dNormalizedInputData[flatIndex_k] * (l.inputTensor.Data[flatIndex_k] - l.mean.Data[i])
				}

				invStdDev_i := l.invStdDev.Data[i]
				x_i_minus_mean_i := l.inputTensor.Data[flatIndex] - l.mean.Data[i]

				// Calculate dLoss/dx_i
				dL_dx_i := invStdDev_i * (dLoss_dNormalizedInputData[flatIndex] - sum_dL_dNorm/float64(lastDimSize) - x_i_minus_mean_i*invStdDev_i*invStdDev_i*sum_dL_dNorm_x_minus_mean/float64(lastDimSize))

				l.inputTensor.Grad.Data[flatIndex] += dL_dx_i // Accumulate gradient

			}
		}
	}
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
func (l *LayerNormalization) Forward(input *Tensor) (*Tensor, error) {
	if input == nil || input.Data == nil {
		return nil, errors.New("input tensor is nil or has no data")
	}
	if len(input.Shape) == 0 {
		return nil, errors.New("input tensor cannot be a scalar for layer normalization")
	}
	if l.Scale == nil || l.Bias == nil {
		return nil, errors.New("layer normalization scale or bias is nil")
	}
	if len(l.Scale.Shape) != 1 || l.Scale.Shape[0] != input.Shape[len(input.Shape)-1] ||
		len(l.Bias.Shape) != 1 || l.Bias.Shape[0] != input.Shape[len(input.Shape)-1] {
		return nil, fmt.Errorf("layer normalization scale/bias shape mismatch with input last dimension: %v vs %d", l.Scale.Shape, input.Shape[len(input.Shape)-1])
	}

	// Store input shape for backward pass
	l.inputShape = input.Shape

	// Calculate mean and variance across the last dimension
	lastDimSize := input.Shape[len(input.Shape)-1]
	numElementsToNormalize := len(input.Data) / lastDimSize // Number of elements to calculate mean/variance over for each feature

	meanData := make([]float64, numElementsToNormalize)
	varianceData := make([]float64, numElementsToNormalize)
	normalizedInputData := make([]float64, len(input.Data))

	for i := 0; i < numElementsToNormalize; i++ {
		// Calculate mean
		sum := 0.0
		for j := 0; j < lastDimSize; j++ {
			sum += input.Data[i*lastDimSize+j]
		}
		meanData[i] = sum / float64(lastDimSize)

		// Calculate variance
		sumSqDiff := 0.0
		for j := 0; j < lastDimSize; j++ {
			diff := input.Data[i*lastDimSize+j] - meanData[i]
			sumSqDiff += diff * diff
		}
		varianceData[i] = sumSqDiff / float64(lastDimSize)

		// Calculate normalized input
		invStdDev := 1.0 / math.Sqrt(varianceData[i]+l.Epsilon)
		l.invStdDev.Data[i] = invStdDev // Store inverse standard deviation

		for j := 0; j < lastDimSize; j++ {
			normalizedInputData[i*lastDimSize+j] = (input.Data[i*lastDimSize+j] - meanData[i]) * invStdDev
		}
	}

	// Store mean and normalized input
	l.mean = NewTensor(meanData, input.Shape[:len(input.Shape)-1], false)              // Mean has shape of input minus last dim
	l.invStdDev = NewTensor(l.invStdDev.Data, input.Shape[:len(input.Shape)-1], false) // InvStdDev has shape of input minus last dim
	l.normalizedInput = NewTensor(normalizedInputData, input.Shape, false)             // Normalized input has same shape as input

	// Scale and shift
	outputData := make([]float64, len(input.Data))
	for i := 0; i < numElementsToNormalize; i++ {
		for j := 0; j < lastDimSize; j++ {
			outputData[i*lastDimSize+j] = l.Scale.Data[j]*normalizedInputData[i*lastDimSize+j] + l.Bias.Data[j]
		}
	}

	outputTensor := NewTensor(outputData, input.Shape, input.requiresGrad || l.Scale.requiresGrad || l.Bias.requiresGrad)
	if outputTensor.requiresGrad {
		outputTensor.creator = l // Set creator to the layer itself
	}

	return outputTensor, nil
}

// MultiHeadAttention represents a multi-head attention layer.
type MultiHeadAttention struct {
    	Wq, Wk, Wv *Tensor // Linear layers for Q, K, V (learnable weights)
    	Wo         *Tensor // Output linear layer (learnable weights)
 NumHeads int
    	DimModel   int
 HeadDim int // dimModel / numHeads

    	// Stored intermediate tensors for backward pass
    	inputTensor *Tensor // Original input (Q, K, V are the same for self-attention)
    	q, k, v *Tensor // Q, K, V after linear projection and splitting heads
    	attentionScores *Tensor // Q @ K^T
    	attentionWeights *Tensor // Softmax(attentionScores) + Mask
 attentionOutputBeforeConcat *Tensor // attentionWeights @ V (before concatenating heads)
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
		QueryLinear:  queryLinear,
		KeyLinear:    keyLinear,
		ValueLinear:  valueLinear,
		OutputLinear: outputLinear,
	}, nil
}
// Backward performs the backward pass for multi-head self-attention.
// grad is the gradient from the output of the attention layer (after the final linear layer).
func (mha *MultiHeadAttention) Backward(grad *Tensor) {
	if grad == nil || grad.Data == nil {
		return // No gradient to propagate
	}

	// Add checks for nil intermediate tensors and weights
	// Initialize gradients for mha.Wq.Grad, mha.Wk.Grad, mha.Wv.Grad, mha.Wo.Grad, and mha.inputTensor.Grad
	// if requiresGrad is true and Grad is nil.


	// --- 1. Backpropagate through the final linear layer (Wo) ---
	// The incoming gradient 'grad' is w.r.t. the output of Wo.
	// Call Backward on Wo with 'grad'. This calculates gradients for Wo's parameters
	// and the gradient w.r.t. Wo's input (mha.attentionOutput), accumulating it in mha.attentionOutput.Grad.
	if mha.Wo != nil { // Assuming Wo is a Linear layer
		mha.Wo.Backward(grad)
	}

// --- 2. Get the gradient before concatenation/reshape ---
	// The gradient w.r.t. mha.attentionOutput is in mha.attentionOutput.Grad.
	// Reshape this gradient from [batch_size, seq_len, dim_model] to [batch_size, num_heads, seq_len, depth].
	// This is the gradient w.r.t. the output of MatMul(attentionWeights @ V) for each head before concatenation.
	// Implement reshaping of mha.attentionOutput.Grad here. Shape: [b, h, s, d]

	if mha.attentionOutput == nil || mha.attentionOutput.Grad == nil {
		panic("mha.attentionOutput or its gradient is nil in reshape step")
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
	gradBeforeConcat := NewTensor(gradBeforeConcatData, reshapedShape, false) // Gradient tensor does not require gradients

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
		vTransposed, err := mha.v.Transpose(len(mha.v.Shape)-2, len(mha.v.Shape)-1)
		if err != nil {
			panic(fmt.Sprintf("failed to transpose v for MHA backward: %v", err))
		}

		// Need a batched MatMul operation: (b, h, s, d) @ (b, h, d, s) -> (b, h, s, s)
		gradAttentionWeights, err := gradBeforeConcat.MatMul(vTransposed) // Assuming MatMul handles batches
		if err != nil {
			panic(fmt.Sprintf("failed to calculate gradAttentionWeights in MHA backward: %v", err))
		}

		// Add to existing gradient for attention weights
		if mha.attentionWeights.requiresGrad {
			if mha.attentionWeights.Grad == nil {
				mha.attentionWeights.Grad = NewTensor(make([]float64, len(mha.attentionWeights.Data)), mha.attentionWeights.Shape, false)
			}
			// Accumulate gradAttentionWeights to mha.attentionWeights.Grad element-wise
			for i := range mha.attentionWeights.Grad.Data {
				mha.attentionWeights.Grad.Data[i] += gradAttentionWeights.Data[i]
			}
		}


		// dLoss/dv = mha.attentionWeights^T @ gradBeforeConcat
		// Transpose mha.attentionWeights: swap last two dimensions (seq_len and seq_len)
		attentionWeightsTransposed, err := mha.attentionWeights.Transpose(len(mha.attentionWeights.Shape)-2, len(mha.attentionWeights.Shape)-1)
		if err != nil {
			panic(fmt.Sprintf("failed to transpose attentionWeights for MHA backward: %v", err))
		}

		// Need a batched MatMul operation: (b, h, s, s) @ (b, h, s, d) -> (b, h, s, d)
		gradV, err := attentionWeightsTransposed.MatMul(gradBeforeConcat) // Assuming MatMul handles batches
		if err != nil {
			panic(fmt.Sprintf("failed to calculate gradV in MHA backward: %v", err))
		}

		// Add to existing gradient for v
		if mha.v != nil && mha.v.requiresGrad { // Check if v requires grad (it's an intermediate tensor from input projection)
			if mha.v.Grad == nil {
				mha.v.Grad = NewTensor(make([]float64, len(mha.v.Data)), mha.v.Shape, false)
			}
			// Accumulate gradV to mha.v.Grad element-wise
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
    		mha.attentionWeights.Backward(mha.attentionWeights.Grad)
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
		if mha.q.requiresGrad { // Check if q requires grad
			if mha.q.Grad == nil {
				mha.q.Grad = NewTensor(make([]float64, len(mha.q.Data)), mha.q.Shape, false)
			}
			for i := range mha.q.Grad.Data {
				mha.q.Grad.Data[i] += gradQ_per_head.Data[i]
			}
		}


		// dLoss/dk = q^T @ dLoss/dAttentionScores
		// Transpose mha.q: swap last two dimensions (seq_len and depth)
		qTransposedForGradK, err := mha.q.Transpose(len(mha.q.Shape)-2, len(mha.q.Shape)-1)
		if err != nil {
			panic(fmt.Sprintf("failed to transpose q for MHA backward: %v", err))
		}

		// Need a batched MatMul operation: [b, h, d, s] @ [b, h, s, s] -> [b, h, d, s] ? No, [b, h, s, d]
		// The result shape should be [b, h, s, d] for gradK.
		// MatMul(A^T @ C) where A=q, C=attentionScores.Grad
		// Shape: [b, h, d, s] @ [b, h, s, s] -> [b, h, d, s]
		// We need [b, h, s, d]. The formula is dLoss/dk = (Q^T @ dLoss/dAttentionScores)^T = dLoss/dAttentionScores^T @ Q.
		// Let's use the correct formula: dLoss/dk = dLoss/dAttentionScores^T @ q (untransposed)
		attentionScoresGradTransposed, err := mha.attentionScores.Grad.Transpose(len(mha.attentionScores.Grad.Shape)-2, len(mha.attentionScores.Grad.Shape)-1)
		if err != nil {
			panic(fmt.Sprintf("failed to transpose attentionScores.Grad for MHA backward: %v", err))
		}

		// Need a batched MatMul operation: [b, h, s, s] @ [b, h, s, d] -> [b, h, s, d]
		gradK_per_head, err := attentionScoresGradTransposed.MatMul(mha.q) // Batched MatMul
		if err != nil {
			panic(fmt.Sprintf("failed to calculate gradK in MHA backward: %v", err))
		}
		// Accumulate gradK_per_head to mha.k.Grad
		if mha.k != nil && mha.k.requiresGrad { // Check if k requires grad
			if mha.k.Grad == nil {
				mha.k.Grad = NewTensor(make([]float64, len(mha.k.Data)), mha.k.Shape, false)
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

	if mha.q != nil && mha.k != nil && mha.v != nil && mha.q.Grad != nil && mha.k.Grad != nil && mha.v.Grad != nil && mha.inputTensor != nil && mha.inputTensor.requiresGrad {

		batchSize := mha.q.Grad.Shape[0]
		numHeads := mha.NumHeads // Assuming NumHeads is stored in MHA struct
		seqLength := mha.q.Grad.Shape[2]
		depth := mha.Depth // Assuming Depth is stored in MHA struct
		dimModel := mha.DimModel // Assuming DimModel is stored in MHA struct

		// Implement the summation over heads for mha.q.Grad, mha.k.Grad, mha.v.Grad
		// to get gradQCombined, gradKCombined, gradVCombined [b, s, dim_model].
		// See previous explanation on how to manually sum over heads.

		// Sum mha.q.Grad over heads
		combinedShape := []int{batchSize, seqLength, dimModel}
		gradQCombinedData := make([]float64, batchSize * seqLength * dimModel)
		gradQCombined := NewTensor(gradQCombinedData, combinedShape, false)

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
		gradKCombinedData := make([]float64, batchSize * seqLength * dimModel)
		gradKCombined := NewTensor(gradKCombinedData, combinedShape, false)

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
		gradVCombinedData := make([]float64, batchSize * seqLength * dimModel)
		gradVCombined := NewTensor(gradVCombinedData, combinedShape, false)

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
			mha.inputTensor.Grad = NewTensor(make([]float64, len(mha.inputTensor.Data)), mha.inputTensor.Shape, false)
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
	}


	// --- 7. Backpropagate through the initial linear projections (Wq, Wk, Wv) ---
	// Input: mha.inputTensor [b, s, dim_model]
	// Output gradients: gradQCombined, gradKCombined, gradVCombined [b, s, dim_model] (calculated in step 6)
	// Calculate gradients w.r.t. weights Wq, Wk, Wv.
	// dLoss/dWq = inputTensor^T @ gradQCombined
	// dLoss/dWk = inputTensor^T @ gradKCombined
	// dLoss/dWv = inputTensor^T @ gradVCombined
	// Accumulate these gradients to mha.Wq.Grad, mha.Wk.Grad, mha.Wv.Grad.

	if mha.inputTensor != nil && gradQCombined != nil && gradKCombined != nil && gradVCombined != nil {
		// Transpose input tensor for matrix multiplication with combined gradients
		inputTransposedForWeights, err := mha.inputTensor.Transpose(len(mha.inputTensor.Shape)-2, len(mha.inputTensor.Shape)-1) // Transpose last two dimensions
		if err != nil {
			panic(fmt.Sprintf("failed to transpose inputTensor for MHA backward weights: %v", err))
		}

		// dLoss/dWq = inputTensor^T @ gradQCombined
		if mha.Wq != nil && mha.Wq.requiresGrad {
			if mha.Wq.Grad == nil {
				mha.Wq.Grad = NewTensor(make([]float64, len(mha.Wq.Data)), mha.Wq.Shape, false)
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
			tempGradWq := NewTensor(tempGradWqData, tempGradWqShape, false)

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
						for k := 0; k < K; k++ { // inner dimension
							// Access elements in inputSlice [d_m, s] and gradQSlice [s, d_m]
							// inputSlice index: i*K + k
							// gradQSlice index: k*N + j
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
				for i := 0; i < dimModel * dimModel; i++ {
					finalGradWqData[i] += tempGradWq.Data[batchStart+i]
				}
			}
			finalGradWq := NewTensor(finalGradWqData, []int{dimModel, dimModel}, false)


			// Accumulate finalGradWq to mha.Wq.Grad
			for i := range mha.Wq.Grad.Data {
				mha.Wq.Grad.Data[i] += finalGradWq.Data[i]
			}
		}

		// dLoss/dWk = inputTensor^T @ gradKCombined
		if mha.Wk != nil && mha.Wk.requiresGrad {
			if mha.Wk.Grad == nil {
				mha.Wk.Grad = NewTensor(make([]float64, len(mha.Wk.Data)), mha.Wk.Shape, false)
			}
			// Implement calculation and accumulation for gradWk similar to gradWq
			// MatMul: inputTransposedForWeights [b, d_m, s] @ gradKCombined [b, s, d_m] -> [b, d_m, d_m]
			// Sum over batch dimension.
		}

		// dLoss/dWv = inputTensor^T @ gradVCombined
		if mha.Wv != nil && mha.Wv.requiresGrad {
			if mha.Wv.Grad == nil {
				mha.Wv.Grad = NewTensor(make([]float64, len(mha.Wv.Data)), mha.Wv.Shape, false)
			}
			// Implement calculation and accumulation for gradWv similar to gradWq
			// MatMul: inputTransposedForWeights [b, d_m, s] @ gradVCombined [b, s, d_m] -> [b, d_m, d_m]
			// Sum over batch dimension.
		}

		// dLoss/dinputTensor is already accumulated in step 6.
	}

}


// Forward performs the forward pass of the MultiHeadAttention layer.
// This is a simplified version without caching or masks.
func (mha *MultiHeadAttention) Forward(query, key, value *Tensor, mask *Tensor) (*Tensor, error) {
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
	mha.q = qReshaped // Store q after reshape/split

	qTransposed, err := qReshaped.Transpose(1, 2) // Transpose to [batch_size, num_heads, q_seq_length, head_dim]
	if err != nil {
		return nil, fmt.Errorf("failed to transpose query tensor: %w", err)
	}
	kReshaped, err := k.Reshape([]int{batchSize, kvSeqLength, mha.NumHeads, mha.HeadDim})
	if err != nil {
		return nil, fmt.Errorf("failed to reshape key tensor: %w", err)
	}
	mha.k = kReshaped // Store k after reshape/split

	kTransposed, err := kReshaped.Transpose(1, 2) // Transpose to [batch_size, num_heads, kv_seq_length, head_dim]
	if err != nil {
		return nil, fmt.Errorf("failed to transpose key tensor: %w", err)
	}
	vReshaped, err := v.Reshape([]int{batchSize, kvSeqLength, mha.NumHeads, mha.HeadDim})
	if err != nil {
		return nil, fmt.Errorf("failed to reshape value tensor: %w", err)
	}
	mha.v = vReshaped // Store v after reshape/split

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
	mha.attentionScores = attentionScores // Store attention scores


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

	// Apply output linear layer
	output, err := mha.OutputLinear.Forward(contextLayerReshaped)
	if err != nil {
		return nil, fmt.Errorf("multihead attention output linear failed: %w", err)
	}

	// Set creator and requiresGrad for the output tensor
	outputRequiresGrad := query.requiresGrad || key.requiresGrad || value.requiresGrad ||
		mha.Wq.requiresGrad || mha.Wk.requiresGrad || mha.Wv.requiresGrad || mha.Wo.requiresGrad

	output.requiresGrad = outputRequiresGrad
	if output.requiresGrad {
		output.creator = mha // Set the creator to the MultiHeadAttention layer itself
	}

	return output, nil
}

// FeedForward represents a feed-forward network.
type FeedForward struct {
	Linear1 *Linear
	Linear2 *Linear
	// Stored intermediate tensors for backward pass
	activatedHidden *Tensor // Output of the activation function
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

	// Store input for potential future use in backward (if needed for skip connections etc.)
	// For this basic FF, we primarily need the activatedHidden output for backprop.

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
	activatedHidden := NewTensor(activatedHiddenData, hidden.Shape, false)

	// Store the output of the activation function
	ff.activatedHidden = activatedHidden

	// Apply the second linear transformation
	output, err := ff.Linear2.Forward(activatedHidden)
	if err != nil {
		return nil, fmt.Errorf("feed-forward second linear failed: %w", err)
	}

	// Set creator and requiresGrad for the output tensor
	// The output requires gradients if the input or any of the linear layers' weights/biases require gradients
	output.requiresGrad = input.requiresGrad || ff.Linear1.Weights.requiresGrad || (ff.Linear1.Biases != nil && ff.Linear1.Biases.requiresGrad) ||
		ff.Linear2.Weights.requiresGrad || (ff.Linear2.Biases != nil && ff.Linear2.Biases.requiresGrad)

	if output.requiresGrad {
		// Set the creator of the output tensor to the FeedForward layer itself.
		// The FeedForward layer's Backward method will handle the backpropagation
		// through its components.
		output.creator = ff
	}

	return output, nil
}
func (ff *FeedForward) Backward(grad *Tensor) {
	if grad == nil || grad.Data == nil {
		return // No gradient to propagate
	}

	// Ensure input tensor is stored and requires grad if needed
	if ff.inputTensor == nil {
		panic("FeedForward backward called before forward (inputTensor is nil)")
	}
	if ff.inputTensor.requiresGrad {
		if ff.inputTensor.Grad == nil {
			ff.inputTensor.Grad = NewTensor(...) // Initialize gradient
		}
	}

	// Backpropagate through the second linear layer
	// Assuming the second linear layer is a field in the struct (e.g., ff.Layer2)
	// And assuming Layer2.Backward takes the incoming gradient and returns the gradient wrt its input
	// (or directly adds to the input's Grad field, which is more common with the creator pattern)
	// Let's assume the creator pattern where calling Backward on an operation
	// computes and adds gradients to its inputs and parameters.
	if ff.Layer2 == nil {
		panic("FeedForward Layer2 is nil in backward")
	}
	ff.Layer2.Backward(grad) // Call Backward on the second linear layer


	// Backpropagate through the activation function
	// Assuming the activation function is represented by an Operation (e.g., ff.ActivationOp)
	// And assuming activationOutput is stored
	if ff.ActivationOp == nil || ff.activationOutput == nil {
		panic("FeedForward activation op or output is nil in backward")
	}
	// The gradient for the activation function is the gradient of its output,
	// which is the gradient of the second linear layer's input.
	// This is the gradient accumulated in ff.activationOutput.Grad by Layer2.Backward.
	if ff.activationOutput.Grad != nil {
		ff.ActivationOp.Backward(ff.activationOutput.Grad) // Call Backward on the activation function
	}


	// Backpropagate through the first linear layer
	// Assuming the first linear layer is a field in the struct (e.g., ff.Layer1)
	if ff.Layer1 == nil {
		panic("FeedForward Layer1 is nil in backward")
	}
	if ff.Layer1.OutputTensor != nil && ff.Layer1.OutputTensor.Grad != nil { // Assuming OutputTensor field in Linear and it's populated
		ff.Layer1.Backward(ff.Layer1.OutputTensor.Grad)
	} else {
		// If Linear's Backward propagates directly to input.Grad, and ActivationOp.Backward
		// propagates to its input's Grad (which is Layer1's output), then Layer1.Backward
		// will be implicitly called when ActivationOp.Backward propagates to its input.
		// So, we might not need an explicit call to ff.Layer1.Backward here if the
		// backward propagation is handled by the tensor's Backward method traversing
		// the creator chain.
		// Let's rely on the Tensor.Backward traversal.
	}
}

// MultiHeadCrossAttention represents a multi-head cross-attention layer.
type MultiHeadCrossAttention struct {
	NumQHeads int // Number of query heads
	NumKVHeads int // Number of key/value heads
	DimModel int
	DimKVHeads int // Dimension per key/value head

	// Stored intermediate tensors for backward pass
	queryTensor *Tensor // Original query input from decoder
	keyTensor *Tensor // Original key input from encoder
	valueTensor *Tensor // Original value input from encoder
	q, k, v *Tensor // Q, K, V after linear projection and splitting heads
	attentionScores *Tensor // Q @ K^T
	attentionWeights *Tensor // Softmax(attentionScores) + Mask
	attentionOutputBeforeConcat *Tensor // attentionWeights @ V (before concatenating heads)
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

// Backward performs the backward pass for multi-head cross-attention.
// grad is the gradient from the output of the attention layer (after the final linear layer).
func (mha *MultiHeadCrossAttention) Backward(grad *Tensor) {
	if grad == nil || grad.Data == nil {
		return // No gradient to propagate
	}

	// Add checks for nil intermediate tensors and weights
	// Initialize gradients for mha.Wq.Grad, mha.Wk.Grad, mha.Wv.Grad, mha.Wo.Grad,
	// and mha.queryTensor.Grad, mha.keyTensor.Grad, mha.valueTensor.Grad
	// if requiresGrad is true and Grad is nil.


	// --- 1. Backpropagate through the final linear layer (Wo) ---
	// The incoming gradient 'grad' is w.r.t. the output of Wo.
	// Call Backward on Wo with 'grad'. This calculates gradients for Wo's parameters
	// and the gradient w.r.t. Wo's input (mha.attentionOutput), accumulating it in mha.attentionOutput.Grad.
	if mha.Wo != nil { // Assuming Wo is a Linear layer
		mha.Wo.Backward(grad)
	}
	// After this call, the gradient w.r.t. mha.attentionOutput is in mha.attentionOutput.Grad


	// --- 2. Get the gradient before concatenation/reshape ---
	// The gradient w.r.t. mha.attentionOutput is in mha.attentionOutput.Grad.
	// Reshape this gradient from [batch_size, query_seq_len, dim_model] to [batch_size, num_heads, query_seq_len, depth].
	// This is the gradient w.r.t. the output of MatMul(attentionWeights @ V) for each head before concatenation.
	// Implement reshaping of mha.attentionOutput.Grad here. Shape: [b, h, query_s, d]

	if mha.attentionOutput == nil || mha.attentionOutput.Grad == nil {
		panic("mha.attentionOutput or its gradient is nil in reshape step")
	}

	batchSize := mha.attentionOutput.Grad.Shape[0]
	querySeqLen := mha.attentionOutput.Grad.Shape[1]
	dimModel := mha.attentionOutput.Grad.Shape[2]
	numHeads := mha.NumHeads // Assuming NumHeads is stored in MHA struct
	depth := mha.Depth       // Assuming Depth is stored in MHA struct (dimModel / numHeads)

	if dimModel != numHeads*depth {
		panic(fmt.Sprintf("dimModel (%d) does not match numHeads (%d) * depth (%d) in reshape step", dimModel, numHeads, depth))
	}

	// The reshaped gradient will have shape [batch_size, num_heads, query_seq_len, depth]
	reshapedShape := []int{batchSize, numHeads, querySeqLen, depth}
	gradBeforeConcatData := make([]float64, len(mha.attentionOutput.Grad.Data)) // Same size data
	gradBeforeConcat := NewTensor(gradBeforeConcatData, reshapedShape, false) // Gradient tensor does not require gradients

	// Manually reshape the gradient by mapping flattened indices.
	// Original flat index for [b, query_s, d_model_idx]: b * query_seq_len * dimModel + query_s * dimModel + d_model_idx
	// Reshaped flat index for [b, h, query_s, d]: b * num_heads * query_seq_len * depth + h * query_seq_len * depth + query_s * depth + d

	originalFlatIndex := 0
	for b := 0; b < batchSize; b++ {
		for s := 0; s < querySeqLen; s++ {
			for d_model_idx := 0; d_model_idx < dimModel; d_model_idx++ {
				// Calculate original flat index
				originalFlatIndex = b*querySeqLen*dimModel + s*dimModel + d_model_idx

				// Calculate h and d from d_model_idx
				h := d_model_idx / depth
				d := d_model_idx % depth

				// Calculate reshaped flat index
				reshapedFlatIndex := b*numHeads*querySeqLen*depth + h*querySeqLen*depth + s*depth + d

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
	// Inputs: mha.attentionWeights [b, h, query_s, key_s], mha.v [b, h, key_s, d]
	// Output gradient: gradBeforeConcat [b, h, query_s, d]
	// Calculate gradients w.r.t. inputs:
	// dLoss/dattentionWeights [b, h, query_s, key_s] = gradBeforeConcat [b, h, query_s, d] @ mha.v^T [b, h, d, key_s]
	// dLoss/dv [b, h, key_s, d] = attentionWeights^T [b, h, key_s, query_s] @ gradBeforeConcat [b, h, query_s, d]
	// Accumulate these gradients to mha.attentionWeights.Grad and mha.v.Grad.

	if mha.attentionWeights != nil && mha.v != nil && gradBeforeConcat != nil {
		keySeqLen := mha.v.Shape[2] // Assuming v shape is [b, h, key_s, d]

		// dLoss/dattentionWeights = gradBeforeConcat @ mha.v^T
		// Transpose mha.v: swap last two dimensions (key_s and depth)
		vTransposed, err := mha.v.Transpose(len(mha.v.Shape)-2, len(mha.v.Shape)-1)
		if err != nil {
			panic(fmt.Sprintf("failed to transpose v for MHCA backward: %v", err))
		}

		// Need a batched MatMul operation: [b, h, query_s, d] @ [b, h, d, key_s] -> [b, h, query_s, key_s]
		gradAttentionWeights, err := gradBeforeConcat.MatMul(vTransposed) // Assuming MatMul handles batches
		if err != nil {
			panic(fmt.Sprintf("failed to calculate gradAttentionWeights in MHCA backward: %v", err))
		}

		// Add to existing gradient for attention weights
		if mha.attentionWeights.requiresGrad {
			if mha.attentionWeights.Grad == nil {
				mha.attentionWeights.Grad = NewTensor(make([]float64, len(mha.attentionWeights.Data)), mha.attentionWeights.Shape, false)
			}
			// Accumulate gradAttentionWeights to mha.attentionWeights.Grad element-wise
			for i := range mha.attentionWeights.Grad.Data {
				mha.attentionWeights.Grad.Data[i] += gradAttentionWeights.Data[i]
			}
		}


		// dLoss/dv = attentionWeights^T @ gradBeforeConcat
		// Transpose mha.attentionWeights: swap last two dimensions (query_s and key_s)
		attentionWeightsTransposed, err := mha.attentionWeights.Transpose(len(mha.attentionWeights.Shape)-2, len(mha.attentionWeights.Shape)-1)
		if err != nil {
			panic(fmt.Sprintf("failed to transpose attentionWeights for MHCA backward: %v", err))
		}

		// Need a batched MatMul operation: [b, h, key_s, query_s] @ [b, h, query_s, d] -> [b, h, key_s, d]
		gradV, err := attentionWeightsTransposed.MatMul(gradBeforeConcat) // Assuming MatMul handles batches
		if err != nil {
			panic(fmt.Sprintf("failed to calculate gradV in MHCA backward: %v", err))
		}

		// Add to existing gradient for v
		if mha.v != nil && mha.v.requiresGrad { // Check if v requires grad (it's an intermediate tensor from input projection)
			if mha.v.Grad == nil {
				mha.v.Grad = NewTensor(make([]float64, len(mha.v.Data)), mha.v.Shape, false)
			}
			// Accumulate gradV to mha.v.Grad element-wise
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
		mha.attentionWeights.Backward(mha.attentionWeights.Grad)
	}
	// After this, the gradient w.r.t. mha.attentionScores is accumulated in mha.attentionScores.Grad.


	// --- 5. Backpropagate through MatMul(Q @ K^T) ---
	// Inputs: mha.q [b, h, query_s, d], mha.k [b, h, key_s, d] (K^T in forward)
	// Output: mha.attentionScores [b, h, query_s, key_s]
	// Output gradient: mha.attentionScores.Grad (accumulated in step 4)
	// Calculate gradients w.r.t. inputs:
	// dLoss/dq [b, h, query_s, d] = mha.attentionScores.Grad [b, h, query_s, key_s] @ mha.k [b, h, key_s, d] (untransposed K)
	// dLoss/dk [b, h, key_s, d] = mha.q^T [b, h, d, query_s] @ mha.attentionScores.Grad [b, h, query_s, key_s]
	// Accumulate these gradients to mha.q.Grad and mha.k.Grad.

	if mha.q != nil && mha.k != nil && mha.attentionScores != nil && mha.attentionScores.Grad != nil {
		querySeqLen := mha.q.Shape[2] // Assuming q shape is [b, h, query_s, d]
		keySeqLen := mha.k.Shape[2]   // Assuming k shape is [b, h, key_s, d]
		depth := mha.Depth            // Assuming Depth is stored

		// dLoss/dq = dLoss/dAttentionScores @ k (untransposed)
		// Need a batched MatMul operation: [b, h, query_s, key_s] @ [b, h, key_s, d] -> [b, h, query_s, d]
		gradQ_per_head, err := mha.attentionScores.Grad.MatMul(mha.k) // Assuming MatMul handles batches
		if err != nil {
			panic(fmt.Sprintf("failed to calculate gradQ in MHCA backward: %v", err))
		}
		// Accumulate gradQ_per_head to mha.q.Grad
		if mha.q.requiresGrad { // Check if q requires grad
			if mha.q.Grad == nil {
				mha.q.Grad = NewTensor(make([]float64, len(mha.q.Data)), mha.q.Shape, false)
			}
			for i := range mha.q.Grad.Data {
				mha.q.Grad.Data[i] += gradQ_per_head.Data[i]
			}
		}


		// dLoss/dk = q^T @ dLoss/dAttentionScores
		// Transpose mha.q: swap last two dimensions (query_s and depth)
		qTransposedForGradK, err := mha.q.Transpose(len(mha.q.Shape)-2, len(mha.q.Shape)-1)
		if err != nil {
			panic(fmt.Sprintf("failed to transpose q for MHCA backward: %v", err))
		}

		// Need a batched MatMul operation: [b, h, d, query_s] @ [b, h, query_s, key_s] -> [b, h, key_s, d]
		// The result shape should be [b, h, key_s, d] for gradK.
		// MatMul(A^T @ C) where A=q, C=attentionScores.Grad
		// Shape: [b, h, d, query_s] @ [b, h, query_s, key_s] -> [b, h, d, key_s]
		// We need [b, h, key_s, d]. The formula is dLoss/dk = (Q^T @ dLoss/dAttentionScores)^T = dLoss/dAttentionScores^T @ Q.
		// Let's use the correct formula: dLoss/dk = dLoss/dAttentionScores^T @ q (untransposed)
		attentionScoresGradTransposed, err := mha.attentionScores.Grad.Transpose(len(mha.attentionScores.Grad.Shape)-2, len(mha.attentionScores.Grad.Shape)-1)
		if err != nil {
			panic(fmt.Sprintf("failed to transpose attentionScores.Grad for MHCA backward: %v", err))
		}

		// Need a batched MatMul operation: [b, h, key_s, query_s] @ [b, h, query_s, d] -> [b, h, key_s, d]
		gradK_per_head, err := attentionScoresGradTransposed.MatMul(mha.q) // Batched MatMul
		if err != nil {
			panic(fmt.Sprintf("failed to calculate gradK in MHCA backward: %v", err))
		}
		// Accumulate gradK_per_head to mha.k.Grad
		if mha.k != nil && mha.k.requiresGrad { // Check if k requires grad
			if mha.k.Grad == nil {
				mha.k.Grad = NewTensor(make([]float64, len(mha.k.Data)), mha.k.Shape, false)
			}
			for i := range mha.k.Grad.Data {
				mha.k.Grad.Data[i] += gradK_per_head.Data[i]
			}
		}
	}
	// After this, the gradients w.r.t. q and k are accumulated in mha.q.Grad and mha.k.Grad.


	// --- 6. Combine gradients from heads ---
	// Gradients w.r.t. q, k, v are in mha.q.Grad [b, h, query_s, d], mha.k.Grad [b, h, key_s, d], mha.v.Grad [b, h, key_s, d].
	// Sum gradQ over heads to get gradQCombined [b, query_s, dim_model]
	// Sum gradK over heads to get gradKCombined [b, key_s, dim_model]
	// Sum gradV over heads to get gradVCombined [b, key_s, dim_model]
	// Accumulate these combined gradients to the input tensors' gradients (queryTensor.Grad, keyTensor.Grad, valueTensor.Grad).

	if mha.queryTensor != nil && mha.queryTensor.requiresGrad && mha.q != nil && mha.q.Grad != nil {
		// gradQ (per head) [b, h, query_s, d] needs to be combined over heads to [b, query_s, dim_model]
		// Implement summation over heads for mha.q.Grad to get gradQCombined [b, query_s, dim_model]
		batchSize := mha.q.Grad.Shape[0]
		numHeads := mha.NumHeads
		querySeqLen := mha.q.Grad.Shape[2]
		depth := mha.Depth
		dimModel := mha.DimModel

		combinedShapeQ := []int{batchSize, querySeqLen, dimModel}
		gradQCombinedData := make([]float64, batchSize * querySeqLen * dimModel)
		gradQCombined := NewTensor(gradQCombinedData, combinedShapeQ, false)

		qGradFlatIndex := 0
		for b := 0; b < batchSize; b++ {
			for h := 0; h < numHeads; h++ {
				for s := 0; s < querySeqLen; s++ {
					for d := 0; d < depth; d++ {
						qGradFlatIndex = b*numHeads*querySeqLen*depth + h*querySeqLen*depth + s*depth + d
						combinedFlatIndex := b*querySeqLen*dimModel + s*dimModel + (h*depth + d)
						if qGradFlatIndex >= len(mha.q.Grad.Data) || combinedFlatIndex >= len(gradQCombined.Data) {
							panic("index out of bounds during gradient summation over heads for Q")
						}
						gradQCombined.Data[combinedFlatIndex] += mha.q.Grad.Data[qGradFlatIndex]
					}
				}
			}
		}


		// Accumulate gradQCombined to queryTensor.Grad
		if mha.queryTensor.Grad == nil {
			mha.queryTensor.Grad = NewTensor(make([]float64, len(mha.queryTensor.Data)), mha.queryTensor.Shape, false)
		}
		for i := range mha.queryTensor.Grad.Data {
			mha.queryTensor.Grad.Data[i] += gradQCombined.Data[i]
		}
	}


	if mha.keyTensor != nil && mha.keyTensor.requiresGrad && mha.k != nil && mha.k.Grad != nil {
		// gradK (per head) [b, h, key_s, d] needs to be combined over heads to [b, key_s, dim_model]
		// Implement summation over heads for mha.k.Grad to get gradKCombined [b, key_s, dim_model]
		batchSize := mha.k.Grad.Shape[0]
		numHeads := mha.NumHeads
		keySeqLen := mha.k.Grad.Shape[2]
		depth := mha.Depth
		dimModel := mha.DimModel

		combinedShapeK := []int{batchSize, keySeqLen, dimModel}
		gradKCombinedData := make([]float64, batchSize * keySeqLen * dimModel)
		gradKCombined := NewTensor(gradKCombinedData, combinedShapeK, false)

		kGradFlatIndex := 0
		for b := 0; b < batchSize; b++ {
			for h := 0; h < numHeads; h++ {
				for s := 0; s < keySeqLen; s++ {
					for d := 0; d < depth; d++ {
						kGradFlatIndex = b*numHeads*keySeqLen*depth + h*keySeqLen*depth + s*depth + d
						combinedFlatIndex := b*keySeqLen*dimModel + s*dimModel + (h*depth + d)
						if kGradFlatIndex >= len(mha.k.Grad.Data) || combinedFlatIndex >= len(gradKCombined.Data) {
							panic("index out of bounds during gradient summation over heads for K")
						}
						gradKCombined.Data[combinedFlatIndex] += mha.k.Grad.Data[kGradFlatIndex]
					}
				}
			}
		}

		// Accumulate gradKCombined to keyTensor.Grad
		if mha.keyTensor.Grad == nil {
			mha.keyTensor.Grad = NewTensor(make([]float64, len(mha.keyTensor.Data)), mha.keyTensor.Shape, false)
		}
		for i := range mha.keyTensor.Grad.Data {
			mha.keyTensor.Grad.Data[i] += gradKCombined.Data[i]
		}
	}

	if mha.valueTensor != nil && mha.valueTensor.requiresGrad && mha.v != nil && mha.v.Grad != nil {
		// gradV (per head) [b, h, key_s, d] needs to be combined over heads to [b, key_s, dim_model]
		// Implement summation over heads for mha.v.Grad to get gradVCombined [b, key_s, dim_model]
		batchSize := mha.v.Grad.Shape[0]
		numHeads := mha.NumHeads
		keySeqLen := mha.v.Grad.Shape[2]
		depth := mha.Depth
		dimModel := mha.DimModel

		combinedShapeV := []int{batchSize, keySeqLen, dimModel}
		gradVCombinedData := make([]float64, batchSize * keySeqLen * dimModel)
		gradVCombined := NewTensor(gradVCombinedData, combinedShapeV, false)

		vGradFlatIndex := 0
		for b := 0; b < batchSize; b++ {
			for h := 0; h < numHeads; h++ {
				for s := 0; s < keySeqLen; s++ {
					for d := 0; d < depth; d++ {
						vGradFlatIndex = b*numHeads*keySeqLen*depth + h*keySeqLen*depth + s*depth + d
						combinedFlatIndex := b*keySeqLen*dimModel + s*dimModel + (h*depth + d)
						if vGradFlatIndex >= len(mha.v.Grad.Data) || combinedFlatIndex >= len(gradVCombined.Data) {
							panic("index out of bounds during gradient summation over heads for V")
						}
						gradVCombined.Data[combinedFlatIndex] += mha.v.Grad.Data[vGradFlatIndex]
					}
				}
			}
		}

		// Accumulate gradVCombined to valueTensor.Grad
		if mha.valueTensor.Grad == nil {
			mha.valueTensor.Grad = NewTensor(make([]float64, len(mha.valueTensor.Data)), mha.valueTensor.Shape, false)
		}
		for i := range mha.valueTensor.Grad.Data {
			mha.valueTensor.Grad.Data[i] += gradVCombined.Data[i]
		}
	}


	// --- 7. Backpropagate through the initial linear projections (Wq, Wk, Wv) ---
	// Backpropagate gradQCombined to Wq
	// dLoss/dWq = queryTensor^T @ gradQCombined
	if mha.queryTensor != nil && mha.Wq != nil && mha.Wq.requiresGrad && gradQCombined != nil {
		if mha.Wq.Grad == nil {
			mha.Wq.Grad = NewTensor(make([]float64, len(mha.Wq.Data)), mha.Wq.Shape, false)
		}
		// Transpose queryTensor: swap last two dimensions
		queryTransposedForWq, err := mha.queryTensor.Transpose(len(mha.queryTensor.Shape)-2, len(mha.queryTensor.Shape)-1)
		if err != nil {
			panic(fmt.Sprintf("failed to transpose queryTensor for MHCA Wq backward: %v", err))
		}
		// MatMul: queryTransposedForWq [b, dim_model, query_s] @ gradQCombined [b, query_s, dim_model] -> [b, dim_model, dim_model]
		// Sum over batch dimension to get [dim_model, dim_model]
		batchSize := mha.queryTensor.Shape[0]
		dimModel := mha.DimModel
		querySeqLen := mha.queryTensor.Shape[1]

		tempGradWqShape := []int{batchSize, dimModel, dimModel}
		tempGradWqData := make([]float64, batchSize*dimModel*dimModel)
		tempGradWq := NewTensor(tempGradWqData, tempGradWqShape, false)

		// Perform batched matrix multiplication: queryTransposedForWq @ gradQCombined
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
		// Sum tempGradWq over batch dimension
		finalGradWqData := make([]float64, dimModel*dimModel)
		for b := 0; b < batchSize; b++ {
			batchStart := b * dimModel * dimModel
			for i := 0; i < dimModel * dimModel; i++ {
				finalGradWqData[i] += tempGradWq.Data[batchStart+i]
			}
		}
		finalGradWq := NewTensor(finalGradWqData, []int{dimModel, dimModel}, false)

		// Accumulate finalGradWq to mha.Wq.Grad
		for i := range mha.Wq.Grad.Data {
			mha.Wq.Grad.Data[i] += finalGradWq.Data[i]
		}
	}

	// Backpropagate gradKCombined to Wk
	// dLoss/dWk = keyTensor^T @ gradKCombined
	if mha.keyTensor != nil && mha.Wk != nil && mha.Wk.requiresGrad && gradKCombined != nil {
		if mha.Wk.Grad == nil {
			mha.Wk.Grad = NewTensor(make([]float64, len(mha.Wk.Data)), mha.Wk.Shape, false)
		}
		// Transpose keyTensor: swap last two dimensions
		keyTransposedForWk, err := mha.keyTensor.Transpose(len(mha.keyTensor.Shape)-2, len(mha.keyTensor.Shape)-1)
		if err != nil {
			panic(fmt.Sprintf("failed to transpose keyTensor for MHCA Wk backward: %v", err))
		}
		// MatMul: keyTransposedForWk [b, dim_model, key_s] @ gradKCombined [b, key_s, dim_model] -> [b, dim_model, dim_model]
		// Sum over batch dimension.
		batchSize := mha.keyTensor.Shape[0]
		dimModel := mha.DimModel
		keySeqLen := mha.keyTensor.Shape[1]

		tempGradWkShape := []int{batchSize, dimModel, dimModel}
		tempGradWkData := make([]float64, batchSize*dimModel*dimModel)
		tempGradWk := NewTensor(tempGradWkData, tempGradWkShape, false)

		// Perform batched matrix multiplication: keyTransposedForWk @ gradKCombined
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
		// Sum tempGradWk over batch dimension
		finalGradWkData := make([]float64, dimModel*dimModel)
		for b := 0; b < batchSize; b++ {
			batchStart := b * dimModel * dimModel
			for i := 0; i < dimModel * dimModel; i++ {
				finalGradWkData[i] += tempGradWk.Data[batchStart+i]
			}
		}
		finalGradWk := NewTensor(finalGradWkData, []int{dimModel, dimModel}, false)

		// Accumulate finalGradWk to mha.Wk.Grad
		for i := range mha.Wk.Grad.Data {
			mha.Wk.Grad.Data[i] += finalGradWk.Data[i]
		}
	}

	// Backpropagate gradVCombined to Wv
	// dLoss/dWv = valueTensor^T @ gradVCombined
	if mha.valueTensor != nil && mha.Wv != nil && mha.Wv.requiresGrad && gradVCombined != nil {
		if mha.Wv.Grad == nil {
			mha.Wv.Grad = NewTensor(make([]float64, len(mha.Wv.Data)), mha.Wv.Shape, false)
		}
		// Transpose valueTensor: swap last two dimensions
		valueTransposedForWv, err := mha.valueTensor.Transpose(len(mha.valueTensor.Shape)-2, len(mha.valueTensor.Shape)-1)
		if err != nil {
			panic(fmt.Sprintf("failed to transpose valueTensor for MHCA Wv backward: %v", err))
		}
		// MatMul: valueTransposedForWv [b, dim_model, key_s] @ gradVCombined [b, key_s, dim_model] -> [b, dim_model, dim_model]
		// Sum over batch dimension.
		batchSize := mha.valueTensor.Shape[0]
		dimModel := mha.DimModel
		keySeqLen := mha.valueTensor.Shape[1]

		tempGradWvShape := []int{batchSize, dimModel, dimModel}
		tempGradWvData := make([]float64, batchSize*dimModel*dimModel)
		tempGradWv := NewTensor(tempGradWvData, tempGradWvShape, false)

		// Perform batched matrix multiplication: valueTransposedForWv @ gradVCombined
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
		// Sum tempGradWv over batch dimension
		finalGradWvData := make([]float64, dimModel*dimModel)
		for b := 0; b < batchSize; b++ {
			batchStart := b * dimModel * dimModel
			for i := 0; i < dimModel * dimModel; i++ {
				finalGradWvData[i] += tempGradWv.Data[batchStart+i]
			}
		}
		finalGradWv := NewTensor(finalGradWvData, []int{dimModel, dimModel}, false)

		// Accumulate finalGradWv to mha.Wv.Grad






func (mha *MultiHeadAttention) Backward(grad *Tensor) {
    	if grad == nil || grad.Data == nil {
    		return // No gradient to propagate
    	}
		// Backpropagate through the final linear layer (Wo)
    	if mha.Wo != nil { // Assuming Wo is a Linear layer
    		mha.Wo.Backward(grad)
    	}
    	// After this call, the gradient w.r.t. mha.attentionOutput is in mha.attentionOutput.Grad
// Backpropagate through MatMul(attentionWeights @ V)
    	if mha.attentionWeights != nil && mha.v != nil && gradBeforeConcat != nil {
    		// dLoss/dattentionWeights = gradBeforeConcat @ v^T
    		vTransposed, err := mha.v.Transpose(...) // Transpose V with appropriate axes
    		if err != nil { fmt.Println(err) }
    		gradAttentionWeights, err := gradBeforeConcat.MatMul(vTransposed) // Batched MatMul
    		if err != nil { fmt.Println(err) }
    		// Accumulate gradAttentionWeights to mha.attentionWeights.Grad

    		// dLoss/dv = attentionWeights^T @ gradBeforeConcat
    		attentionWeightsTransposed, err := mha.attentionWeights.Transpose(...) // Transpose Attention Weights with appropriate axes
    		if err != nil { fmt.Println(err) }
    		gradV, err := attentionWeightsTransposed.MatMul(gradBeforeConcat) // Batched MatMul
    		if err != nil { fmt.Println(err) }
    		// Accumulate gradV to mha.v.Grad
    	}

		// Backpropagate through Softmax (and Masking)
    	if mha.attentionWeights != nil && mha.attentionWeights.Grad != nil {
    		// The Softmax operation's Backward method needs the output of the Softmax (attentionWeights)
    		// and the gradient w.r.t. the output (mha.attentionWeights.Grad) to calculate
    		// the gradient w.r.t. the input (attentionScores).
    		// If Softmax is an operation and mha.attentionWeights is its output tensor:
    		// mha.attentionWeights.Backward(mha.attentionWeights.Grad) // This triggers Softmax.Backward
    	}
    	// After this, the gradient w.r.t. mha.attentionScores is in mha.attentionScores.Grad.
// Backpropagate through MatMul(Q @ K^T)
    	if mha.q != nil && mha.k != nil && mha.attentionScores != nil && mha.attentionScores.Grad != nil {
    		// dLoss/dq (per head) = mha.attentionScores.Grad @ k.Transpose(...)
    		kTransposedForGradQ, err := mha.k.Transpose(...) // Transpose K with appropriate axes
    		if err != nil { fmt.Println(err) }
    		gradQ_per_head, err := mha.attentionScores.Grad.MatMul(kTransposedForGradQ) // Batched MatMul
    		if err != nil { fmt.Println(err) }
    		// Accumulate gradQ_per_head to mha.q.Grad

    		// dLoss/dk (per head) = q.Transpose(...) @ mha.attentionScores.Grad
    		qTransposedForGradK, err := mha.q.Transpose(...) // Transpose Q with appropriate axes
    		if err != nil { fmt.Println(err) }
    		gradK_per_head, err := qTransposedForGradK.MatMul(mha.attentionScores.Grad) // Batched MatMul
    		if err != nil { fmt.Println(err) }
    		// Accumulate gradK_per_head to mha.k.Grad
    	}

		// Combine gradients from heads
    	// Sum mha.q.Grad over the num_heads dimension to get gradQCombined [batch_size, seq_len, dim_model]
    	// Sum mha.k.Grad over the num_heads dimension to get gradKCombined [batch_size, seq_len, dim_model]
    	// Sum mha.v.Grad over the num_heads dimension to get gradVCombined [batch_size, seq_len, dim_model]
    	gradQCombined := // Sum mha.q.Grad over heads
    	gradKCombined := // Sum mha.k.Grad over heads
    	gradVCombined := // Sum mha.v.Grad over heads


    	// Backpropagate through Linear Projections (Wq, Wk, Wv)
    	if mha.inputTensor != nil && gradQCombined != nil && gradKCombined != nil && gradVCombined != nil {
    		inputTransposedForWeights, err := mha.inputTensor.Transpose(...) // Transpose input tensor
    		if err != nil { fmt.Println(err) }

    		// dLoss/dWq = inputTensor^T @ gradQCombined
    		if mha.Wq != nil && mha.Wq.requiresGrad {
    			gradWq, err := inputTransposedForWeights.MatMul(gradQCombined)
    			if err != nil { fmt.Println(err) }
    			// Accumulate gradWq to mha.Wq.Grad
    		}

    		// dLoss/dWk = inputTensor^T @ gradKCombined
    		if mha.Wk != nil && mha.Wk.requiresGrad {
    			gradWk, err := inputTransposedForWeights.MatMul(gradKCombined)
    			if err != nil { fmt.Println(err) }
    			// Accumulate gradWk to mha.Wk.Grad
    		}

    		// dLoss/dWv = inputTensor^T @ gradVCombined
    		if mha.Wv != nil && mha.Wv.requiresGrad {
    			gradWv, err := inputTransposedForWeights.MatMul(gradVCombined)
    			if err != nil { fmt.Println(err) }
    			// Accumulate gradWv to mha.Wv.Grad
    		}

    		// dLoss/dinputTensor = gradQCombined @ Wq^T + gradKCombined @ Wk^T + gradVCombined @ Wv^T
    		// Accumulate these to mha.inputTensor.Grad

    		if mha.inputTensor.requiresGrad {
    			// Calculate gradient from Q path
    			wqTransposedForGradInput, err := mha.Wq.Transpose(...)
    			if err != nil { fmt.Println(err) }
    			gradInputQ, err := gradQCombined.MatMul(wqTransposedForGradInput)
    			if err != nil { fmt.Println(err) }
    			// Accumulate gradInputQ to mha.inputTensor.Grad

    			// Calculate gradient from K path
    			wkTransposedForGradInput, err := mha.Wk.Transpose(...)
    			if err != nil { fmt.Println(err) }
    			gradInputK, err := gradKCombined.MatMul(wkTransposedForGradInput)
    			if err != nil { fmt.Println(err) }
    			// Accumulate gradInputK to mha.inputTensor.Grad

    			// Calculate gradient from V path
    			wvTransposedForGradInput, err := mha.Wv.Transpose(...)
    			if err != nil { fmt.Println(err) }
    			gradInputV, err := gradVCombined.MatMul(wvTransposedForGradInput)
    			if err != nil { fmt.Println(err) }
    			// Accumulate gradInputV to mha.inputTensor.Grad
    		}
    	}
    }


batchSize := mha.q.Grad.Shape[0]
    numHeads := mha.q.Grad.Shape[1]
    seqLength := mha.q.Grad.Shape[2]
    depth := mha.q.Grad.Shape[3]
    dimModel := mha.DimModel // Assuming dimModel is stored in the MHA struct
// Shape for the combined gradient: [batch_size, seq_len, dim_model]
    	combinedShape := []int{batchSize, seqLength, dimModel}
    	gradQCombinedData := make([]float64, batchSize * seqLength * dimModel)
    	gradQCombined := NewTensor(gradQCombinedData, combinedShape, false) // Gradient tensor does not require gradients
// Iterate through the flattened data of mha.q.Grad and sum over the heads dimension
    
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


qGradFlatIndex := 0
    	for b := 0; b < batchSize; b++ {
    		for h := 0; h < numHeads; h++ {
    			for s := 0; s < seqLength; s++ {
    				for d := 0; d < depth; d++ {
    					// Calculate flattened index in mha.q.Grad
    					qGradFlatIndex = b*numHeads*seqLength*depth + h*seqLength*depth + s*depth + d

    					// Calculate corresponding flattened index in gradQCombined
    					// The depth dimension of each head maps to a part of the dim_model dimension.
    					// The index in the dim_model dimension is h * depth + d.
    					combinedFlatIndex := b*seqLength*dimModel + s*dimModel + (h*depth + d)

    					// Ensure indices are within bounds
    					if qGradFlatIndex >= len(mha.q.Grad.Data) || combinedFlatIndex >= len(gradQCombined.Data) {
    						panic("index out of bounds during gradient summation over heads")
    					}

    					// Add the gradient element to the combined gradient
    					gradQCombined.Data[combinedFlatIndex] += mha.q.Grad.Data[qGradFlatIndex]
    				}
    			}
    		}
    	}
		qGradFlatIndex := 0
		for b := 0; b < batchSize; b++ {
			for h := 0; h < numHeads; h++ {
				for s := 0; s < seqLength; s++ {
					for d := 0; d < depth; d++ {
						// Calculate flattened index in mha.q.Grad
						qGradFlatIndex = b*numHeads*seqLength*depth + h*seqLength*depth + s*depth + d
	
						// Calculate corresponding flattened index in gradQCombined
						// The depth dimension of each head maps to a part of the dim_model dimension.
						// The index in the dim_model dimension is h * depth + d.
						combinedFlatIndex := b*seqLength*dimModel + s*dimModel + (h*depth + d)
	
						// Ensure indices are within bounds
						if qGradFlatIndex >= len(mha.q.Grad.Data) || combinedFlatIndex >= len(gradQCombined.Data) {
							panic("index out of bounds during gradient summation over heads")
						}
	
						// Add the gradient element to the combined gradient
						gradQCombined.Data[combinedFlatIndex] += mha.q.Grad.Data[qGradFlatIndex]
combinedFlatIndex := b*seqLength*dimModel + s*dimModel + (h*depth + d)
					}
				}
			}
		}

		
		kGradFlatIndex := 0
		for b := 0; b < batchSizeK; b++ {
			for h := 0; h < numHeadsK; h++ {
				for s := 0; s < seqLengthK; s++ {
					for d := 0; d < depthK; d++ {
						// Calculate flattened index in mha.k.Grad
						kGradFlatIndex = b*numHeadsK*seqLengthK*depthK + h*seqLengthK*depthK + s*depthK + d
	
						// Calculate corresponding flattened index in gradKCombined
						combinedFlatIndexK := b*seqLengthK*dimModelK + s*dimModelK + (h*depthK + d)
	
						// Ensure indices are within bounds
						if kGradFlatIndex >= len(mha.k.Grad.Data) || combinedFlatIndexK >= len(gradKCombined.Data) {
							panic("index out of bounds during gradient summation over heads for K")
						}
	
						// Add the gradient element to the combined gradient
						gradKCombined.Data[combinedFlatIndexK] += mha.k.Grad.Data[kGradFlatIndex]
					}
				}
			}
		}


    	vGradFlatIndex := 0
	for b := 0; b < batchSizeV; b++ {
		for h := 0; h < numHeadsV; h++ {
			for s := 0; s < seqLengthV; s++ {
				for d := 0; d < depthV; d++ {
					// Calculate flattened index in mha.v.Grad
					vGradFlatIndex = b*numHeadsV*seqLengthV*depthV + h*seqLengthV*depthV + s*depthV + d

					// Calculate corresponding flattened index in gradVCombined
					combinedFlatIndexV := b*seqLengthV*dimModelV + s*dimModelV + (h*depthV + d)

					// Ensure indices are within bounds
					if vGradFlatIndex >= len(mha.v.Grad.Data) || combinedFlatIndexV >= len(gradVCombined.Data) {
						panic("index out of bounds during gradient summation over heads for V")
					}

					// Add the gradient element to the combined gradient
					gradVCombined.Data[combinedFlatIndexV] += mha.v.Grad.Data[vGradFlatIndex]
				}
			}
		}
	}
	vTransposed, err := mha.v.Transpose(len(mha.v.Shape)-2, len(mha.v.Shape)-1)
	if err != nil { panic(fmt.Sprintf("failed to transpose v for MHA backward: %v", err)) }

	// Need a batched MatMul operation: (b, h, s, d) @ (b, h, d, s) -> (b, h, s, s)
	gradAttentionWeights, err := gradBeforeConcat.MatMul(vTransposed) // Assuming MatMul handles batches
	if err != nil { panic(fmt.Sprintf("failed to calculate gradAttentionWeights in MHA backward: %v", err)) }

	// Add to existing gradient for attention weights
	if mha.attentionWeights.requiresGrad {
		if mha.attentionWeights.Grad == nil {
			mha.attentionWeights.Grad = NewTensor(make([]float64, len(mha.attentionWeights.Data)), mha.attentionWeights.Shape, false)
		}
		// Accumulate gradAttentionWeights to mha.attentionWeights.Grad element-wise
		for i := range mha.attentionWeights.Grad.Data {
			mha.attentionWeights.Grad.Data[i] += gradAttentionWeights.Data[i]
		}
	}


	// dLoss/dv = mha.attentionWeights^T @ gradBeforeConcat
	// Transpose mha.attentionWeights: swap last two dimensions (seq_len and seq_len)
	attentionWeightsTransposed, err := mha.attentionWeights.Transpose(len(mha.attentionWeights.Shape)-2, len(mha.attentionWeights.Shape)-1)
	if err != nil { panic(fmt.Sprintf("failed to transpose attentionWeights for MHA backward: %v", err)) }

	// Need a batched MatMul operation: (b, h, s, s) @ (b, h, s, d) -> (b, h, s, d)
	gradV, err := attentionWeightsTransposed.MatMul(gradBeforeConcat) // Assuming MatMul handles batches
	if err != nil { panic(fmt.Sprintf("failed to calculate gradV in MHA backward: %v", err)) }

	// Add to existing gradient for v
	if mha.v != nil && mha.v.requiresGrad { // Check if v requires grad (it's an intermediate tensor from input projection)
		if mha.v.Grad == nil {
			mha.v.Grad = NewTensor(make([]float64, len(mha.v.Data)), mha.v.Shape, false)
		}
		// Accumulate gradV to mha.v.Grad element-wise
		for i := range mha.v.Grad.Data {
			mha.v.Grad.Data[i] += gradV.Data[i]
		}
	}
}
// Accumulate finalGradWv to mha.Wv.Grad
		for i := range mha.Wv.Grad.Data {
			mha.Wv.Grad.Data[i] += finalGradWv.Data[i]
		}
	}

	// --- Backpropagate gradients to the original input tensors (queryTensor, keyTensor, valueTensor) ---
	// The gradients for the input tensors are the sum of gradients propagated through the Wq, Wk, and Wv paths.
	// These combined gradients (gradQCombined, gradKCombined, gradVCombined) are already calculated in step 6.
	// We need to backpropagate these combined gradients through the initial linear projections
	// back to the original input tensors.

	// Backpropagate gradQCombined to queryTensor
	// dLoss/dqueryTensor = gradQCombined @ Wq^T
	if mha.queryTensor != nil && mha.queryTensor.requiresGrad && mha.Wq != nil && gradQCombined != nil {
		if mha.queryTensor.Grad == nil {
			mha.queryTensor.Grad = NewTensor(make([]float64, len(mha.queryTensor.Data)), mha.queryTensor.Shape, false)
		}
		// Transpose Wq: swap last two dimensions
		wqTransposedForGradInput, err := mha.Wq.Transpose(len(mha.Wq.Shape)-2, len(mha.Wq.Shape)-1)
		if err != nil {
			panic(fmt.Sprintf("failed to transpose Wq for MHCA queryTensor backward: %v", err))
		}
		// MatMul: gradQCombined [b, query_s, dim_model] @ wqTransposedForGradInput [dim_model, dim_model] ? No, [b, query_s, dim_model] @ [d_m, d_m] is wrong.
		// The dimensions for the matrix multiplication should be [b, query_s, dim_model] @ [dim_model, dim_model] -> [b, query_s, dim_model]
		// This requires a batched matrix multiplication or a way to handle broadcasting.
		// Assuming your MatMul handles batched operations: [b, query_s, dim_model] @ [dim_model, dim_model] -> [b, query_s, dim_model]

		gradInputQ, err := gradQCombined.MatMul(wqTransposedForGradInput) // Assuming MatMul handles batches and correct dimensions
		if err != nil {
			panic(fmt.Sprintf("failed to calculate gradInputQ in MHCA backward: %v", err))
		}
		// Accumulate gradInputQ to mha.queryTensor.Grad element-wise
		for i := range mha.queryTensor.Grad.Data {
			mha.queryTensor.Grad.Data[i] += gradInputQ.Data[i]
		}
	}


	// Backpropagate gradKCombined to keyTensor
	// dLoss/dkeyTensor = gradKCombined @ Wk^T
	if mha.keyTensor != nil && mha.keyTensor.requiresGrad && mha.Wk != nil && gradKCombined != nil {
		if mha.keyTensor.Grad == nil {
			mha.keyTensor.Grad = NewTensor(make([]float64, len(mha.keyTensor.Data)), mha.keyTensor.Shape, false)
		}
		// Transpose Wk: swap last two dimensions
		wkTransposedForGradInput, err := mha.Wk.Transpose(len(mha.Wk.Shape)-2, len(mha.Wk.Shape)-1)
		if err != nil {
			panic(fmt.Sprintf("failed to transpose Wk for MHCA keyTensor backward: %v", err))
		}
		// MatMul: gradKCombined [b, key_s, dim_model] @ wkTransposedForGradInput [dim_model, dim_model] -> [b, key_s, dim_model]
		gradInputK, err := gradKCombined.MatMul(wkTransposedForGradInput) // Assuming MatMul handles batches and correct dimensions
		if err != nil {
			panic(fmt.Sprintf("failed to calculate gradInputK in MHCA backward: %v", err))
		}
		// Accumulate gradInputK to mha.keyTensor.Grad element-wise
		for i := range mha.keyTensor.Grad.Data {
			mha.keyTensor.Grad.Data[i] += gradInputK.Data[i]
		}
	}


	// Backpropagate gradVCombined to valueTensor
	// dLoss/dvalueTensor = gradVCombined @ Wv^T
	if mha.valueTensor != nil && mha.valueTensor.requiresGrad && mha.Wv != nil && gradVCombined != nil {
		if mha.valueTensor.Grad == nil {
			mha.valueTensor.Grad = NewTensor(make([]float64, len(mha.valueTensor.Data)), mha.valueTensor.Shape, false)
		}
		// Transpose Wv: swap last two dimensions
		wvTransposedForGradInput, err := mha.Wv.Transpose(len(mha.Wv.Shape)-2, len(mha.Wv.Shape)-1)
		if err != nil {
			panic(fmt.Sprintf("failed to transpose Wv for MHCA valueTensor backward: %v", err))
		}
		// MatMul: gradVCombined [b, key_s, dim_model] @ wvTransposedForGradInput [dim_model, dim_model] -> [b, key_s, dim_model]
		gradInputV, err := gradVCombined.MatMul(wvTransposedForGradInput) // Assuming MatMul handles batches and correct dimensions
		if err != nil {
			panic(fmt.Sprintf("failed to calculate gradInputV in MHCA backward: %v", err))
		}
		// Accumulate gradInputV to mha.valueTensor.Grad element-wise
		for i := range mha.valueTensor.Grad.Data {
			mha.valueTensor.Grad.Data[i] += gradInputV.Data[i]
		}
	}

	// Add checks for nil intermediate tensors and weights
	// Initialize gradients for mha.Wq.Grad, mha.Wk.Grad, mha.Wv.Grad, mha.Wo.Grad,
	// and mha.queryTensor.Grad, mha.keyTensor.Grad, mha.valueTensor.Grad
	// if requiresGrad is true and Grad is nil.
} // Closing curly brace for the function

// Forward performs the forward pass of the MultiHeadCrossAttention layer.
// query: Input from the decoder layer (shape: [batch_size, target_sequence_length, dim_model]).
// key/value: Input from the encoder output (shape: [batch_size, source_sequence_length, dim_model]).
// mask: Optional mask for attention (e.g., padding mask for encoder output).
func (mha *MultiHeadCrossAttention) Forward(query, key, value *Tensor, mask *Tensor) (*Tensor, error) {
	// Store the original input tensors
	mha.queryTensor = query
	mha.keyTensor = key
	mha.valueTensor = value
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
	}

	// K, V shapes: [batch_size, num_kv_heads, kv_seq_length, head_dim]
	kReshaped, err := k.Reshape([]int{batchSize, kvSeqLength, mha.NumKVHeads, mha.DimKVHeads}) // Use DimKVHeads
	if err != nil {
		return nil, fmt.Errorf("failed to reshape cross-attention key tensor: %w", err)
	}
	mha.k = kReshaped // Store k after reshape/split
	kTransposed, err := kReshaped.Transpose(1, 2)
	if err != nil {
		return nil, fmt.Errorf("failed to transpose cross-attention key tensor: %w", err)
	}

	vReshaped, err := v.Reshape([]int{batchSize, kvSeqLength, mha.NumKVHeads, mha.DimKVHeads}) // Use DimKVHeads
	if err != nil {
		return nil, fmt.Errorf("failed to reshape cross-attention value tensor: %w", err)
	}
	mha.v = vReshaped // Store v after reshape/split
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
	mha.attentionScores = attentionScores // Store attention scores

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
	mha.attentionWeights = attentionWeights // Store attention weights

	// Apply attention weights to Value: Attention Weights @ V
	contextLayer, err := attentionWeights.MatMul(vTransposed)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate cross-attention context layer (Attention@V): %w", err)
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

	// Set creator and requiresGrad for the output tensor
	outputRequiresGrad := query.requiresGrad || key.requiresGrad || value.requiresGrad ||
		mha.QueryLinear.Weights.requiresGrad || (mha.QueryLinear.Biases != nil && mha.QueryLinear.Biases.requiresGrad) ||
		mha.KeyLinear.Weights.requiresGrad || (mha.KeyLinear.Biases != nil && mha.KeyLinear.Biases.requiresGrad) ||
		mha.ValueLinear.Weights.requiresGrad || (mha.ValueLinear.Biases != nil && mha.ValueLinear.Biases.requiresGrad) ||
		mha.OutputLinear.Weights.requiresGrad || (mha.OutputLinear.Biases != nil && mha.OutputLinear.Biases.requiresGrad)

	output.requiresGrad = outputRequiresGrad
	if output.requiresGrad {
		output.creator = mha // Set the creator to the MultiHeadCrossAttention layer itself
	}

	return output, nil
}
