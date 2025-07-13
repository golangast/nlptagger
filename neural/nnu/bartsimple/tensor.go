package bartsimple

import (
	"errors"
	"fmt"
	"math"
)

// Tensor represents a multi-dimensional array of float64 values.
type Tensor struct {
	Data  []float64
	Shape []int
	// Fields for automatic differentiation
	requiresGrad bool
	creator      Operation
	Grad         *Tensor
	// For lazy execution
	Operation Operation
}

// Add performs element-wise addition with another tensor.
func (t *Tensor) Add(other *Tensor) (*Tensor, error) {
	// Ensure the shapes of the two tensors are identical for element-wise addition.
	// Check for compatible shapes (element-wise addition requires same shape)
	if len(t.Shape) != len(other.Shape) {
		return nil, fmt.Errorf("tensor shapes are not compatible for addition: %v and %v", t.Shape, other.Shape)
	}
	for i := range t.Shape {
		if t.Shape[i] != other.Shape[i] {
			return nil, fmt.Errorf("tensor shapes are not compatible for addition: %v and %v", t.Shape, other.Shape)
		}
	}

	// Create a new data slice for the result with the same size as the input tensors.
	// Create a new tensor with the same shape as the input tensors
	resultData := make([]float64, len(t.Data))
	resultShape := make([]int, len(t.Shape))
	copy(resultShape, t.Shape)

	// Iterate through the flattened data and perform element-wise addition.
	// Perform element-wise addition
	for i := range t.Data {
		resultData[i] = t.Data[i] + other.Data[i]
	}

	return NewTensor(resultData, resultShape), nil // Return a new tensor containing the result.
}

// NextBatch moves the iterator to the next batch and returns a Batch tensor view.
// It returns nil and false if there are no more batches.
// This version returns a view struct instead of a new tensor.
type BatchView struct {
	originalTensor *Tensor
	batchIndex     int
	rows           int
	cols           int
}

// NewTensor creates a new Tensor with the given shape and initializes data with zeros.
func NewTensor(data []float64, shape []int) *Tensor {
	// Basic validation (can be expanded)
	expectedSize := 1
	for _, dim := range shape {
		expectedSize *= dim
	}
	if len(data) != expectedSize {
		// In a real implementation, you might want to handle this differently,\n\t\t// maybe return an error or panic. For simplicity here, we\'ll just print a warning
		// and create a zero-filled tensor of the correct size if data length is mismatched.
		data = make([]float64, expectedSize)
	}

	return &Tensor{
		Data:  data,
		Shape: shape,
	}
}

// Get returns the element at the given indices. (Simplified)
func (t *Tensor) Get(indices ...int) (float64, error) {
	// Simplified: Basic index calculation for flattened data
	if len(indices) != len(t.Shape) {
		return 0, fmt.Errorf("incorrect number of indices: expected %d, got %d", len(t.Shape), len(indices))
	}

	flatIndex := 0
	stride := 1
	for i := len(t.Shape) - 1; i >= 0; i-- {
		if indices[i] < 0 || indices[i] >= t.Shape[i] {
			return 0, fmt.Errorf("index %d out of bounds for dimension %d (size %d)", indices[i], i, t.Shape[i])
		}
		flatIndex += indices[i] * stride
		stride *= t.Shape[i]
	}

	if flatIndex >= len(t.Data) {
		return 0, fmt.Errorf("flattened index %d out of bounds for data length %d", flatIndex, len(t.Data))
	}

	return t.Data[flatIndex], nil
}

// Set sets the element at the given indices. (Simplified)
func (t *Tensor) Set(value float64, indices ...int) error {
	// Simplified: Basic index calculation for flattened data
	if len(indices) != len(t.Shape) {
		return fmt.Errorf("incorrect number of indices: expected %d, got %d", len(t.Shape), len(indices))
	}

	flatIndex := 0
	stride := 1
	for i := len(t.Shape) - 1; i >= 0; i-- {
		if indices[i] < 0 || indices[i] >= t.Shape[i] {
			return fmt.Errorf("index %d out of bounds for dimension %d (size %d)", indices[i], i, t.Shape[i])
		}
		flatIndex += indices[i] * stride
		stride *= t.Shape[i]
	}

	if flatIndex >= len(t.Data) {
		return fmt.Errorf("flattened index %d out of bounds for data length %d", flatIndex, len(t.Data))
	}

	t.Data[flatIndex] = value
	return nil
}

// SetElement sets an element in the tensor based on its shape.
func (t *Tensor) SetElement(value float64, indices ...int) {
	// With lazy execution, setting an element might not be done directly
	// if the tensor has a creator. This method might be used during Compute().
	if t.creator != nil {
		// Handle error or panic: cannot set element on a tensor with a creator
		panic("cannot set element on a tensor with a creator")
	}

	if len(indices) != len(t.Shape) {
		panic(fmt.Sprintf("Incorrect number of indices: expected %d, got %d", len(t.Shape), len(indices)))
	}

	flatIndex := 0
	stride := 1
	for i := len(t.Shape) - 1; i >= 0; i-- {
		// if indices[i] < 0 || indices[i] >= t.Shape[i] {
		// 	panic(fmt.Sprintf("Index out of bounds: index %d at dimension %d (shape %v)", indices[i], i, t.Shape))
		// }
		flatIndex += indices[i] * stride
		stride *= t.Shape[i]
	}

	t.Data[flatIndex] = value
}

// ScalarMul performs element-wise multiplication by a scalar.
func (t *Tensor) ScalarMul(scalar float64) (*Tensor, error) {
	// Create a new tensor for the result
	outputData := make([]float64, len(t.Data))
	output := NewTensor(outputData, t.Shape) // Use NewTensor to create the output tensor

	// Perform element-wise multiplication
	for i := range t.Data {
		output.Data[i] = t.Data[i] * scalar
	}

	return output, nil
}

// Softmax applies the softmax function along a specified axis.
// This implementation is numerically stable.
func (t *Tensor) Softmax(axis int) (*Tensor, error) {
	if axis < 0 || axis >= len(t.Shape) {
		return nil, fmt.Errorf("softmax axis %d is out of bounds for tensor shape %v", axis, t.Shape)
	}

	outputData := make([]float64, len(t.Data))
	copy(outputData, t.Data) // Work on a copy

	// Reshape for easier iteration along the specified axis
	// This part depends on your Tensor implementation's ability to handle reshapes or provide views
	// For a simple flattened slice, you'll need to calculate indices manually.

	// Assuming the outputLogits in BartProcessCommand are [batch_size, sequence_length, vocab_size]
	// and you apply softmax along the vocab_size axis (axis 2)
	// The following logic is for softmax along the last axis (axis = len(t.Shape) - 1)

	if axis != len(t.Shape)-1 {
		return nil, errors.New("softmax placeholder only implemented for the last axis")
		// A full implementation would require reshaping or more complex index calculations
	}

	// Assuming the last axis is the one we iterate over for softmax
	sizeOfAxis := t.Shape[axis]
	numVectors := len(t.Data) / sizeOfAxis // Total number of vectors to apply softmax to

	for i := 0; i < numVectors; i++ {
		// Find the maximum value in the current vector (for numerical stability)
		maxVal := outputData[i*sizeOfAxis]
		for j := 1; j < sizeOfAxis; j++ {
			if outputData[i*sizeOfAxis+j] > maxVal {
				maxVal = outputData[i*sizeOfAxis+j]
			}
		}

		// Exponentiate and sum
		sumExp := 0.0
		for j := 0; j < sizeOfAxis; j++ {
			outputData[i*sizeOfAxis+j] = math.Exp(outputData[i*sizeOfAxis+j] - maxVal) // Subtract maxVal
			sumExp += outputData[i*sizeOfAxis+j]
		}

		// Normalize to get probabilities
		if sumExp == 0 {
			// Handle case where sumExp is zero to avoid division by zero
			fmt.Printf("Warning: Sum of exponentiated values is zero for vector %d. Setting probabilities to zero.", i)
			for j := 0; j < sizeOfAxis; j++ {
				outputData[i*sizeOfAxis+j] = 0.0
			}
		} else {
			for j := 0; j < sizeOfAxis; j++ {
				outputData[i*sizeOfAxis+j] /= sumExp
			}
		}

	}

	output := NewTensor(outputData, t.Shape)
	return output, nil
}

// Reshape changes the shape of the tensor and reorders the underlying data based on standard C-order (row-major) indexing.
func (t *Tensor) Reshape(newShape []int) (*Tensor, error) {
	currentSize := 1
	for _, dim := range t.Shape {
		currentSize *= dim
	}
	newSize := 1
	for _, dim := range newShape {
		newSize *= dim
	}

	if currentSize != newSize {
		return nil, fmt.Errorf("cannot reshape tensor of size %d to shape %v (size %d): sizes do not match", currentSize, newShape, newSize)
	}

	if currentSize == 0 {
		// Handle empty tensors
		return NewTensor([]float64{}, newShape), nil
	}

	// Calculate strides for both original and new shapes (C-order/row-major)
	newStrides := calculateStrides(newShape)

	// Create a new data slice for the reshaped tensor
	newData := make([]float64, currentSize) // Use currentSize (which equals newSize)

	// Iterate through the logical indices of the new (reshaped) tensor
	// and calculate the corresponding flattened index in the original tensor
	// based on the new shape and original strides. Then copy the data.

	newIndices := make([]int, len(newShape)) // Slice to hold current indices for new shape

	// Iterate through all elements of the new tensor using their flattened index
	for newFlatIndex := 0; newFlatIndex < newSize; newFlatIndex++ {
		// Calculate multi-dimensional indices for the current flattened index in the new tensor
		tempNewFlatIndex := newFlatIndex
		for i := 0; i < len(newShape); i++ {
			if newStrides[i] > 0 {
				newIndices[i] = tempNewFlatIndex / newStrides[i]
				tempNewFlatIndex %= newStrides[i]
			} else {
				// This case should ideally not happen if newShape is valid and has non-zero dimensions
				// If a dimension is 0, the size is 0, and the loop for newFlatIndex wouldn't run.
				// If it does happen, assume index is 0 for a 0-dimension.
				newIndices[i] = 0
			}
		}

		// Calculate the corresponding flattened index in the original tensor
		// based on the new indices and the original strides.
		// This is the core of the reshape logic: how indices in the new shape
		// map to the flattened data based on the original data layout.
		// For standard C-order reshape, the mapping is simply based on the
		// linear index. The `newFlatIndex` corresponds to the `originalFlatIndex`
		// if the underlying data layout remains the same.

		// The previous simple data copy was actually closer to the standard C-order
		// reshape where the memory layout doesn't change, only the interpretation
		// through strides and shape.

		// Let's reconsider: the issue might not be reordering but accessing
		// the data in `AddWithBroadcast` based on potentially incorrect indices
		// derived from a reshaped tensor.

		// Let's go back to a simpler Reshape that just copies data, and focus
		// on debugging index calculation in operations like AddWithBroadcast and MatMul
		// when they receive reshaped tensors.

		// Reverting Reshape to a simple data copy is more consistent with how
		// many tensor libraries handle standard reshape when data is contiguous.

		// If you need a Transpose operation, use your existing `Transpose` method.
		// If you need other specific dimension permutations, you'll need a separate
		// operation that maps indices accordingly.

		// The error is likely coming from how operations *consume* the reshaped tensor,
		// not necessarily how `Reshape` itself creates the new tensor object.

		// Let's keep the simple data copy implementation for Reshape and focus
		// our debugging efforts on the operations that use the reshaped tensors,
		// particularly in the Multi-Head Attention where Reshape is used to split/combine heads.

		// This means the recursive index mapping within Reshape is not the standard
		// behavior for a simple reshape; it's for operations like Transpose or
		// arbitrary dimension permutations.

		// Let's ensure the simple data copy Reshape is in place and focus on
		// debugging where it's used.

		// Reverting to the simple data copy:
		newData := make([]float64, len(t.Data)) // Still make a copy to avoid modifying original tensor's data slice
		copy(newData, t.Data)

		return NewTensor(newData, newShape), nil // Return new tensor with copied data and new shape

		// The code below this point (recursive copyData) is for operations that
		// reorder data, not standard reshape.
	}

	// Remove the recursive data copying logic here.
	// It's not standard for a basic Reshape assuming contiguous data.

	// The issue is likely in how operations that take a reshaped tensor
	// calculate which element in the flattened data corresponds to a given
	// multi-dimensional index in the *new* shape, using the *original* strides.

	// Let's go back to the debug prints in the Decoder's Forward pass and
	// within the Multi-Head Attention to see the shapes and data *after*
	// Reshape is applied and *before* and *after* matrix multiplications.

	// Since the output still shows the "Warning: Using placeholder Reshape",
	// the most immediate task is to ensure the placeholder is replaced
	// with *any* implementation, even the simple data copy one, so that the
	// warning goes away and we know the intended Reshape is being called.
	// Then we can debug how that reshaped tensor is being used.

	// Let's ensure this simple data copy version is the one in your tensor.go:
	newData = make([]float64, len(t.Data))
	copy(newData, t.Data)
	return NewTensor(newData, newShape), nil
}

// AddWithBroadcast performs element-wise addition with broadcasting.\n\n\n// AddWithBroadcast performs element-wise addition with broadcasting.
func (t *Tensor) AddWithBroadcast(other *Tensor) (*Tensor, error) {
	// Determine the output shape based on broadcasting rules
	outputShape, err := calculateBroadcastShape(t.Shape, other.Shape)
	if err != nil {
		return nil, fmt.Errorf("broadcasting failed: %w", err)
	}

	outputSize := 1
	for _, dim := range outputShape {
		outputSize *= dim
	}
	outputData := make([]float64, outputSize) // Initialize with zeros

	// Calculate strides for input tensors and output tensor
	tStride := calculateStrides(t.Shape)
	otherStride := calculateStrides(other.Shape)
	outputStride := calculateStrides(outputShape)

	// Iterate over the flattened index of the output tensor
	outputIndices := make([]int, len(outputShape)) // Reuse slice
	for i := 0; i < outputSize; i++ {
		// Calculate multi-dimensional indices for the current flattened index 'i' in the output tensor
		tempNewFlatIndex := i
		for j := 0; j < len(outputShape); j++ {
			// This part of index calculation seems correct for C-order (row-major)
			if outputStride[j] > 0 { // Avoid division by zero
				outputIndices[j] = tempNewFlatIndex / outputStride[j]
				tempNewFlatIndex %= outputStride[j]
			} else {
				// This case (outputStride[j] == 0) should ideally not happen if outputShape is valid
				// and calculated based on broadcasting non-zero dimensions.
				// If an output dimension is 0, the size is 0, and this loop wouldn't run.
				// If it does happen, it might indicate an issue in calculateBroadcastShape.
				// For now, assume it means the index for a 0-dimension is 0.
				outputIndices[j] = 0
			}
		}

		// Calculate the corresponding indices in the input tensors based on broadcasting rules
		// Aligning from the right: If an input dimension is 1, the index is always 0.
		// Otherwise, the input index matches the corresponding output index.

		tIndices := make([]int, len(t.Shape))         // Reuse slice
		otherIndices := make([]int, len(other.Shape)) // Reuse slice

		// Map output indices back to input indices based on broadcasting
		// Iterate from the rightmost dimension (highest index in shape slice)
		for j := 1; j <= len(outputShape); j++ {
			// Index in outputShape slice, starting from right (maxLen-j)
			outputDimIndex := len(outputShape) - j

			// Corresponding index in input shapes, aligned from the right
			tDimIndex := len(t.Shape) - j
			otherDimIndex := len(other.Shape) - j

			// Map the output index back to the input index
			if tDimIndex >= 0 { // Ensure index is within bounds of input shape
				if t.Shape[tDimIndex] == 1 {
					tIndices[tDimIndex] = 0 // Broadcasted dimension
				} else {
					tIndices[tDimIndex] = outputIndices[outputDimIndex] // Matching dimension
				}
			}
			// If tDimIndex < 0, this dimension was added by broadcasting, no corresponding input index

			if otherDimIndex >= 0 { // Ensure index is within bounds of input shape
				if other.Shape[otherDimIndex] == 1 {
					otherIndices[otherDimIndex] = 0 // Broadcasted dimension
				} else {
					otherIndices[otherDimIndex] = outputIndices[outputDimIndex] // Matching dimension
				}
			}
			// If otherDimIndex < 0, this dimension was added by broadcasting, no corresponding input index
		}

		// Calculate the flattened indices in the input tensors using their strides
		tFlatIndex := 0
		// Iterate through the input indices and strides
		for j := 0; j < len(tIndices); j++ {
			// Add bounds check for j
			if j >= len(tStride) {
				// This should not happen if calculateStrides is correct
				panic("internal error: tIndices length exceeds tStride length")
			}
			tFlatIndex += tIndices[j] * tStride[j]
		}

		otherFlatIndex := 0
		// Iterate through the input indices and strides
		for j := 0; j < len(otherIndices); j++ {
			// Add bounds check for j
			if j >= len(otherStride) {
				// This should not happen if calculateStrides is correct
				panic("internal error: otherIndices length exceeds otherStride length")
			}
			otherFlatIndex += otherIndices[j] * otherStride[j]
		}

		// Add the elements and store in the output tensor
		// Ensure flattened indices are within the data bounds
		if tFlatIndex < 0 || tFlatIndex >= len(t.Data) || otherFlatIndex < 0 || otherFlatIndex >= len(other.Data) || i < 0 || i >= len(outputData) {
			// This indicates an error in index calculation or broadcasting logic
			return nil, fmt.Errorf("internal error: index out of bounds during broadcasted addition. tFlatIndex: %d/%d, otherFlatIndex: %d/%d, outputFlatIndex: %d/%d",
				tFlatIndex, len(t.Data), otherFlatIndex, len(other.Data), i, len(outputData))
		}

		outputData[i] = t.Data[tFlatIndex] + other.Data[otherFlatIndex]
	}

	return NewTensor(outputData, outputShape), nil
}

// calculateBroadcastShape calculates the resulting shape after broadcasting two shapes.
// (Keep your existing calculateBroadcastShape function here)
func calculateBroadcastShape(shape1, shape2 []int) ([]int, error) {
	// Assume shapes are aligned to the right
	// Example: shape1 [2, 3], shape2 [3] -> broadcasted [2, 3]
	// Example: shape1 [2, 3], shape2 [1, 3] -> broadcasted [2, 3]
	// Example: shape1 [2, 3], shape2 [1] -> broadcasted [2, 3]
	// Example: shape1 [2, 3], shape2 [4] -> Error

	maxLen := max(len(shape1), len(shape2))
	broadcastedShape := make([]int, maxLen)

	// Iterate from the right
	for i := 1; i <= maxLen; i++ {
		d1 := 1
		if len(shape1) >= i {
			d1 = shape1[len(shape1)-i]
		}

		d2 := 1
		if len(shape2) >= i {
			d2 = shape2[len(shape2)-i]
		}

		if d1 != d2 && d1 != 1 && d2 != 1 {
			return nil, fmt.Errorf("shapes %v and %v cannot be broadcasted", shape1, shape2)
		}

		broadcastedShape[maxLen-i] = max(d1, d2)
	}
	return broadcastedShape, nil
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// Transpose swaps two dimensions of the tensor.
func (t *Tensor) Transpose(axis1, axis2 int) (*Tensor, error) {
	if axis1 < 0 || axis1 >= len(t.Shape) || axis2 < 0 || axis2 >= len(t.Shape) {
		return nil, fmt.Errorf("invalid axes for transpose: %d, %d", axis1, axis2)
	}

	// Create the new shape with swapped dimensions
	newShape := make([]int, len(t.Shape))
	copy(newShape, t.Shape)
	newShape[axis1], newShape[axis2] = newShape[axis2], newShape[axis1]

	// Calculate strides for both original and new shapes
	originalStrides := calculateStrides(t.Shape)
	newStrides := calculateStrides(newShape)

	// Create a new data slice for the transposed tensor
	newData := make([]float64, len(t.Data))

	// Iterate through the logical indices of the new (transposed) tensor
	// and copy data from the corresponding original logical index.
	// We can use a recursive helper function for multi-dimensional indexing.

	var copyData func(newIndices, originalIndices []int, dim int)
	copyData = func(newIndices, originalIndices []int, dim int) {
		if dim == len(newShape) {
			// Base case: We have a complete set of indices for both tensors
			// Calculate flattened indices
			newFlatIndex := 0
			for i := range newIndices {
				newFlatIndex += newIndices[i] * newStrides[i]
			}

			originalFlatIndex := 0
			for i := range originalIndices {
				originalFlatIndex += originalIndices[i] * originalStrides[i]
			}

			// Copy the data
			newData[newFlatIndex] = t.Data[originalFlatIndex]
			return
		}

		// Recursive step: Iterate through the current dimension
		for i := 0; i < newShape[dim]; i++ {
			// Update indices for the current dimension
			currentNewIndices := make([]int, len(newIndices))
			copy(currentNewIndices, newIndices)
			currentNewIndices[dim] = i

			currentOriginalIndices := make([]int, len(originalIndices))
			copy(currentOriginalIndices, originalIndices)

			// Map the new index 'i' in the transposed dimension 'dim'
			// back to the corresponding index in the original dimensions.
			// If 'dim' in new shape was 'axis1' in original shape,
			// the original index is 'i' at 'axis2'.
			// If 'dim' in new shape was 'axis2' in original shape,
			// the original index is 'i' at 'axis1'.
			// Otherwise, the index remains 'i' at the corresponding dimension.

			originalDim := dim
			if dim == axis1 {
				originalDim = axis2
			} else if dim == axis2 {
				originalDim = axis1
			}
			currentOriginalIndices[originalDim] = i

			// Recurse for the next dimension
			copyData(currentNewIndices, currentOriginalIndices, dim+1)
		}
	}

	// Start the recursive data copying from dimension 0
	initialNewIndices := make([]int, len(newShape))
	initialOriginalIndices := make([]int, len(t.Shape)) // Initialize with zeros
	copyData(initialNewIndices, initialOriginalIndices, 0)

	return NewTensor(newData, newShape), nil
}

// calculateStrides calculates the strides for accessing elements in a flattened tensor data slice.
// (Keep your existing calculateStrides function here)
func calculateStrides(shape []int) []int {
	ndim := len(shape)
	strides := make([]int, ndim)
	stride := 1
	for i := ndim - 1; i >= 0; i-- {
		strides[i] = stride
		if shape[i] > 0 {
			stride *= shape[i]
		}
	}
	return strides
}

// Compute evaluates the tensor's value by executing the computation graph that created it.
func (t *Tensor) Compute() (*Tensor, error) {
	if t.creator == nil {
		// If the tensor does not have a creator, it is an input tensor,
		// and its data is already available.
		return t, nil
	}

	// If the tensor's data is already computed, return it.
	if t.Data != nil {
		return t, nil
	}

	// Build a topological order of the computation graph.
	var visitedNodes map[*Tensor]bool = make(map[*Tensor]bool)
	var topologicalOrder []*Tensor

	var buildTopologicalOrder func(tensor *Tensor)
	buildTopologicalOrder = func(tensor *Tensor) {
		if !visitedNodes[tensor] {
			visitedNodes[tensor] = true
			if tensor.creator != nil {
				// Get the input tensors from the creator operation.
				inputs := (tensor.creator).Inputs() // Assuming Inputs() method exists in Operation interface
				for _, inputTensor := range inputs {
					buildTopologicalOrder(inputTensor)
				}
			}
			topologicalOrder = append(topologicalOrder, tensor)
		}
	}

	// Start building the topological order from the current tensor.
	buildTopologicalOrder(t)

	// The topological order is currently from inputs to output.
	// We need to execute operations in this order.

	// Execute the operations in topological order.
	for _, tensor := range topologicalOrder {
		if tensor.creator != nil && tensor.Data == nil {
			// If the tensor has a creator and its data is not yet computed,
			// execute the creator operation's Forward method.
			// The Forward method should compute the tensor's data and store it.
			_ = tensor.creator.Forward() // Assuming Forward() computes and sets tensor.Data
			// You might want to add error handling here if Forward() can return an error.
		}
	}

	// After executing all necessary operations, the current tensor's data should be computed.
	if t.Data == nil {
		return nil, errors.New("failed to compute tensor data")
	}

	return t, nil
}
