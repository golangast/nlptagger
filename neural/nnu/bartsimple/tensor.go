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
	Grad  *Tensor
	Mask  *Tensor
	// Fields for automatic differentiation and lazy execution
	creator Operation
	// For lazy execution
	requiresGrad bool
	Operation    Operation
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

	resultTensor := NewTensor(resultData, resultShape, false)
	// Set the creator and inputs for backpropagation
	if t.requiresGrad || other.requiresGrad {
		resultTensor.creator = &addOperation{input1: t, input2: other}
		resultTensor.requiresGrad = true
	}
	return resultTensor, nil
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

// NewTensor creates a new Tensor with the given data, shape, and requiresGrad flag.
// It initializes data if nil and allocates space for gradients if requiresGrad is true.
func NewTensor(data []float64, shape []int, requiresGrad bool) *Tensor {
	// Basic validation (can be expanded)
	expectedSize := 1
	for _, dim := range shape {
		expectedSize *= dim
	}
	if data != nil && len(data) != expectedSize {
		// In a real implementation, you might want to handle this differently,
		// maybe return an error or panic. For simplicity here, we'll just print a warning
		// and create a zero-filled tensor of the correct size if data length is mismatched.
		fmt.Printf("Warning: Mismatched data length and shape. Expected size %d, got %d. Initializing with zeros.\n", expectedSize, len(data))
		data = make([]float64, expectedSize)
	} else if data == nil {
		data = make([]float64, expectedSize)
	}

	var grad *Tensor
	if requiresGrad {
		// Allocate space for gradients only if requiresGrad is true
		grad = NewTensor(make([]float64, expectedSize), shape, false) // Gradients themselves don't require gradients
	}

	return &Tensor{
		Data:  data,
		Shape: shape,
		Grad:  grad,
		// Initialize requiresGrad field
		requiresGrad: requiresGrad,
	}
}

// ZeroGrad resets the gradient of the tensor to zero.
// It iterates through the gradient's data slice and sets each element to 0.
func (t *Tensor) ZeroGrad() {
	if t.Grad != nil {
		for i := range t.Grad.Data {
			t.Grad.Data[i] = 0
		}
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

// Mul performs element-wise multiplication with another tensor.
func (t *Tensor) Mul(other *Tensor) (*Tensor, error) {
	// Ensure the shapes are compatible for element-wise multiplication
	if len(t.Shape) != len(other.Shape) {
		return nil, fmt.Errorf("tensor shapes are not compatible for multiplication: %v and %v", t.Shape, other.Shape)
	}
	for i := range t.Shape {
		if t.Shape[i] != other.Shape[i] {
			return nil, fmt.Errorf("tensor shapes are not compatible for multiplication: %v and %v", t.Shape, other.Shape)
		}
	}

	// Create a new tensor for the result
	resultData := make([]float64, len(t.Data))
	resultShape := make([]int, len(t.Shape))
	copy(resultShape, t.Shape)

	resultTensor := NewTensor(resultData, resultShape, true)
	// Set the creator and inputs for backpropagation
	resultTensor.creator = &mulOperation{input1: t, input2: other, output: resultTensor}

	return resultTensor, nil
}

// ScalarMul performs element-wise multiplication by a scalar.
func (t *Tensor) ScalarMul(scalar float64) (*Tensor, error) {
	// Create a new tensor for the result
	outputData := make([]float64, len(t.Data))
	// The output requires grad if the input tensor does.
	output := NewTensor(outputData, t.Shape, t.requiresGrad)

	// Perform element-wise multiplication
	for i := range t.Data {
		output.Data[i] = t.Data[i] * scalar
	}

	return output, nil
}

// Softmax applies the softmax function along a specified axis. (Placeholder implementation for the last axis)

// Softmax applies the softmax function along a specified axis.
// This implementation is a placeholder and currently only works along the last axis.
// This implementation is numerically stable.
func (t *Tensor) Softmax(axis int) (*Tensor, error) {
	// Create the softmax operation
	operation := &softmaxOperation{input: t, axis: axis}

	// Perform the forward pass using the operation
	outputTensor, err := operation.Forward(t)
	if err != nil {
		return nil, fmt.Errorf("error during softmax forward pass: %w", err)
	}

	return outputTensor, nil
}

// Reshape changes the shape of the tensor. This implementation simply copies the data.
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

	newData := make([]float64, len(t.Data))
	copy(newData, t.Data)

	return NewTensor(newData, newShape, t.requiresGrad), nil
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

	return NewTensor(newData, newShape, t.requiresGrad), nil
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
			outputTensor, err := tensor.creator.Forward()
			if err != nil {
				// Handle the error appropriately, e.g., return it or log it
				return nil, fmt.Errorf("error during forward pass for tensor creator: %w", err)
			}
			// Assign the computed data to the tensor
			tensor.Data = outputTensor.Data
			// Assuming Forward() computes and sets tensor.Data
			// You might want to add error handling here if Forward() can return an error.
		}
	}

	// After executing all necessary operations, the current tensor's data should be computed.
	if t.Data == nil {
		return nil, errors.New("failed to compute tensor data")
	}

	return t, nil
}

// Operation interface defines the methods required for an operation in the computation graph.
type Operation interface {
	Forward(...*Tensor) (*Tensor, error) // Use variadic arguments if the number of inputs varies
	Backward(*Tensor) error
	Inputs() []*Tensor
}

// addOperation represents the element-wise addition operation.
type addOperation struct {
	input1 *Tensor
	input2 *Tensor
	output *Tensor // Store output tensor to set its data during forward
}

// Forward performs the forward pass of the addition operation.
// Forward performs the forward pass of the addition operation.
func (op *addOperation) Forward(inputs ...*Tensor) (*Tensor, error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("addOperation requires exactly two input tensors, got %d", len(inputs))
	}

	op.input1 = inputs[0]
	op.input2 = inputs[1]

	// Perform element-wise addition
	if len(op.input1.Data) != len(op.input2.Data) {
		return nil, fmt.Errorf("input tensor data lengths do not match for addition: %d vs %d", len(op.input1.Data), len(op.input2.Data))
	}

	op.output = NewTensor(make([]float64, len(op.input1.Data)), op.input1.Shape, op.input1.requiresGrad || op.input2.requiresGrad) // Assuming element-wise op preserves shape and combines requiresGrad

	for i := range op.input1.Data {
		op.output.Data[i] = op.input1.Data[i] + op.input2.Data[i]
	}

	// Set the creator of the output tensor
	if op.output.requiresGrad {
		op.output.creator = op // Set the creator to the addOperation itself
	}

	return op.output, nil // Return the output tensor and nil for no error
}

func (op *addOperation) Inputs() []*Tensor { return []*Tensor{op.input1, op.input2} }

// mulOperation represents the element-wise multiplication operation.
type mulOperation struct {
	input1 *Tensor
	input2 *Tensor
	output *Tensor // Store output tensor to set its data during forward
}

// Forward performs the forward pass of the multiplication operation.
func (op *mulOperation) Forward(inputs ...*Tensor) (*Tensor, error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("mulOperation requires exactly two input tensors, got %d", len(inputs))
	}

	op.input1 = inputs[0]
	op.input2 = inputs[1]

	if len(op.input1.Data) != len(op.input2.Data) {
		return nil, fmt.Errorf("input tensor data lengths do not match for multiplication: %d vs %d", len(op.input1.Data), len(op.input2.Data))
	}

	// Create the output tensor
	op.output = NewTensor(make([]float64, len(op.input1.Data)), op.input1.Shape, op.input1.requiresGrad || op.input2.requiresGrad)

	// Perform element-wise multiplication
	for i := range op.input1.Data {
		op.output.Data[i] = op.input1.Data[i] * op.input2.Data[i]
	}

	// Set the creator of the output tensor
	if op.output.requiresGrad {
		op.output.creator = op // Set the creator to the mulOperation itself
	}

	return op.output, nil // Return the output tensor and nil for no error
}

// Backward performs the backward pass for the multiplication operation.
func (op *mulOperation) Backward(grad *Tensor) error {
	// Gradient of multiplication: d(x*y)/dx = y, d(x*y)/dy = x
	// Using the chain rule: dLoss/dx = dLoss/dOutput * dOutput/dx
	// dLoss/dx = grad * input2.Data
	// dLoss/dy = grad * input1.Data

	if op.input1.requiresGrad {
		if op.input1.Grad == nil {
			op.input1.Grad = NewTensor(make([]float64, len(op.input1.Data)), op.input1.Shape, true)
		}
		for i := range op.input1.Grad.Data {
			op.input1.Grad.Data[i] += grad.Data[i] * op.input2.Data[i]
		}
	}

	if op.input2.requiresGrad {
		if op.input2.Grad == nil {
			op.input2.Grad = NewTensor(make([]float64, len(op.input2.Data)), op.input2.Shape, true)
		}
		for i := range op.input2.Grad.Data {
			op.input2.Grad.Data[i] += grad.Data[i] * op.input1.Data[i]
		}
	}
	return nil
}

// Inputs returns the input tensors of the multiplication operation.
func (op *mulOperation) Inputs() []*Tensor {
	return []*Tensor{op.input1, op.input2}
}

// Backward initiates the backpropagation process from this tensor.
func (t *Tensor) Backward(grad *Tensor) {
	// If the tensor does not require gradients, stop backpropagation.
	if !t.requiresGrad {
		return
	}

	// If the gradient is not provided (e.g., for the loss tensor), initialize it with ones.
	if grad == nil {
		// Initialize gradient with ones if it's the starting point of backprop
		if t.Shape == nil || len(t.Shape) == 0 {
			// Handle scalar tensor case
			t.Grad = NewTensor([]float64{1.0}, []int{1}, false) // Gradient of a scalar with respect to itself is 1
		} else {
			// Initialize gradient with ones tensor of the same shape
			onesData := make([]float64, len(t.Data))
			for i := range onesData {
				onesData[i] = 1.0
			}
			t.Grad = NewTensor(onesData, t.Shape, false) // Gradients themselves don't require gradients
		}
	} else {
		// If a gradient is provided (from the next operation in the backward pass),
		// accumulate it. This is important for tensors that are inputs to multiple operations.
		if t.Grad == nil {
			t.Grad = NewTensor(make([]float64, len(t.Data)), t.Shape, false)
		}
		// Add the incoming gradient to the existing gradient
		if len(t.Grad.Data) != len(grad.Data) {
			panic("gradient data length mismatch during accumulation")
		}
		for i := range t.Grad.Data {
			t.Grad.Data[i] += grad.Data[i]
		}
	}

	// Build a topological order of the computation graph in reverse.
	// This involves traversing the graph backward from the current tensor's creator.
	// The 'creator' field links a tensor to the operation that produced it.
	// The 'Inputs()' method of the Operation interface gives the input tensors to that operation.
	// We need to process operations in the reverse order of execution in the forward pass.

	// To perform backpropagation, we need to traverse the computation graph
	// in reverse topological order. Each node in our traversal will be a tensor.
	var topologicalOrder []*Tensor
	visited := make(map[*Tensor]bool)

	var buildTopologicalOrder func(t *Tensor)
	buildTopologicalOrder = func(t *Tensor) {
		if visited[t] {
			return
		}
		visited[t] = true

		if t.creator != nil {
			// Recursively visit the inputs of the creator operation
			inputs := t.creator.Inputs()
			for _, inputTensor := range inputs {
				buildTopologicalOrder(inputTensor)
			}
		}
		// Add the tensor to the order after visiting all its dependencies.
		topologicalOrder = append(topologicalOrder, t)
	}

	// Start building the topological order from the current tensor 't'.
	buildTopologicalOrder(t)

	// Iterate through the topologically sorted tensors in reverse order and call Backward.
	for i := len(topologicalOrder) - 1; i >= 0; i-- {
		tensor := topologicalOrder[i]

		if tensor.creator != nil {
			// The gradient for the inputs of this operation is calculated
			// by calling the creator's Backward method with the gradient of the output (this tensor).
			err := tensor.creator.Backward(tensor.Grad)
			if err != nil {
				// Handle the error during backward pass
				fmt.Printf("Error during backward pass for operation: %v\n", err)
			}
		}
	}

}

// addWithBroadcastOperation represents the element-wise addition operation with broadcasting.
type addWithBroadcastOperation struct {
	input1 *Tensor
	input2 *Tensor
	output *Tensor // Store output tensor
	// Store the original shapes to determine broadcasted dimensions in backward
	shape1 []int
	shape2 []int
}

// Inputs returns the input tensors of the broadcasted addition operation.
func (op *addWithBroadcastOperation) Inputs() []*Tensor {
	return []*Tensor{op.input1, op.input2}
}

// Forward performs the forward pass of the broadcasted addition operation.
func (op *addWithBroadcastOperation) Forward(inputs ...*Tensor) (*Tensor, error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("addWithBroadcastOperation requires exactly two input tensors, got %d", len(inputs))
	}

	op.input1 = inputs[0]
	op.input2 = inputs[1]

	// Determine the output shape based on broadcasting rules
	outputShape, err := calculateBroadcastShape(op.input1.Shape, op.input2.Shape)
	if err != nil {
		return nil, fmt.Errorf("broadcasting failed in addWithBroadcastOperation: %w", err)
	}

	outputSize := 1
	for _, dim := range outputShape {
		outputSize *= dim
	}

	// Create the output tensor
	op.output = NewTensor(make([]float64, outputSize), outputShape, op.input1.requiresGrad || op.input2.requiresGrad) // Initialize with zeros and set shape

	op.output.Data = make([]float64, outputSize) // Initialize with zeros
	op.output.Shape = outputShape                // Set the output shape

	// Store original shapes for backward pass
	op.shape1 = op.input1.Shape
	op.shape2 = op.input2.Shape

	// Calculate strides for input tensors and output tensor
	tStride := calculateStrides(op.input1.Shape)
	otherStride := calculateStrides(op.input2.Shape)
	outputStride := calculateStrides(outputShape)

	// Iterate over the flattened index of the output tensor
	outputIndices := make([]int, len(outputShape)) // Reuse slice
	for i := 0; i < outputSize; i++ {
		// Calculate multi-dimensional indices for the current flattened index 'i' in the output tensor
		tempNewFlatIndex := i
		for j := 0; j < len(outputShape); j++ {
			if outputStride[j] > 0 {
				outputIndices[j] = tempNewFlatIndex / outputStride[j]
				tempNewFlatIndex %= outputStride[j]
			} else {
				outputIndices[j] = 0
			}
		}

		// Calculate the corresponding indices in the input tensors based on broadcasting rules
		tIndices := make([]int, len(op.input1.Shape))     // Reuse slice
		otherIndices := make([]int, len(op.input2.Shape)) // Reuse slice

		// Map output indices back to input indices based on broadcasting
		for j := 1; j <= len(outputShape); j++ {
			outputDimIndex := len(outputShape) - j

			tDimIndex := len(op.input1.Shape) - j
			otherDimIndex := len(op.input2.Shape) - j

			if tDimIndex >= 0 {
				if op.input1.Shape[tDimIndex] == 1 {
					tIndices[tDimIndex] = 0
				} else {
					tIndices[tDimIndex] = outputIndices[outputDimIndex]
				}
			}

			if otherDimIndex >= 0 {
				if op.input2.Shape[otherDimIndex] == 1 {
					otherIndices[otherDimIndex] = 0
				} else {
					otherIndices[otherDimIndex] = outputIndices[outputDimIndex]
				}
			}
		}

		// Calculate the flattened indices in the input tensors using their strides
		tFlatIndex := 0
		for j := 0; j < len(tIndices); j++ {
			if j >= len(tStride) {
				panic("internal error: tIndices length exceeds tStride length")
			}
			tFlatIndex += tIndices[j] * tStride[j]
		}

		otherFlatIndex := 0
		for j := 0; j < len(otherIndices); j++ {
			if j >= len(otherStride) {
				panic("internal error: otherIndices length exceeds otherStride length")
			}
			otherFlatIndex += otherIndices[j] * otherStride[j]
		}

		// Add the elements and store in the output tensor
		if tFlatIndex < 0 || tFlatIndex >= len(op.input1.Data) || otherFlatIndex < 0 || otherFlatIndex >= len(op.input2.Data) || i < 0 || i >= len(op.output.Data) {
			panic(fmt.Sprintf("internal error: index out of bounds during broadcasted addition. tFlatIndex: %d/%d, otherFlatIndex: %d/%d, outputFlatIndex: %d/%d",
				tFlatIndex, len(op.input1.Data), otherFlatIndex, len(op.input2.Data), i, len(op.output.Data)))
		}

		op.output.Data[i] = op.input1.Data[tFlatIndex] + op.input2.Data[otherFlatIndex]
	}

	// Determine if output requires gradients
	outputRequiresGrad := op.input1.requiresGrad || op.input2.requiresGrad
	op.output.requiresGrad = outputRequiresGrad
	if outputRequiresGrad {
		op.output.creator = op // Set the creator only if gradients are required
	}

	return op.output, nil
}

// Backward performs the backward pass for the broadcasted addition operation.
// grad is the gradient from the output (dLoss/dOutput).
func (op *addWithBroadcastOperation) Backward(grad *Tensor) error {
	if grad == nil || grad.Data == nil {
		return nil
	}

	if op.input1.requiresGrad {
		if op.input1.Grad == nil {
			op.input1.Grad = NewTensor(make([]float64, len(op.input1.Data)), op.input1.Shape, false)
		}
	}
	if op.input2.requiresGrad {
		if op.input2.Grad == nil {
			op.input2.Grad = NewTensor(make([]float64, len(op.input2.Data)), op.input2.Shape, false)
		}
	}

	gradStride := calculateStrides(grad.Shape)
	input1Stride := calculateStrides(op.shape1)
	input2Stride := calculateStrides(op.shape2)

	// For input1
	if op.input1.requiresGrad {
		for i := 0; i < len(grad.Data); i++ {
			input1FlatIndex, err := mapOutputIndexToInputIndex(grad.Shape, op.shape1, i, gradStride, input1Stride)
			if err != nil {
				return fmt.Errorf("error mapping index for input1 gradient: %w", err)
			}
			op.input1.Grad.Data[input1FlatIndex] += grad.Data[i]
		}
	}

	// For input2
	if op.input2.requiresGrad {
		for i := 0; i < len(grad.Data); i++ {
			input2FlatIndex, err := mapOutputIndexToInputIndex(grad.Shape, op.shape2, i, gradStride, input2Stride)
			if err != nil {
				return fmt.Errorf("error mapping index for input2 gradient: %w", err)
			}
			op.input2.Grad.Data[input2FlatIndex] += grad.Data[i]
		}
	}

	return nil
}

// mapOutputIndexToInputIndex maps a flattened output tensor index to a flattened input tensor index,
// considering broadcasting rules. It sums gradients over broadcasted dimensions.
func mapOutputIndexToInputIndex(outputShape, inputShape []int, outputFlatIndex int, outputStride, inputStride []int) (int, error) {
	outputIndices := make([]int, len(outputShape))
	tempFlatIndex := outputFlatIndex
	for j := 0; j < len(outputShape); j++ {
		if outputStride[j] > 0 {
			outputIndices[j] = tempFlatIndex / outputStride[j]
			tempFlatIndex %= outputStride[j]
		} else {
			outputIndices[j] = 0
		}
	}

	inputIndices := make([]int, len(inputShape))
	for j := 1; j <= len(outputShape); j++ {
		outputDimIndex := len(outputShape) - j
		inputDimIndex := len(inputShape) - j

		if inputDimIndex >= 0 {
			if inputShape[inputDimIndex] == 1 {
				inputIndices[inputDimIndex] = 0 // Broadcasted dimension
			} else {
				inputIndices[inputDimIndex] = outputIndices[outputDimIndex] // Matching dimension
			}
		}
	}

	inputFlatIndex := 0
	for j := 0; j < len(inputIndices); j++ {
		if j >= len(inputStride) {
			return 0, errors.New("internal error: inputIndices length exceeds inputStride length")
		}
		inputFlatIndex += inputIndices[j] * inputStride[j]
	}
	return inputFlatIndex, nil
}

func (t *Tensor) AddWithBroadcast(other *Tensor) (*Tensor, error) {
	// Determine the output shape based on broadcasting rules
	outputShape, err := calculateBroadcastShape(t.Shape, other.Shape)
	if err != nil {
		return nil, fmt.Errorf("broadcasting failed: %w", err)
	}

	// Create a placeholder output tensor
	outputTensor := NewTensor(nil, outputShape, t.requiresGrad || other.requiresGrad) // Output requires grad if any input requires grad

	// Create the addWithBroadcastOperation
	operation := &addWithBroadcastOperation{input1: t, input2: other, output: outputTensor}

	// Set the creator of the output tensor
	if outputTensor.requiresGrad {
		outputTensor.creator = operation
	}

	// Perform the forward pass
	outputTensor, err = operation.Forward(t, other) // Call the Forward method of the operation
	if err != nil {
		return nil, fmt.Errorf("error during forward pass: %w", err)
	}
	return outputTensor, nil

}

// softmaxOperation represents the softmax activation function.
type softmaxOperation struct {
	input  *Tensor
	output *Tensor // Store output for backward pass
	axis   int
}

// Forward performs the forward pass of the softmax operation.
func (op *softmaxOperation) Forward(inputs ...*Tensor) (*Tensor, error) {
	if len(inputs) != 1 {
		return nil, fmt.Errorf("softmaxOperation requires exactly one input tensor, got %d", len(inputs))
	}
	op.input = inputs[0]

	if op.axis < 0 || op.axis >= len(op.input.Shape) {
		return nil, fmt.Errorf("softmax axis %d is out of bounds for tensor shape %v", op.axis, op.input.Shape)
	}

	outputData := make([]float64, len(op.input.Data))
	copy(outputData, op.input.Data) // Work on a copy

	// Assuming the last axis is the one we iterate over for softmax
	if op.axis != len(op.input.Shape)-1 {
		return nil, errors.New("softmax operation only implemented for the last axis")
	}

	sizeOfAxis := op.input.Shape[op.axis]
	numVectors := len(op.input.Data) / sizeOfAxis // Total number of vectors to apply softmax to

	for i := 0; i < numVectors; i++ {
		maxVal := outputData[i*sizeOfAxis]
		for j := 1; j < sizeOfAxis; j++ {
			if outputData[i*sizeOfAxis+j] > maxVal {
				maxVal = outputData[i*sizeOfAxis+j]
			}
		}

		sumExp := 0.0
		for j := 0; j < sizeOfAxis; j++ {
			outputData[i*sizeOfAxis+j] = math.Exp(outputData[i*sizeOfAxis+j] - maxVal)
			sumExp += outputData[i*sizeOfAxis+j]
		}

		if sumExp == 0 {
			for j := 0; j < sizeOfAxis; j++ {
				outputData[i*sizeOfAxis+j] = 0.0
			}
		} else {
			for j := 0; j < sizeOfAxis; j++ {
				outputData[i*sizeOfAxis+j] /= sumExp
			}
		}
	}

	op.output = NewTensor(outputData, op.input.Shape, op.input.requiresGrad)
	if op.output.requiresGrad {
		op.output.creator = op
	}
	return op.output, nil
}

func (op *softmaxOperation) Inputs() []*Tensor {
	return []*Tensor{op.input}
}

// Backward performs the backward pass for the softmax operation.
func (op *softmaxOperation) Backward(grad *Tensor) error {
	if grad == nil || grad.Data == nil {
		return nil
	}

	if op.input.requiresGrad {
		if op.input.Grad == nil {
			op.input.Grad = NewTensor(make([]float64, len(op.input.Data)), op.input.Shape, false)
		}

		// Gradient of softmax: dL/dx_i = sum_j (dL/dy_j * dy_j/dx_i)
		// dy_j/dx_i = y_j * (delta_ij - y_i)
		// dL/dx_i = sum_j (dL/dy_j * y_j * (delta_ij - y_i))
		// dL/dx_i = dL/dy_i * y_i - y_i * sum_j (dL/dy_j * y_j)

		// For each vector (batch item, sequence position)
		sizeOfAxis := op.input.Shape[op.axis]
		numVectors := len(op.input.Data) / sizeOfAxis

		for i := 0; i < numVectors; i++ {
			// Calculate sum_j (dL/dy_j * y_j) for the current vector
			sum_dL_dy_y := 0.0
			for j := 0; j < sizeOfAxis; j++ {
				outputFlatIndex := i*sizeOfAxis + j
				sum_dL_dy_y += grad.Data[outputFlatIndex] * op.output.Data[outputFlatIndex]
			}

			// Calculate dL/dx_k for each element in the current vector
			for k := 0; k < sizeOfAxis; k++ {
				outputFlatIndex := i*sizeOfAxis + k
				dL_dy_k := grad.Data[outputFlatIndex]
				y_k := op.output.Data[outputFlatIndex]

				dL_dx_k := y_k * (dL_dy_k - sum_dL_dy_y)
				op.input.Grad.Data[outputFlatIndex] += dL_dx_k
			}
		}
	}
	return nil
}

// Slice extracts a slice from a tensor along a specified dimension.
func (t *Tensor) Slice(axis, start, end int) (*Tensor, error) {
	if axis < 0 || axis >= len(t.Shape) {
		return nil, fmt.Errorf("invalid axis for slicing: %d", axis)
	}
	if start < 0 || end > t.Shape[axis] || start > end {
		return nil, fmt.Errorf("invalid slice indices: start %d, end %d, for shape %v", start, end, t.Shape)
	}

	newShape := make([]int, len(t.Shape))
	copy(newShape, t.Shape)
	newShape[axis] = end - start

	// This is a simplified implementation and will not work for all cases.
	// A more robust implementation would require more complex index calculations.
	stride := 1
	for i := axis + 1; i < len(t.Shape); i++ {
		stride *= t.Shape[i]
	}

	newData := t.Data[start*stride : end*stride]

	return NewTensor(newData, newShape, t.requiresGrad), nil
}
