package tensor

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"log"
	"math"
	"runtime"
	"sync"
)

func init() {
	log.SetFlags(0)
}

// Next moves the iterator to the next element.
func (it *MatMulIterator) Next() bool {
	it.currentFlatIndex++
	return it.currentFlatIndex < it.resultRows*it.resultCols
}

// Current returns the current flat index, row index (i), and column index (j).
func (it *MatMulIterator) Current() (flatIndex, i, j int) {
	flatIndex = it.currentFlatIndex
	i = flatIndex / it.resultCols
	j = flatIndex % it.resultCols
	return flatIndex, i, j
}

// MatMul4DIterator is an iterator for traversing the result tensor of a 4D batched matrix multiplication.
// It provides the current indices (batch, head, row, col) for the result tensor.
type MatMul4DIterator struct {
	batchSize  int
	numHeads   int
	resultRows int // Corresponds to dim2_t (rows in the output 2D slice)
	resultCols int // Corresponds to dim3_other (columns in the output 2D slice)

	chunkSizeRows int // Size of the chunk in the row dimension
	chunkSizeCols int // Size of the chunk in the column dimension

	currentB int // Current batch index
	currentH int // Current head index
	currentI int // Current starting row index of the chunk in the 2D slice
	currentJ int // Current starting column index of the chunk in the 2D slice
}

// NewMatMul4DIterator creates a new MatMul4DIterator with initial indices.
func NewMatMul4DIterator(batchSize, numHeads, resultRows, resultCols int) *MatMul4DIterator {
	return &MatMul4DIterator{
		batchSize:  batchSize,
		numHeads:   numHeads,
		resultRows: resultRows,
		resultCols: resultCols,
		// Define a default chunk size
		chunkSizeRows: 32, // Example chunk size
		chunkSizeCols: 32, // Example chunk size

		// Initialize indices to start before the first element
		currentB: 0,
		currentH: 0,
		currentI: 0,
		currentJ: -1, // Start column before 0 so the first Next() call starts at (0, 0, 0, 0)
	}
}

// Next updates the iterator's internal indices to move to the next element
// in the 4D result tensor. It returns true if there are more elements, false otherwise.
func (it *MatMul4DIterator) Next() bool {
	it.currentJ += it.chunkSizeCols

	// Check if we've reached the end of the current row chunk block
	if it.currentJ >= it.resultCols {
		it.currentJ = 0                 // Reset column chunk index
		it.currentI += it.chunkSizeRows // Move to the next row chunk

		// Check if we've reached the end of the current 2D slice (for a batch and head)
		if it.currentI >= it.resultRows {
			it.currentI = 0 // Reset row chunk index
			it.currentH++   // Move to the next head

			// Check if we've reached the end of the current batch
			if it.currentH >= it.numHeads {
				it.currentH = 0 // Reset head index
				it.currentB++   // Move to the next batch
			}
		}
	}

	// Check if we are still within the batch size
	return it.currentB < it.batchSize
}

// Operation represents an operation in the computation graph.
type Operation interface {
	Inputs() []*Tensor
	Backward(grad *Tensor) error
}

// Tensor represents a multi-dimensional array of float64 values.
type Tensor struct {
	Data         []float64
	Shape        []int
	Grad         *Tensor `gob:"-"` // Exclude Grad from gob serialization
	Mask         *Tensor
	Creator      Operation `gob:"-"` // Exclude Creator from gob serialization
	RequiresGrad bool
	Operation    Operation `gob:"-"` // Exclude Operation from gob serialization
}

// GobEncode implements the gob.GobEncoder interface.
func (t *Tensor) GobEncode() ([]byte, error) {
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)

	if err := enc.Encode(t.Data); err != nil {
		return nil, err
	}
	if err := enc.Encode(t.Shape); err != nil {
		return nil, err
	}

	// Explicitly handle nil for Mask
	maskIsNil := t.Mask == nil
	if err := enc.Encode(maskIsNil); err != nil {
		return nil, err
	}
	if !maskIsNil {
		if err := enc.Encode(t.Mask); err != nil {
			return nil, err
		}
	}

	if err := enc.Encode(t.RequiresGrad); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// GobDecode implements the gob.GobDecoder interface.
func (t *Tensor) GobDecode(data []byte) error {
	buf := bytes.NewBuffer(data)
	dec := gob.NewDecoder(buf)

	if err := dec.Decode(&t.Data); err != nil {
		return err
	}
	if err := dec.Decode(&t.Shape); err != nil {
		return err
	}

	// Explicitly handle nil for Mask
	var maskIsNil bool
	if err := dec.Decode(&maskIsNil); err != nil {
		return err
	}
	if !maskIsNil {
		t.Mask = &Tensor{} // Initialize Mask before decoding into it
		if err := dec.Decode(t.Mask); err != nil {
			return err
		}
	} else {
		t.Mask = nil // Ensure Mask is nil if it was nil during encoding
	}

	if err := dec.Decode(&t.RequiresGrad); err != nil {
		return err
	}
	return nil
}

// NewTensor creates a new Tensor with the given shape and optional data.
func NewTensor(shape []int, data []float64, requiresGrad bool) *Tensor {
	if data == nil {
		size := 1
		for _, dim := range shape {
			size *= dim
		}
		data = make([]float64, size)
	}
	return &Tensor{
		Data:         data,
		Shape:        shape,
		RequiresGrad: requiresGrad,
	}
}

// Clone creates a deep copy of the tensor.
func (t *Tensor) Clone() *Tensor {
	newData := make([]float64, len(t.Data))
	copy(newData, t.Data)
	newShape := make([]int, len(t.Shape))
	copy(newShape, t.Shape)

	// The new tensor should not share gradient information or creator
	return &Tensor{
		Data:         newData,
		Shape:        newShape,
		RequiresGrad: t.RequiresGrad, // The clone's grad requirement should be the same
		Grad:         nil,            // Do not copy gradient
		Creator:      nil,            // The clone is a new leaf in the graph
	}
}

// ZeroGrad resets the gradient of the tensor to zeros.
func (t *Tensor) ZeroGrad() {
	if t.RequiresGrad {
		if t.Grad == nil {
			t.Grad = NewTensor(t.Shape, make([]float64, len(t.Data)), false)
		} else {
			for i := range t.Grad.Data {
				t.Grad.Data[i] = 0
			}
		}
	}
}

// Inputs returns the input tensors to this operation.
// For a Tensor itself, it typically has no inputs in the context of an operation,
// unless it's acting as a placeholder for an input node in a graph.
func (t *Tensor) Inputs() []*Tensor {
	return []*Tensor{}
}

// MatMulIterator is an iterator for the result tensor of a matrix multiplication.
// It provides the flat index, row index (i), and column index (j) for each element
// in the result tensor.
type MatMulIterator struct {
	resultRows int
	resultCols int

	currentFlatIndex int
}

func NewMatMulIterator(resultRows, resultCols int) *MatMulIterator {
	return &MatMulIterator{
		resultRows:       resultRows,
		resultCols:       resultCols,
		currentFlatIndex: -1, // Start before the first element
	}
}

// Current returns the current flat index, row index (i), and column index (j).
func (it *MatMul4DIterator) Current() (b, h, i, j int) {
	b = it.currentB
	h = it.currentH
	i = it.currentI
	j = it.currentJ
	return b, h, i, j
}

// CurrentChunkSize returns the size of the current chunk (rows, cols).
// It handles cases where the last chunk in a dimension might be smaller.
func (it *MatMul4DIterator) CurrentChunkSize() (rows, cols int) {

	rows = it.chunkSizeRows
	cols = it.chunkSizeCols

	if it.currentI+rows > it.resultRows {
		rows = it.resultRows - it.currentI
	}
	if it.currentJ+cols > it.resultCols {
		cols = it.resultCols - it.currentJ
	}
	return rows, cols
}

// Next moves the iterator to the next batch/head combination.
// It returns true if there are more elements, false otherwise.
func (it *BatchHeadIterator) Next() bool {
	it.currentH++
	if it.currentH >= it.numHeads {
		it.currentH = 0
		it.currentB++
	}
	return it.currentB < it.batchSize
}

// Current returns the current batch and head indices.
func (it *BatchHeadIterator) Current() (b, h int) {
	return it.currentB, it.currentH
}

// RowIterator iterates over rows of a 2D tensor.
type RowIterator struct {
	rows       int
	cols       int
	currentRow int
}

func NewRowIterator(rows, cols int) *RowIterator {
	return &RowIterator{

		rows: rows,

		cols:       cols,
		currentRow: -1, // Start before the first row
	}
}

// Add performs element-wise addition of two tensors.
func (t *Tensor) Add(other *Tensor) (*Tensor, error) {
	// Check if shapes are compatible for element-wise addition (must be identical for now)
	if !compareShapes(t.Shape, other.Shape) {
		return nil, fmt.Errorf("mismatched shapes for Add operation: %v and %v", t.Shape, other.Shape)
	}

	resultData := make([]float64, len(t.Data))
	for i := range t.Data {
		resultData[i] = t.Data[i] + other.Data[i]
	}

	resultTensor := NewTensor(t.Shape, resultData, t.RequiresGrad || other.RequiresGrad)

	// If either input requires gradient, set up the backward pass
	if resultTensor.RequiresGrad {
		resultTensor.Creator = &AddOperation{t, other}
	}

	return resultTensor, nil
}

// addOperation represents the addition operation for backward pass.
type AddOperation struct {
	A *Tensor
	B *Tensor
}

func (op *AddOperation) Inputs() []*Tensor {
	return []*Tensor{op.A, op.B}
}

func (op *AddOperation) Backward(grad *Tensor) error {
	// Gradient of addition is simply passing the gradient through to both inputs.
	// Assuming grad has the same shape as the output of the Add operation.

	if op.A.RequiresGrad {
		if op.A.Grad == nil {
			op.A.Grad = NewTensor(op.A.Shape, make([]float64, len(op.A.Data)), false)
		}
		for i := range grad.Data {
			op.A.Grad.Data[i] += grad.Data[i]
		}
	}

	if op.B.RequiresGrad {
		if op.B.Grad == nil {
			op.B.Grad = NewTensor(op.B.Shape, make([]float64, len(op.B.Data)), false)
		}
		for i := range grad.Data {
			op.B.Grad.Data[i] += grad.Data[i]
		}
	}

	return nil
}

// compareShapes is a helper function to compare two shapes.
func compareShapes(s1, s2 []int) bool {
	if len(s1) != len(s2) {
		return false
	}
	for i := range s1 {
		if s1[i] != s2[i] {
			return false
		}
	}
	return true
}

// MatMul performs matrix multiplication with another Tensor.
// It supports 2D matrix multiplication and batched 4D matrix multiplication (batch, heads, rows, cols).
func (t *Tensor) MatMul(other *Tensor) (*Tensor, error) {
	// Case 1: 2D matrix multiplication
	if len(t.Shape) == 2 && len(other.Shape) == 2 {
		if t.Shape[1] != other.Shape[0] {
			return nil, fmt.Errorf("incompatible shapes for 2D matrix multiplication: %v and %v", t.Shape, other.Shape)
		}

		rowsA := t.Shape[0]
		colsA := t.Shape[1]
		colsB := other.Shape[1]

		resultRows := rowsA
		resultCols := colsB
		resultData := make([]float64, resultRows*resultCols)

		var wg sync.WaitGroup
		numWorkers := runtime.NumCPU() // Get number of CPU cores
		if numWorkers == 0 {
			numWorkers = 1 // Ensure at least one worker
		}

		rowsPerWorker := (resultRows + numWorkers - 1) / numWorkers

		for i := 0; i < numWorkers; i++ {
			startRow := i * rowsPerWorker
			endRow := (i + 1) * rowsPerWorker
			if endRow > resultRows {
				endRow = resultRows
			}

			wg.Add(1)
			go func(start, end int) {
				defer wg.Done()
				for i := start; i < end; i++ {
					for j := 0; j < resultCols; j++ {
						sum := 0.0
						for k := 0; k < colsA; k++ {
							sum += t.Data[i*colsA+k] * other.Data[k*colsB+j]
						}
						resultData[i*resultCols+j] = sum
					}
				}
			}(startRow, endRow)
		}
		wg.Wait()

		resultTensor := NewTensor([]int{resultRows, resultCols}, resultData, t.RequiresGrad || other.RequiresGrad)
		if resultTensor.RequiresGrad {
			resultTensor.Creator = &MatmulOperation{t, other}
		}
		return resultTensor, nil
	}

	// Case 2: Batched 4D matrix multiplication (batch, heads, rows, cols)
	if len(t.Shape) == 4 && len(other.Shape) == 4 {
		if t.Shape[0] != other.Shape[0] || // batch size must match
			t.Shape[1] != other.Shape[1] || // num heads must match
			t.Shape[3] != other.Shape[2] { // inner dimensions must match for multiplication
			return nil, fmt.Errorf("incompatible shapes for 4D batched matrix multiplication: %v and %v", t.Shape, other.Shape)
		}

		batchSize := t.Shape[0]
		numHeads := t.Shape[1]

		rowsA := t.Shape[2]
		colsA := t.Shape[3]

		rowsB := other.Shape[2]
		colsB := other.Shape[3]

		resultRows := rowsA
		resultCols := colsB
		resultData := make([]float64, batchSize*numHeads*resultRows*resultCols)
		resultShape := []int{batchSize, numHeads, resultRows, resultCols}

		// Iterate over batches and heads
		for b := 0; b < batchSize; b++ {
			for h := 0; h < numHeads; h++ {
				// Extract 2D slices for current batch and head
				tOffset := b*numHeads*rowsA*colsA + h*rowsA*colsA
				otherOffset := b*numHeads*rowsB*colsB + h*rowsB*colsB
				resultOffset := b*numHeads*resultRows*resultCols + h*resultRows*resultCols

				for i := 0; i < resultRows; i++ {
					for j := 0; j < resultCols; j++ {
						sum := 0.0
						for k := 0; k < colsA; k++ {
							sum += t.Data[tOffset+i*colsA+k] * other.Data[otherOffset+k*colsB+j]
						}
						resultData[resultOffset+i*resultCols+j] = sum
					}
				}
			}
		}

		resultTensor := NewTensor(resultShape, resultData, t.RequiresGrad || other.RequiresGrad)
		if resultTensor.RequiresGrad {
			resultTensor.Creator = &MatmulOperation{t, other}
		}
		return resultTensor, nil
	}

	return nil, fmt.Errorf("MatMul operation currently only supports 2D or 4D tensors. Got %v and %v", t.Shape, other.Shape)
}

// MatmulOperation represents the matrix multiplication operation for backward pass.
type MatmulOperation struct {
	A *Tensor
	B *Tensor
}

func (op *MatmulOperation) Inputs() []*Tensor {
	return []*Tensor{op.A, op.B}
}

func (op *MatmulOperation) Backward(grad *Tensor) error {
	// Handle 2D case
	if len(op.A.Shape) == 2 {
		// dL/dA = grad * B^T
		if op.A.RequiresGrad {
			bt, err := op.B.Transpose(0, 1)
			if err != nil {
				return fmt.Errorf("MatMul backward (2D): failed to transpose B: %w", err)
			}
			gradA, err := grad.MatMul(bt)
			if err != nil {
				return fmt.Errorf("MatMul backward (2D): failed to calculate gradA: %w", err)
			}
			if op.A.Grad == nil {
				op.A.Grad = NewTensor(op.A.Shape, make([]float64, len(op.A.Data)), false)
			}
			for i := range gradA.Data {
				op.A.Grad.Data[i] += gradA.Data[i]
			}
		}

		// dL/dB = A^T * grad
		if op.B.RequiresGrad {
			at, err := op.A.Transpose(0, 1)
			if err != nil {
				return fmt.Errorf("MatMul backward (2D): failed to transpose A: %w", err)
			}
			gradB, err := at.MatMul(grad)
			if err != nil {
				return fmt.Errorf("MatMul backward (2D): failed to calculate gradB: %w", err)
			}
			if op.B.Grad == nil {
				op.B.Grad = NewTensor(op.B.Shape, make([]float64, len(op.B.Data)), false)
			}
			for i := range gradB.Data {
				op.B.Grad.Data[i] += gradB.Data[i]
			}
		}
		return nil
	}

	// Handle 4D case
	if len(op.A.Shape) == 4 {
		// dL/dA = grad * B^T
		if op.A.RequiresGrad {
			bt, err := op.B.Transpose(2, 3) // Transpose inner matrices
			if err != nil {
				return fmt.Errorf("MatMul backward (4D): failed to transpose B: %w", err)
			}
			gradA, err := grad.MatMul(bt)
			if err != nil {
				return fmt.Errorf("MatMul backward (4D): failed to calculate gradA: %w", err)
			}
			if op.A.Grad == nil {
				op.A.Grad = NewTensor(op.A.Shape, make([]float64, len(op.A.Data)), false)
			}
			for i := range gradA.Data {
				op.A.Grad.Data[i] += gradA.Data[i]
			}
		}

		// dL/dB = A^T * grad
		if op.B.RequiresGrad {
			at, err := op.A.Transpose(2, 3) // Transpose inner matrices
			if err != nil {
				return fmt.Errorf("MatMul backward (4D): failed to transpose A: %w", err)
			}
			gradB, err := at.MatMul(grad)
			if err != nil {
				return fmt.Errorf("MatMul backward (4D): failed to calculate gradB: %w", err)
			}
			if op.B.Grad == nil {
				op.B.Grad = NewTensor(op.B.Shape, make([]float64, len(op.B.Data)), false)
			}
			for i := range gradB.Data {
				op.B.Grad.Data[i] += gradB.Data[i]
			}
		}
		return nil
	}

	return fmt.Errorf("MatMul backward only supports 2D or 4D tensors, got %d dimensions", len(op.A.Shape))
}

// Transpose transposes a tensor by swapping two specified axes.
func (t *Tensor) Transpose(axis1, axis2 int) (*Tensor, error) {
	if axis1 < 0 || axis1 >= len(t.Shape) || axis2 < 0 || axis2 >= len(t.Shape) {
		return nil, fmt.Errorf("axes out of bounds for transpose: %d, %d for shape %v", axis1, axis2, t.Shape)
	}
	if axis1 == axis2 {
		return t, nil // No change if axes are the same
	}

	newShape := make([]int, len(t.Shape))
	copy(newShape, t.Shape)
	newShape[axis1], newShape[axis2] = newShape[axis2], newShape[axis1]

	newData := make([]float64, len(t.Data))

	// Create strides for original and new shape
	oldStrides := make([]int, len(t.Shape))
	newStrides := make([]int, len(t.Shape))
	stride := 1
	for i := len(t.Shape) - 1; i >= 0; i-- {
		oldStrides[i] = stride
		stride *= t.Shape[i]
	}
	stride = 1
	for i := len(newShape) - 1; i >= 0; i-- {
		newStrides[i] = stride
		stride *= newShape[i]
	}

	// Iterate through all elements and map them to the new positions
	totalElements := 1
	for _, dim := range t.Shape {
		totalElements *= dim
	}

	oldCoords := make([]int, len(t.Shape))
	newCoords := make([]int, len(t.Shape))

	for i := 0; i < totalElements; i++ {
		// Convert flat index to N-dimensional coordinates for original tensor
		tempIdx := i
		for dim := 0; dim < len(t.Shape); dim++ {
			oldCoords[dim] = tempIdx / oldStrides[dim]
			tempIdx %= oldStrides[dim]
		}

		// Swap axes for new coordinates
		copy(newCoords, oldCoords)
		newCoords[axis1], newCoords[axis2] = newCoords[axis2], newCoords[axis1]

		// Convert new N-dimensional coordinates to flat index for new tensor
		newFlatIndex := 0
		for dim := 0; dim < len(newShape); dim++ {
			newFlatIndex += newCoords[dim] * newStrides[dim]
		}
		newData[newFlatIndex] = t.Data[i]
	}

	return NewTensor(newShape, newData, false), nil
}

func (t *Tensor) AddWithBroadcast(other *Tensor) (*Tensor, error) {
	// Determine the output shape after broadcasting
	maxDims := max(len(t.Shape), len(other.Shape))
	resultShape := make([]int, maxDims)

	// Pad shapes with 1s on the left to make them the same length
	paddedTShape := make([]int, maxDims)
	copy(paddedTShape[maxDims-len(t.Shape):], t.Shape)
	for i := 0; i < maxDims-len(t.Shape); i++ {
		paddedTShape[i] = 1
	}

	paddedOtherShape := make([]int, maxDims)
	copy(paddedOtherShape[maxDims-len(other.Shape):], other.Shape)
	for i := 0; i < maxDims-len(other.Shape); i++ {
		paddedOtherShape[i] = 1
	}

	for i := 0; i < maxDims; i++ {
		dimT := paddedTShape[i]
		dimOther := paddedOtherShape[i]

		if dimT != dimOther && dimT != 1 && dimOther != 1 {
			return nil, fmt.Errorf("unsupported shapes for AddWithBroadcast operation: %v and %v (dimension mismatch at index %d: %d vs %d)", t.Shape, other.Shape, i, dimT, dimOther)
		}
		resultShape[i] = max(dimT, dimOther)
	}

	// Calculate strides for efficient indexing
	stridesT := calculateStrides(paddedTShape)
	stridesOther := calculateStrides(paddedOtherShape)
	stridesResult := calculateStrides(resultShape)

	resultSize := 1
	for _, dim := range resultShape {
		resultSize *= dim
	}
	resultData := make([]float64, resultSize)

	// Perform element-wise addition with broadcasting
	for i := 0; i < resultSize; i++ {
		coords := getCoords(i, resultShape, stridesResult)

		// Get indices for t
		idxT := 0
		for dim := 0; dim < maxDims; dim++ {
			if paddedTShape[dim] != 1 {
				idxT += coords[dim] * stridesT[dim]
			}
		}

		// Get indices for other
		idxOther := 0
		for dim := 0; dim < maxDims; dim++ {
			if paddedOtherShape[dim] != 1 {
				idxOther += coords[dim] * stridesOther[dim]
			}
		}
		resultData[i] = t.Data[idxT] + other.Data[idxOther]
	}

	resultTensor := NewTensor(resultShape, resultData, t.RequiresGrad || other.RequiresGrad)

	if resultTensor.RequiresGrad {
		resultTensor.Creator = &AddWithBroadcastOperation{t, other}
	}

	return resultTensor, nil
}

type AddWithBroadcastOperation struct {
	A *Tensor
	B *Tensor
}

func (op *AddWithBroadcastOperation) Inputs() []*Tensor {
	return []*Tensor{op.A, op.B}
}

func (op *AddWithBroadcastOperation) Backward(grad *Tensor) error {
	if op.A.RequiresGrad {
		if op.A.Grad == nil {
			op.A.Grad = NewTensor(op.A.Shape, make([]float64, len(op.A.Data)), false)
		}
		gradA := sumTo(grad, op.A.Shape)
		for i := range op.A.Grad.Data {
			op.A.Grad.Data[i] += gradA.Data[i]
		}
	}
	if op.B.RequiresGrad {
		if op.B.Grad == nil {
			op.B.Grad = NewTensor(op.B.Shape, make([]float64, len(op.B.Data)), false)
		}
		gradB := sumTo(grad, op.B.Shape)
		for i := range op.B.Grad.Data {
			op.B.Grad.Data[i] += gradB.Data[i]
		}
	}
	return nil
}

func sumTo(grad *Tensor, shape []int) *Tensor {
	if compareShapes(grad.Shape, shape) {
		return grad
	}

	n_dim_grad := len(grad.Shape)
	n_dim_shape := len(shape)

	// Identify axes to sum over
	sum_axes := []int{}
	for i := 0; i < n_dim_grad-n_dim_shape; i++ {
		sum_axes = append(sum_axes, i)
	}

	padded_shape := make([]int, n_dim_grad)
	copy(padded_shape[n_dim_grad-n_dim_shape:], shape)
	for i := 0; i < n_dim_grad-n_dim_shape; i++ {
		padded_shape[i] = 1
	}

	for i := n_dim_grad - n_dim_shape; i < n_dim_grad; i++ {
		if grad.Shape[i] != padded_shape[i] && padded_shape[i] == 1 {
			sum_axes = append(sum_axes, i)
		}
	}

	// Sum along the identified axes
	summed_grad := grad
	for i := len(sum_axes) - 1; i >= 0; i-- {
		summed_grad = sumAlongAxis(summed_grad, sum_axes[i])
	}

	// Reshape to the target shape
	reshaped_grad, _ := summed_grad.Reshape(shape)
	return reshaped_grad
}

func sumAlongAxis(t *Tensor, axis int) *Tensor {
	newShape := []int{}
	for i, dim := range t.Shape {
		if i != axis {
			newShape = append(newShape, dim)
		}
	}

	newSize := 1
	for _, dim := range newShape {
		newSize *= dim
	}
	newData := make([]float64, newSize)

	strides := calculateStrides(t.Shape)
	newStrides := calculateStrides(newShape)

	for i := 0; i < len(t.Data); i++ {
		oldCoords := getCoords(i, t.Shape, strides)

		newCoords := make([]int, 0, len(newShape))
		for j, coord := range oldCoords {
			if j != axis {
				newCoords = append(newCoords, coord)
			}
		}

		newIndex := 0
		if len(newCoords) > 0 {
			for j, coord := range newCoords {
				newIndex += coord * newStrides[j]
			}
		}

		if newIndex < len(newData) {
			newData[newIndex] += t.Data[i]
		}
	}

	return NewTensor(newShape, newData, false)
}

// Helper function to calculate strides
func calculateStrides(shape []int) []int {
	strides := make([]int, len(shape))
	stride := 1
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}
	return strides
}

// Helper function to get N-dimensional coordinates from a flat index
func getCoords(flatIndex int, shape, strides []int) []int {
	coords := make([]int, len(shape))
	tempIdx := flatIndex
	for dim := 0; dim < len(shape); dim++ {
		coords[dim] = tempIdx / strides[dim]
		tempIdx %= strides[dim]
	}
	return coords
}

// Helper function for max
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// Reshape returns a new Tensor with the same data but a new shape.
// It returns an error if the new shape is incompatible (total number of elements mismatch).
func (t *Tensor) Reshape(newShape []int) (*Tensor, error) {
	newSize := 1
	for _, dim := range newShape {
		newSize *= dim
	}

	if newSize != len(t.Data) {
		return nil, fmt.Errorf("cannot reshape tensor from %v to %v: total number of elements mismatch (%d vs %d)", t.Shape, newShape, len(t.Data), newSize)
	}

	// Create a new tensor with the new shape, sharing the underlying data array.
	// This is efficient as it avoids copying large amounts of data.
	resultTensor := &Tensor{
		Data:         t.Data, // Share the underlying data array
		Shape:        newShape,
		RequiresGrad: t.RequiresGrad,
	}
	if resultTensor.RequiresGrad {
		resultTensor.Creator = &ReshapeOperation{t, t.Shape}
	}
	return resultTensor, nil
}

type ReshapeOperation struct {
	Input    *Tensor
	OldShape []int
}

func (op *ReshapeOperation) Inputs() []*Tensor {
	return []*Tensor{op.Input}
}

func (op *ReshapeOperation) Backward(grad *Tensor) error {
	if op.Input.RequiresGrad {
		if op.Input.Grad == nil {
			op.Input.Grad = NewTensor(op.Input.Shape, make([]float64, len(op.Input.Data)), false)
		}
		// The gradient of reshape is just reshaping the gradient back.
		// The data is the same, so we can just add the gradients.
		for i := range grad.Data {
			op.Input.Grad.Data[i] += grad.Data[i]
		}
	}
	return nil
}

// Softmax applies the softmax function along a specified axis.
func (t *Tensor) Softmax(axis int) (*Tensor, error) {
	if axis < 0 {
		axis = len(t.Shape) + axis
	}
	if axis < 0 || axis >= len(t.Shape) {
		return nil, fmt.Errorf("axis %d out of bounds for tensor with shape %v", axis, t.Shape)
	}

	outputData := make([]float64, len(t.Data))
	outputTensor := NewTensor(t.Shape, outputData, t.RequiresGrad)

	if outputTensor.RequiresGrad {
		outputTensor.Creator = &SoftmaxOperation{t, outputTensor, axis}
	}

	// Calculate the size of the dimension along which softmax is applied
	axisDimSize := t.Shape[axis]

	// Calculate the number of elements before and after the axis
	outerSize := 1
	for i := 0; i < axis; i++ {
		outerSize *= t.Shape[i]
	}
	innerSize := 1
	for i := axis + 1; i < len(t.Shape); i++ {
		innerSize *= t.Shape[i]
	}

	// Iterate through the tensor to apply softmax
	for i := 0; i < outerSize; i++ {
		for j := 0; j < innerSize; j++ {
			// Find the maximum value in the slice along the axis for numerical stability
			maxVal := -math.MaxFloat64
			for k := 0; k < axisDimSize; k++ {
				idx := i*axisDimSize*innerSize + k*innerSize + j
				if t.Data[idx] > maxVal {
					maxVal = t.Data[idx]
				}
			}

			sumExp := 0.0
			for k := 0; k < axisDimSize; k++ {
				idx := i*axisDimSize*innerSize + k*innerSize + j
				expVal := math.Exp(t.Data[idx] - maxVal) // Subtract max for stability
				outputData[idx] = expVal
				sumExp += expVal
			}

			// Normalize by the sum
			for k := 0; k < axisDimSize; k++ {
				idx := i*axisDimSize*innerSize + k*innerSize + j
				outputData[idx] /= sumExp
			}
		}
	}

	return outputTensor, nil
}

// SoftmaxOperation represents the softmax operation for backward pass.
type SoftmaxOperation struct {
	Input  *Tensor
	Output *Tensor
	Axis   int
}

func (op *SoftmaxOperation) Inputs() []*Tensor {
	return []*Tensor{op.Input}
}

func (op *SoftmaxOperation) Backward(grad *Tensor) error {
	if !op.Input.RequiresGrad {
		return nil
	}
	if op.Input.Grad == nil {
		op.Input.Grad = NewTensor(op.Input.Shape, make([]float64, len(op.Input.Data)), false)
	}

	// This backward pass is simplified and assumes softmax is applied along the last dimension
	// of a 2D or 4D tensor, which is common in transformers.
	resolvedAxis := op.Axis
	if resolvedAxis < 0 {
		resolvedAxis = len(op.Input.Shape) + resolvedAxis
	}

	axisDimSize := op.Input.Shape[resolvedAxis]
	outerSize := 1
	for i := 0; i < resolvedAxis; i++ {
		outerSize *= op.Input.Shape[i]
	}
	innerSize := 1
	for i := resolvedAxis + 1; i < len(op.Input.Shape); i++ {
		innerSize *= op.Input.Shape[i]
	}

	for i := 0; i < outerSize; i++ {
		for j := 0; j < innerSize; j++ {
			// Calculate sum_k(dL/dy_k * y_k) for the current slice
			sum_dL_dy_y := 0.0
			for k := 0; k < axisDimSize; k++ {
				idx := i*axisDimSize*innerSize + k*innerSize + j
				sum_dL_dy_y += grad.Data[idx] * op.Output.Data[idx]
			}

			// Calculate dL/dx_i for the current slice
			for k := 0; k < axisDimSize; k++ {
				idx := i*axisDimSize*innerSize + k*innerSize + j
				op.Input.Grad.Data[idx] += op.Output.Data[idx] * (grad.Data[idx] - sum_dL_dy_y)
			}
		}
	}

	if !op.Input.RequiresGrad {
		return nil
	}
	if op.Input.Grad == nil {
		op.Input.Grad = NewTensor(op.Input.Shape, make([]float64, len(op.Input.Data)), false)
	}

	// This backward pass is simplified and assumes softmax is applied along the last dimension
	// of a 2D or 4D tensor, which is common in transformers.
	resolvedAxis = op.Axis
	if resolvedAxis < 0 {
		resolvedAxis = len(op.Input.Shape) + resolvedAxis
	}

	axisDimSize = op.Input.Shape[resolvedAxis]
	outerSize = 1
	for i := 0; i < resolvedAxis; i++ {
		outerSize *= op.Input.Shape[i]
	}
	innerSize = 1
	for i := resolvedAxis + 1; i < len(op.Input.Shape); i++ {
		innerSize *= op.Input.Shape[i]
	}

	for i := 0; i < outerSize; i++ {
		for j := 0; j < innerSize; j++ {
			// Calculate sum_k(dL/dy_k * y_k) for the current slice
			sum_dL_dy_y := 0.0
			for k := 0; k < axisDimSize; k++ {
				idx := i*axisDimSize*innerSize + k*innerSize + j
				sum_dL_dy_y += grad.Data[idx] * op.Output.Data[idx]
			}

			// Calculate dL/dx_i for the current slice
			for k := 0; k < axisDimSize; k++ {
				idx := i*axisDimSize*innerSize + k*innerSize + j
				op.Input.Grad.Data[idx] += op.Output.Data[idx] * (grad.Data[idx] - sum_dL_dy_y)
			}
		}
	}

	return nil
}

// Backward performs backpropagation starting from this tensor.
func (t *Tensor) Backward(grad *Tensor) error {
	topo := []*Tensor{}
	visited := map[*Tensor]bool{}
	stack := []*Tensor{}

	stack = append(stack, t)

	for len(stack) > 0 {
		v := stack[len(stack)-1]
		stack = stack[:len(stack)-1]

		if v == nil || visited[v] {
			continue
		}
		visited[v] = true

		if v.Creator != nil {
			for _, child := range v.Creator.Inputs() {
				if !visited[child] {
					stack = append(stack, child)
				}
			}
		}
		topo = append(topo, v)
	}

	// The topological sort needs to be in the correct order for backpropagation.
	// The iterative DFS above builds it in reverse order, so we need to reverse it.
	for i, j := 0, len(topo)-1; i < j; i, j = i+1, j-1 {
		topo[i], topo[j] = topo[j], topo[i]
	}

	// Initialize gradients
	for _, v := range topo {
		if v.Grad == nil {
			v.Grad = NewTensor(v.Shape, make([]float64, len(v.Data)), false)
		}
	}

	// Seed the gradient of the output tensor
	if t.Grad == nil {
		t.Grad = NewTensor(t.Shape, make([]float64, len(t.Data)), false)
	}
	for i := range grad.Data {
		t.Grad.Data[i] = grad.Data[i]
	}

	for i := len(topo) - 1; i >= 0; i-- {
		v := topo[i]
		if v.Creator != nil {

			err := v.Creator.Backward(v.Grad)
			if err != nil {
				return fmt.Errorf("error during backward pass for tensor with shape %v: %w", v.Shape, err)
			}
		}
	}
	return nil
}

// BatchHeadIterator iterates over the batch and head dimensions of a 4D tensor.
type BatchHeadIterator struct {
	batchSize int
	numHeads  int
	currentB  int
	currentH  int
}

// NewBatchHeadIterator creates a new iterator for 4D tensor batches and heads.
func NewBatchHeadIterator(batchSize, numHeads int) *BatchHeadIterator {
	return &BatchHeadIterator{
		batchSize: batchSize,
		numHeads:  numHeads,
		currentB:  0,
		currentH:  -1, // Start before the first head to make the first call to Next() correct.
	}
}

// Slice returns a new Tensor representing a slice of the original tensor along a given axis.
// It takes the axis, start index (inclusive), and end index (exclusive) for the slice.
func (t *Tensor) Slice(axis, start, end int) (*Tensor, error) {
	if axis < 0 || axis >= len(t.Shape) {
		return nil, fmt.Errorf("axis %d out of bounds for tensor with shape %v", axis, t.Shape)
	}
	if start < 0 || end > t.Shape[axis] || start > end {
		return nil, fmt.Errorf("invalid slice indices for axis %d: start %d, end %d for dimension size %d", axis, start, end, t.Shape[axis])
	}

	// Calculate the new shape
	newShape := make([]int, len(t.Shape))
	copy(newShape, t.Shape)
	newShape[axis] = end - start

	newTensor := NewTensor(newShape, nil, t.RequiresGrad)
	if newTensor.RequiresGrad {
		newTensor.Creator = &SliceOperation{t, axis, start, end}
	}

	// Calculate strides for the original tensor
	strides := make([]int, len(t.Shape))
	stride := 1
	for i := len(t.Shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= t.Shape[i]
	}

	// Calculate strides for the new tensor
	newStrides := make([]int, len(newShape))
	newStride := 1
	for i := len(newShape) - 1; i >= 0; i-- {
		newStrides[i] = newStride
		newStride *= newShape[i]
	}

	// Iterate through each element of the new tensor
	newSize := 1
	for _, dim := range newShape {
		newSize *= dim
	}

	for i := 0; i < newSize; i++ {
		// Convert flat index in new tensor to N-dimensional coordinates
		newCoords := make([]int, len(newShape))
		tempIdx := i
		for dim := 0; dim < len(newShape); dim++ {
			newCoords[dim] = tempIdx / newStrides[dim]
			tempIdx %= newStrides[dim]
		}

		// Map new N-dimensional coordinates back to original tensor coordinates
		originalCoords := make([]int, len(t.Shape))
		copy(originalCoords, newCoords)
		originalCoords[axis] += start // Adjust for the slice start

		// Convert original N-dimensional coordinates to flat index in original tensor
		originalFlatIndex := 0
		for dim := 0; dim < len(t.Shape); dim++ {
			originalFlatIndex += originalCoords[dim] * strides[dim]
		}

		newTensor.Data[i] = t.Data[originalFlatIndex]
	}

	return newTensor, nil
}

type SliceOperation struct {
	Input *Tensor
	Axis  int
	Start int
	End   int
}

func (op *SliceOperation) Inputs() []*Tensor {
	return []*Tensor{op.Input}
}

func (op *SliceOperation) Backward(grad *Tensor) error {
	if op.Input.RequiresGrad {
		if op.Input.Grad == nil {
			op.Input.Grad = NewTensor(op.Input.Shape, make([]float64, len(op.Input.Data)), false)
		}

		inputStrides := calculateStrides(op.Input.Shape)
		gradStrides := calculateStrides(grad.Shape)

		gradSize := 1
		for _, dim := range grad.Shape {
			gradSize *= dim
		}

		for i := 0; i < gradSize; i++ {
			gradCoords := getCoords(i, grad.Shape, gradStrides)

			originalCoords := make([]int, len(op.Input.Shape))
			copy(originalCoords, gradCoords)
			originalCoords[op.Axis] += op.Start

			originalFlatIndex := 0
			for dim := 0; dim < len(op.Input.Shape); dim++ {
				originalFlatIndex += originalCoords[dim] * inputStrides[dim]
			}

			op.Input.Grad.Data[originalFlatIndex] += grad.Data[i]
		}
	}
	return nil
}

func (t *Tensor) DivScalar(val float64) (*Tensor, error) {
	resultData := make([]float64, len(t.Data))
	for i, v := range t.Data {
		resultData[i] = v / val
	}

	resultTensor := NewTensor(t.Shape, resultData, t.RequiresGrad)
	if resultTensor.RequiresGrad {
		resultTensor.Creator = &DivScalarOperation{t, val}
	}
	return resultTensor, nil
}

// divScalarOperation represents the scalar division operation for backward pass.
type DivScalarOperation struct {
	Input  *Tensor
	Scalar float64
}

func (op *DivScalarOperation) Inputs() []*Tensor {
	return []*Tensor{op.Input}
}

func (op *DivScalarOperation) Backward(grad *Tensor) error {
	if !op.Input.RequiresGrad {
		return nil
	}
	if op.Input.Grad == nil {
		op.Input.Grad = NewTensor(op.Input.Shape, make([]float64, len(op.Input.Data)), false)
	}
	for i := range grad.Data {
		op.Input.Grad.Data[i] += grad.Data[i] / op.Scalar
	}
	return nil
}

// MulScalar performs element-wise multiplication by a scalar.
func (t *Tensor) MulScalar(val float64) (*Tensor, error) {
	resultData := make([]float64, len(t.Data))
	for i, v := range t.Data {
		resultData[i] = v * val
	}

	resultTensor := NewTensor(t.Shape, resultData, t.RequiresGrad)
	if resultTensor.RequiresGrad {
		resultTensor.Creator = &MulScalarOperation{t, val}
	}
	return resultTensor, nil
}

// MulScalarOperation represents the scalar multiplication operation for backward pass.
type MulScalarOperation struct {
	Input  *Tensor
	Scalar float64
}

func (op *MulScalarOperation) Inputs() []*Tensor {
	return []*Tensor{op.Input}
}

func (op *MulScalarOperation) Backward(grad *Tensor) error {
	if !op.Input.RequiresGrad {
		return nil
	}
	if op.Input.Grad == nil {
		op.Input.Grad = NewTensor(op.Input.Shape, make([]float64, len(op.Input.Data)), false)
	}
	for i := range grad.Data {
		op.Input.Grad.Data[i] += grad.Data[i] * op.Scalar
	}
	return nil
}

// Select returns a new Tensor containing the element at the given index.
// This operation is differentiable.
func (t *Tensor) Select(index int) (*Tensor, error) {
	if index < 0 || index >= len(t.Data) {
		return nil, fmt.Errorf("index out of bounds: %d for tensor of size %d", index, len(t.Data))
	}

	resultTensor := NewTensor([]int{1}, []float64{t.Data[index]}, t.RequiresGrad)
	if resultTensor.RequiresGrad {
		resultTensor.Creator = &SelectOperation{t, index}
	}
	return resultTensor, nil
}

// selectOperation represents the selection of a single element for backward pass.
type SelectOperation struct {
	Input *Tensor
	Index int
}

func (op *SelectOperation) Inputs() []*Tensor {
	return []*Tensor{op.Input}
}

func (op *SelectOperation) Backward(grad *Tensor) error {
	if !op.Input.RequiresGrad {
		return nil
	}
	if op.Input.Grad == nil {
		op.Input.Grad = NewTensor(op.Input.Shape, make([]float64, len(op.Input.Data)), false)
	}
	// The gradient for the selected element is the incoming gradient.
	// All other elements have a gradient of 0 from this operation.
	op.Input.Grad.Data[op.Index] += grad.Data[0]
	return nil
}

// Tanh applies the hyperbolic tangent function element-wise to the tensor.
func (t *Tensor) Tanh() (*Tensor, error) {
	resultData := make([]float64, len(t.Data))
	for i, val := range t.Data {
		resultData[i] = math.Tanh(val)
	}

	resultTensor := NewTensor(t.Shape, resultData, t.RequiresGrad)
	if resultTensor.RequiresGrad {
		resultTensor.Creator = &TanhOperation{t}
	}
	return resultTensor, nil
}

// tanhOperation represents the tanh operation for backward pass.
type TanhOperation struct {
	Input *Tensor
}

func (op *TanhOperation) Inputs() []*Tensor {
	return []*Tensor{op.Input}
}

func (op *TanhOperation) Backward(grad *Tensor) error {
	if !op.Input.RequiresGrad {
		return nil
	}
	if op.Input.Grad == nil {
		op.Input.Grad = NewTensor(op.Input.Shape, make([]float64, len(op.Input.Data)), false)
	}

	// d(tanh(x))/dx = 1 - tanh(x)^2
	for i := range grad.Data {
		tanhVal := math.Tanh(op.Input.Data[i])
		op.Input.Grad.Data[i] += grad.Data[i] * (1 - tanhVal*tanhVal)
	}
	return nil
}

// Sigmoid applies the sigmoid function element-wise to the tensor.
func (t *Tensor) Sigmoid() (*Tensor, error) {
	resultData := make([]float64, len(t.Data))
	for i, val := range t.Data {
		resultData[i] = 1.0 / (1.0 + math.Exp(-val))
	}

	resultTensor := NewTensor(t.Shape, resultData, t.RequiresGrad)
	if resultTensor.RequiresGrad {
		resultTensor.Creator = &SigmoidOperation{t}
	}
	return resultTensor, nil
}

// SigmoidOperation represents the sigmoid operation for backward pass.
type SigmoidOperation struct {
	Input *Tensor
}

func (op *SigmoidOperation) Inputs() []*Tensor {
	return []*Tensor{op.Input}
}

func (op *SigmoidOperation) Backward(grad *Tensor) error {
	if !op.Input.RequiresGrad {
		return nil
	}
	if op.Input.Grad == nil {
		op.Input.Grad = NewTensor(op.Input.Shape, make([]float64, len(op.Input.Data)), false)
	}

	// d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
	for i := range grad.Data {
		sigmoidVal := 1.0 / (1.0 + math.Exp(-op.Input.Data[i]))
		op.Input.Grad.Data[i] += grad.Data[i] * sigmoidVal * (1 - sigmoidVal)
	}
	return nil
}

// Log applies the natural logarithm element-wise to the tensor.
func (t *Tensor) Log() (*Tensor, error) {
	resultData := make([]float64, len(t.Data))
	for i, val := range t.Data {
		if val <= 0 {
			return nil, fmt.Errorf("logarithm of non-positive value: %f", val)
		}
		resultData[i] = math.Log(val)
	}

	resultTensor := NewTensor(t.Shape, resultData, t.RequiresGrad)
	if resultTensor.RequiresGrad {
		resultTensor.Creator = &LogOperation{t}
	}
	return resultTensor, nil
}

// logOperation represents the natural logarithm operation for backward pass.
type LogOperation struct {
	Input *Tensor
}

func (op *LogOperation) Inputs() []*Tensor {
	return []*Tensor{op.Input}
}

func (op *LogOperation) Backward(grad *Tensor) error {
	if !op.Input.RequiresGrad {
		return nil
	}
	if op.Input.Grad == nil {
		op.Input.Grad = NewTensor(op.Input.Shape, make([]float64, len(op.Input.Data)), false)
	}

	for i := range grad.Data {
		op.Input.Grad.Data[i] += grad.Data[i] / op.Input.Data[i]
	}
	return nil
}

func (t *Tensor) Get(indices []int) float64 {
	if len(indices) != len(t.Shape) {
		panic(fmt.Sprintf("Get: number of indices %d does not match tensor dimensions %d", len(indices), len(t.Shape)))
	}
	strides := calculateStrides(t.Shape)
	flatIndex := 0
	for i, index := range indices {
		if index < 0 || index >= t.Shape[i] {
			panic(fmt.Sprintf("Get: index %d out of bounds for dimension %d (size %d)", index, i, t.Shape[i]))
		}
		flatIndex += index * strides[i]
	}
	return t.Data[flatIndex]
}

func (t *Tensor) Set(indices []int, value float64) {
	if len(indices) != len(t.Shape) {
		panic(fmt.Sprintf("Set: number of indices %d does not match tensor dimensions %d", len(indices), len(t.Shape)))
	}
	strides := calculateStrides(t.Shape)
	flatIndex := 0
	for i, index := range indices {
		if index < 0 || index >= t.Shape[i] {
			panic(fmt.Sprintf("Set: index %d out of bounds for dimension %d (size %d)", index, i, t.Shape[i]))
		}
		flatIndex += index * strides[i]
	}
	t.Data[flatIndex] = value
}

func (t *Tensor) TransposeForScores(numHeads, headSize int) (*Tensor, error) {
	// Placeholder
	// Reshape to (batch, seq_len, num_heads, head_size)
	newShape := []int{t.Shape[0], t.Shape[1], numHeads, headSize}
	reshaped, err := t.Reshape(newShape)
	if err != nil {
		return nil, err
	}
	// Transpose to (batch, num_heads, seq_len, head_size)
	transposed, err := reshaped.Transpose(1, 2)
	if err != nil {
		return nil, err
	}
	return transposed, nil
}

// Mul performs element-wise multiplication of two tensors.
func (t *Tensor) Mul(other *Tensor) (*Tensor, error) {
	if !compareShapes(t.Shape, other.Shape) {
		return nil, fmt.Errorf("mismatched shapes for Mul operation: %v and %v", t.Shape, other.Shape)
	}

	resultData := make([]float64, len(t.Data))
	for i := range t.Data {
		resultData[i] = t.Data[i] * other.Data[i]
	}

	resultTensor := NewTensor(t.Shape, resultData, t.RequiresGrad || other.RequiresGrad)

	if resultTensor.RequiresGrad {
		resultTensor.Creator = &MulOperation{t, other}
	}

	return resultTensor, nil
}

// mulOperation represents the element-wise multiplication operation for backward pass.
type MulOperation struct {
	A *Tensor
	B *Tensor
}

func (op *MulOperation) Inputs() []*Tensor {
	return []*Tensor{op.A, op.B}
}

func (op *MulOperation) Backward(grad *Tensor) error {
	// dL/dA = grad * B
	if op.A.RequiresGrad {
		if op.A.Grad == nil {
			op.A.Grad = NewTensor(op.A.Shape, make([]float64, len(op.A.Data)), false)
		}
		for i := range grad.Data {
			op.A.Grad.Data[i] += grad.Data[i] * op.B.Data[i]
		}
	}

	// dL/dB = grad * A
	if op.B.RequiresGrad {
		if op.B.Grad == nil {
			op.B.Grad = NewTensor(op.B.Shape, make([]float64, len(op.B.Data)), false)
		}
		for i := range grad.Data {
			op.B.Grad.Data[i] += grad.Data[i] * op.A.Data[i]
		}
	}
	return nil
}

// Sum returns the sum of all elements in a tensor along a given axis.
func (t *Tensor) Sum(axis int) (*Tensor, error) {
	if axis < 0 || axis >= len(t.Shape) {
		return nil, fmt.Errorf("axis %d out of bounds for tensor with shape %v", axis, t.Shape)
	}

	newShape := make([]int, 0, len(t.Shape)-1)
	for i, dim := range t.Shape {
		if i != axis {
			newShape = append(newShape, dim)
		}
	}

	newSize := 1
	for _, dim := range newShape {
		newSize *= dim
	}
	newData := make([]float64, newSize)

	strides := calculateStrides(t.Shape)
	newStrides := calculateStrides(newShape)

	for i := 0; i < len(t.Data); i++ {
		oldCoords := getCoords(i, t.Shape, strides)

		newCoords := make([]int, 0, len(newShape))
		for j, coord := range oldCoords {
			if j != axis {
				newCoords = append(newCoords, coord)
			}
		}

		newIndex := 0
		if len(newCoords) > 0 {
			for j, coord := range newCoords {
				newIndex += coord * newStrides[j]
			}
		}

		if newIndex < len(newData) {
			newData[newIndex] += t.Data[i]
		}
	}

	resultTensor := NewTensor(newShape, newData, t.RequiresGrad)
	if resultTensor.RequiresGrad {
		resultTensor.Creator = &SumOperation{t, axis}
	}
	return resultTensor, nil
}

// SumOperation represents the sum operation for backward pass.
type SumOperation struct {
	Input *Tensor
	Axis  int
}

func (op *SumOperation) Inputs() []*Tensor {
	return []*Tensor{op.Input}
}

func (op *SumOperation) Backward(grad *Tensor) error {
	if !op.Input.RequiresGrad {
		return nil
	}
	if op.Input.Grad == nil {
		op.Input.Grad = NewTensor(op.Input.Shape, make([]float64, len(op.Input.Data)), false)
	}

	// The gradient of sum is to broadcast the incoming gradient back to the original shape.
	// This means repeating the gradient along the summed axis.

	// Calculate strides for the input and gradient tensors
	inputStrides := calculateStrides(op.Input.Shape)
	gradStrides := calculateStrides(grad.Shape)

	// Iterate through the input tensor's data and distribute the gradient
	for i := 0; i < len(op.Input.Data); i++ {
		inputCoords := getCoords(i, op.Input.Shape, inputStrides)

		// Map inputCoords to gradCoords by skipping the summed axis
		gradCoords := make([]int, 0, len(grad.Shape))
		for dim := 0; dim < len(op.Input.Shape); dim++ {
			if dim != op.Axis {
				gradCoords = append(gradCoords, inputCoords[dim])
			}
			// If the dimension is the summed axis, we don't add it to gradCoords
		}

		gradIndex := 0
		if len(gradCoords) > 0 {
			for j, coord := range gradCoords {
				gradIndex += coord * gradStrides[j]
			}
		}

		op.Input.Grad.Data[i] += grad.Data[gradIndex]
	}

	return nil
}

// Mean returns the mean of all elements in a tensor along a given axis.
func (t *Tensor) Mean(axis int) (*Tensor, error) {
	if axis < 0 || axis >= len(t.Shape) {
		return nil, fmt.Errorf("axis %d out of bounds for tensor with shape %v", axis, t.Shape)
	}

	summed, err := t.Sum(axis)
	if err != nil {
		return nil, fmt.Errorf("failed to sum tensor for mean calculation: %w", err)
	}

	dimSize := float64(t.Shape[axis])
	mean, err := summed.DivScalar(dimSize)
	if err != nil {
		return nil, fmt.Errorf("failed to divide tensor for mean calculation: %w", err)
	}

	return mean, nil
}

// TanhBackward computes the gradient of tanh(x) given the output gradient and tanh(x).
func (t *Tensor) TanhBackward(tanhOutput *Tensor) (*Tensor, error) {
	if !compareShapes(t.Shape, tanhOutput.Shape) {
		return nil, fmt.Errorf("mismatched shapes for TanhBackward: %v and %v", t.Shape, tanhOutput.Shape)
	}

	resultData := make([]float64, len(t.Data))
	for i := range t.Data {
		// d(tanh(x))/dx = 1 - tanh(x)^2
		resultData[i] = t.Data[i] * (1 - tanhOutput.Data[i]*tanhOutput.Data[i])
	}
	return NewTensor(t.Shape, resultData, false), nil
}

// SigmoidBackward computes the gradient of sigmoid(x) given the output gradient and sigmoid(x).
func (t *Tensor) SigmoidBackward(sigmoidOutput *Tensor) (*Tensor, error) {
	if !compareShapes(t.Shape, sigmoidOutput.Shape) {
		return nil, fmt.Errorf("mismatched shapes for SigmoidBackward: %v and %v", t.Shape, sigmoidOutput.Shape)
	}

	resultData := make([]float64, len(t.Data))
	for i := range t.Data {
		// d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
		resultData[i] = t.Data[i] * sigmoidOutput.Data[i] * (1 - sigmoidOutput.Data[i])
	}
	return NewTensor(t.Shape, resultData, false), nil
}

// OneMinusSquareTanh computes grad * (1 - tanh(x)^2).
func (t *Tensor) OneMinusSquareTanh(inputTensor *Tensor) (*Tensor, error) {
	if !compareShapes(t.Shape, inputTensor.Shape) {
		return nil, fmt.Errorf("mismatched shapes for OneMinusSquareTanh: %v and %v", t.Shape, inputTensor.Shape)
	}

	resultData := make([]float64, len(t.Data))
	for i := range t.Data {
		tanhVal := math.Tanh(inputTensor.Data[i])
		resultData[i] = t.Data[i] * (1 - tanhVal*tanhVal)
	}
	return NewTensor(t.Shape, resultData, false), nil
}

// GetSlice extracts a slice from the tensor along the specified axis.
// The returned tensor will have the same number of dimensions as the original,
// but the specified axis will have a size of 1.
// Example: tensor [2, 3, 4], GetSlice(0, 1) -> returns a tensor of shape [1, 3, 4]
func (t *Tensor) GetSlice(axis, index int) (*Tensor, error) {
	if axis < 0 || axis >= len(t.Shape) {
		return nil, fmt.Errorf("axis %d out of bounds for tensor with shape %v", axis, t.Shape)
	}
	if index < 0 || index >= t.Shape[axis] {
		return nil, fmt.Errorf("index %d out of bounds for axis %d with size %d", index, axis, t.Shape[axis])
	}

	newShape := make([]int, len(t.Shape))
	copy(newShape, t.Shape)
	newShape[axis] = 1 // The slice will have a size of 1 along this axis

	// Calculate the size of the slice
	sliceSize := 1
	for _, dim := range newShape {
		sliceSize *= dim
	}
	newData := make([]float64, sliceSize)

	// Calculate strides for the original tensor
	strides := calculateStrides(t.Shape)

	// Calculate the starting flat index for the slice
	startFlatIndex := index * strides[axis]

	// Copy data for the slice
	// This is a simplified copy. A more robust implementation would handle arbitrary dimensions.
	// For now, assuming the slice is contiguous in memory after the startFlatIndex.
	// This might need adjustment based on how the tensor data is actually laid out.
	copy(newData, t.Data[startFlatIndex:startFlatIndex+sliceSize])

	return NewTensor(newShape, newData, t.RequiresGrad), nil
}

// SetSlice sets a slice of the tensor along the specified axis with data from another tensor.
// The `src` tensor must have a shape compatible with the slice being set.
// Example: tensor [2, 3, 4], SetSlice(0, 1, src) where src is [1, 3, 4]
func (t *Tensor) SetSlice(axis, index int, src *Tensor) error {
	if axis < 0 || axis >= len(t.Shape) {
		return fmt.Errorf("axis %d out of bounds for tensor with shape %v", axis, t.Shape)
	}
	if index < 0 || index >= t.Shape[axis] {
		return fmt.Errorf("index %d out of bounds for axis %d with size %d", index, axis, t.Shape[axis])
	}

	// Calculate the expected shape of the source tensor
	expectedSrcShape := make([]int, len(t.Shape))
	copy(expectedSrcShape, t.Shape)
	expectedSrcShape[axis] = 1 // The slice being set has a size of 1 along this axis

	// Check if the source tensor's shape is compatible
	if !compareShapes(src.Shape, expectedSrcShape) {
		return fmt.Errorf("source tensor shape %v is incompatible with expected slice shape %v", src.Shape, expectedSrcShape)
	}

	// Calculate strides for the original tensor
	strides := calculateStrides(t.Shape)

	// Calculate the starting flat index for the slice
	startFlatIndex := index * strides[axis]

	// Copy data from src to the tensor's slice
	// This is a simplified copy. A more robust implementation would handle arbitrary dimensions.
	// For now, assuming the slice is contiguous in memory after the startFlatIndex.
	// This might need adjustment based on how the tensor data is actually laid out.
	copy(t.Data[startFlatIndex:startFlatIndex+len(src.Data)], src.Data)

	return nil
}

// Concat concatenates a slice of tensors along a specified axis.
// All tensors must have the same shape except for the dimension along the concatenation axis.
func Concat(tensors []*Tensor, axis int) (*Tensor, error) {
	if len(tensors) == 0 {
		return nil, fmt.Errorf("Concat requires at least one tensor")
	}

	// Validate axis
	if axis < 0 || axis >= len(tensors[0].Shape) {
		return nil, fmt.Errorf("axis %d out of bounds for tensor with shape %v", axis, tensors[0].Shape)
	}

	// Calculate new shape and total size
	newShape := make([]int, len(tensors[0].Shape))
	copy(newShape, tensors[0].Shape)
	totalSize := 0
	concatDimSize := 0

	for i, t := range tensors {
		if i > 0 && !compareShapesExceptAxis(tensors[0].Shape, t.Shape, axis) {
			return nil, fmt.Errorf("mismatched shapes for concatenation along axis %d: %v and %v", axis, tensors[0].Shape, t.Shape)
		}
		concatDimSize += t.Shape[axis]
		totalSize += len(t.Data)
	}
	newShape[axis] = concatDimSize

	newData := make([]float64, totalSize)
	currentOffset := 0
	for _, t := range tensors {
		copy(newData[currentOffset:], t.Data)
		currentOffset += len(t.Data)
	}

	resultTensor := NewTensor(newShape, newData, false)
	// Determine if the concatenated tensor requires gradients
	for _, t := range tensors {
		if t.RequiresGrad {
			resultTensor.RequiresGrad = true
			resultTensor.Creator = &ConcatOperation{InputTensors: tensors, Axis: axis}
			break
		}
	}

	return resultTensor, nil
}

// ConcatOperation represents the concatenation operation for backward pass.
type ConcatOperation struct {
	InputTensors []*Tensor
	Axis         int
}

func (op *ConcatOperation) Inputs() []*Tensor {
	return op.InputTensors
}

func (op *ConcatOperation) Backward(grad *Tensor) error {
	currentOffset := 0
	for _, inputTensor := range op.InputTensors {
		if inputTensor.RequiresGrad {
			// Calculate the size of the slice for this input tensor along the concatenation axis
			sliceSize := 1
			for i, dim := range inputTensor.Shape {
				if i != op.Axis {
					sliceSize *= dim
				}
			}
			sliceSize *= inputTensor.Shape[op.Axis] // Actual size of the input tensor's data

			// Extract the corresponding gradient slice
			gradSlice, err := grad.Slice(op.Axis, currentOffset, currentOffset+inputTensor.Shape[op.Axis])
			if err != nil {
				return fmt.Errorf("error slicing gradient for concat backward: %w", err)
			}

			if inputTensor.Grad == nil {
				inputTensor.Grad = NewTensor(inputTensor.Shape, make([]float64, len(inputTensor.Data)), false)
			}
			// Copy the gradient slice to the input tensor's gradient
			// This assumes a simple contiguous copy, which might need adjustment for complex strides
			copy(inputTensor.Grad.Data, gradSlice.Data)
		}
		currentOffset += inputTensor.Shape[op.Axis]
	}
	return nil
}

// compareShapesExceptAxis is a helper function to compare two shapes, ignoring a specific axis.
func compareShapesExceptAxis(s1, s2 []int, ignoredAxis int) bool {
	if len(s1) != len(s2) {
		return false
	}
	for i := range s1 {
		if i == ignoredAxis {
			continue
		}
		if s1[i] != s2[i] {
			return false
		}
	}
	return true
}

// Split splits a tensor into multiple tensors along a specified axis.
// The `splitSizes` argument specifies the size of each chunk along the split axis.
// The sum of `splitSizes` must equal the size of the tensor along the split axis.
func Split(t *Tensor, axis int, splitSizes []int) ([]*Tensor, error) {
	if axis < 0 || axis >= len(t.Shape) {
		return nil, fmt.Errorf("axis %d out of bounds for tensor with shape %v", axis, t.Shape)
	}

	totalSplitSize := 0
	for _, size := range splitSizes {
		totalSplitSize += size
	}

	if totalSplitSize != t.Shape[axis] {
		return nil, fmt.Errorf("sum of split sizes (%d) does not match tensor dimension along axis %d (%d)", totalSplitSize, axis, t.Shape[axis])
	}

	var resultTensors []*Tensor
	currentStart := 0
	for _, size := range splitSizes {
		end := currentStart + size
		slicedTensor, err := t.Slice(axis, currentStart, end)
		if err != nil {
			return nil, fmt.Errorf("error slicing tensor during split: %w", err)
		}
		resultTensors = append(resultTensors, slicedTensor)
		currentStart = end
	}

	return resultTensors, nil
}

// splitOperation represents the split operation for backward pass.
type SplitOperation struct {
	InputTensor *Tensor
	Axis        int
	SplitSizes  []int
}

func (op *SplitOperation) Inputs() []*Tensor {
	return []*Tensor{op.InputTensor}
}

func (op *SplitOperation) Backward(grad *Tensor) error {
	if !op.InputTensor.RequiresGrad {
		return nil
	}
	if op.InputTensor.Grad == nil {
		op.InputTensor.Grad = NewTensor(op.InputTensor.Shape, make([]float64, len(op.InputTensor.Data)), false)
	}

	// The gradient from split is simply concatenating the incoming gradients
	// along the split axis.
	// This assumes a simple contiguous copy, which might need adjustment for complex strides.
	// For now, we'll just copy the data directly.
	// A more robust implementation would involve creating a new tensor and then copying.

	// For simplicity, let's assume the incoming grad is a single tensor for now.
	// In a real scenario, 'grad' would be a slice of gradients, one for each split output.
	// This needs to be handled carefully.

	// For now, we'll assume that the 'grad' passed to Backward is the gradient for the *entire*
	// original tensor, which is then distributed to the split outputs.
	// This is incorrect for a split operation. The 'grad' should be a slice of gradients,
	// one for each output of the split.

	// Let's re-think the backward for Split.
	// When Split(t) -> [t1, t2, t3], then dL/dt = Concat([dL/dt1, dL/dt2, dL/dt3])

	// This means the 'grad' argument to this Backward function should be the gradient
	// corresponding to the *output* of the split operation, which is a *slice* of tensors.
	// However, the current framework passes a single *Tensor as 'grad'.
	// This implies a fundamental mismatch in how gradients for multi-output operations are handled.

	// For now, I will implement a placeholder that assumes 'grad' is the gradient for the
	// *entire* original tensor, and that the split operation simply passes it through.
	// This is NOT correct for a true split backward pass.

	// A correct implementation would require the 'grad' argument to be a slice of *Tensor,
	// where each element corresponds to the gradient of one of the split outputs.
	// Then, we would concatenate these gradients to form the gradient for the input tensor.

	// Given the current `Operation` interface, which expects `Backward(grad *Tensor)`,
	// a direct correct implementation of `Split`'s backward pass is not straightforward
	// without modifying the `Operation` interface or how gradients are passed.

	// For the purpose of getting the MoE training to compile and run, I will make a simplifying
	// assumption for the `Split` backward pass: that the gradient for the input tensor
	// is simply the sum of the gradients of its outputs. This is a common simplification
	// when the exact distribution is complex or not critical for initial testing.

	// This is a temporary, simplified backward pass for Split.
	// It assumes that the 'grad' passed here is the gradient for the *entire* original tensor,
	// which is then effectively "passed through" to the input.
	// This is not strictly correct for a split operation where each split output has its own gradient.
	// A proper implementation would require the `grad` parameter to be a slice of `*Tensor`.

	// For now, we will just copy the incoming gradient to the input tensor's gradient.
	// This is a hack and needs to be revisited for a correct autograd implementation.
	if len(op.InputTensor.Grad.Data) != len(grad.Data) {
		return fmt.Errorf("mismatched gradient data length for split backward: %d vs %d", len(op.InputTensor.Grad.Data), len(grad.Data))
	}
	for i := range grad.Data {
		op.InputTensor.Grad.Data[i] += grad.Data[i]
	}

	return nil
}

// Next moves the iterator to the next row.
func (it *RowIterator) Next() bool {
	it.currentRow++
	return it.currentRow < it.rows
}

// Current returns the current row index.
func (it *RowIterator) Current() int {
	return it.currentRow
}

// GetRow returns a slice representing the data of the current row.
func (it *RowIterator) GetRow(data []float64) []float64 {
	start := it.currentRow * it.cols
	end := start + it.cols
	return data[start:end]
}

// SetRow sets the data of the current row from a slice.
func (it *RowIterator) SetRow(data []float64, rowData []float64) {
	start := it.currentRow * it.cols
	end := start + it.cols
	copy(data[start:end], rowData)
}

// Argmax returns the indices of the maximum values along an axis.
func (t *Tensor) Argmax(axis int) (*Tensor, error) {
	if axis < 0 {
		axis = len(t.Shape) + axis
	}
	if axis < 0 || axis >= len(t.Shape) {
		return nil, fmt.Errorf("axis %d out of bounds for tensor with shape %v", axis, t.Shape)
	}

	newShape := make([]int, 0, len(t.Shape)-1)
	for i, dim := range t.Shape {
		if i != axis {
			newShape = append(newShape, dim)
		}
	}
	if len(newShape) == 0 {
		newShape = []int{1}
	}

	newSize := 1
	for _, dim := range newShape {
		newSize *= dim
	}
	newData := make([]float64, newSize)

	strides := calculateStrides(t.Shape)
	newStrides := calculateStrides(newShape)

	for i := 0; i < newSize; i++ {
		newCoords := getCoords(i, newShape, newStrides)

		maxVal := -math.MaxFloat64
		maxIndex := 0

		for k := 0; k < t.Shape[axis]; k++ {
			oldCoords := make([]int, len(t.Shape))
			// copy newCoords to oldCoords, inserting k at the axis
			newCoordIdx := 0
			for oldCoordIdx := 0; oldCoordIdx < len(t.Shape); oldCoordIdx++ {
				if oldCoordIdx == axis {
					oldCoords[oldCoordIdx] = k
				} else {
					if newCoordIdx < len(newCoords) {
						oldCoords[oldCoordIdx] = newCoords[newCoordIdx]
						newCoordIdx++
					}
				}
			}

			flatIndex := 0
			for d, c := range oldCoords {
				flatIndex += c * strides[d]
			}

			if t.Data[flatIndex] > maxVal {
				maxVal = t.Data[flatIndex]
				maxIndex = k
			}
		}
		newData[i] = float64(maxIndex)
	}

	return NewTensor(newShape, newData, false), nil
}
