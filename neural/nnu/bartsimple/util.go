package bartsimple

import (
	"fmt"
	"math"
)

// Optimizer represents an optimization algorithm (e.g., Adam, SGD) for updating model parameters.
type Optimizer struct {
	LearningRate float64
	Parameters   []*Tensor // Slice of tensors representing the model parameters
}

// argmax returns the index of the maximum value in a slice of float64.
// This is a simple implementation; you might need a more robust version.
// This function will likely need to be called after evaluating a tensor.
func argmax(data []float64) int {
	maxVal := data[0]
	maxIndex := 0
	for i, v := range data {
		if v > maxVal {
			maxVal = v
			maxIndex = i
		}
	}
	return int(maxIndex)
}

// onesTensor creates a tensor of the given shape with all elements set to 1.0.
func onesTensor(shape []int) *Tensor {
	size := 1
	for _, dim := range shape {
		size *= dim
	}

	data := make([]float64, size)
	for i := range data {
		data[i] = 1.0
	}

	return &Tensor{
		Data:  data,
		Shape: shape,
	}
}

// BatchIterator allows iterating over batches of a 3D tensor.
type BatchIterator struct {
	tensor       *Tensor
	currentBatch int
}

func Softmax(logits []float64) []float64 {
	if len(logits) == 0 {
		return []float64{}
	}

	// 1. Find the maximum logit for numerical stability
	maxLogit := logits[0]
	for _, logit := range logits {
		if logit > maxLogit {
			maxLogit = logit
		}
	}

	// 2. Calculate exponentials, subtracting the maximum logit
	e := make([]float64, len(logits))
	sumExp := 0.0
	for i, logit := range logits {
		// Subtract the maximum logit BEFORE exponentiation
		e[i] = math.Exp(logit - maxLogit)
		sumExp += e[i]
	}

	// 3. Calculate probabilities
	probabilities := make([]float64, len(logits))
	for i, expValue := range e {
		// Handle the case where sumExp is 0 (should only happen if all logits are -Inf after subtracting max)
		if sumExp == 0 {
			probabilities[i] = 0.0 // Or handle as an error condition depending on desired behavior
		} else {
			probabilities[i] = expValue / sumExp
		}
	}

	return probabilities
}

// BatchRowIterator allows iterating over rows within a specific batch tensor view.
type BatchRowIterator struct {
	batchView  *BatchView
	currentRow int
}

// BatchColumnIterator allows iterating over columns within a specific batch view.
type BatchColumnIterator struct {
	batchView  *BatchView // This is the batch view
	currentCol int
}

// RowIterator returns a new BatchRowIterator for a batch view.
func (bv *BatchView) BatchRowIterator() *BatchRowIterator {
	return &BatchRowIterator{
		batchView:  bv, // Correctly using batchView here
		currentRow: -1,
	}
}

// NextRow moves the iterator to the next row and returns it as a slice.
// It returns nil and false if there are no more rows.
func (bri *BatchRowIterator) NextRow() (Row, bool) {
	bri.currentRow++
	// CORRECTED: Ensure this uses bri.batchView
	if bri.currentRow >= bri.batchView.rows {
		return nil, false
	}

	// Create a new slice for the row data
	// CORRECTED: Ensure this uses bri.batchView
	rowSlice := make([]float64, bri.batchView.cols)
	// CORRECTED: Ensure this uses bri.batchView
	for col := 0; col < bri.batchView.cols; col++ {
		// CORRECTED: Ensure this uses bri.batchView
		rowSlice[col] = bri.batchView.GetElement(bri.currentRow, col)
	}

	return Row(rowSlice), true
}

// GetElement gets an element from the batch view.
// This will trigger computation if the original tensor has a creator.
func (bv *BatchView) GetElement(row, col int) float64 {
	if bv.originalTensor.creator != nil && bv.originalTensor.Data == nil {
		_, err := bv.originalTensor.Compute() // Assuming Compute() fills Data
		if err != nil {
			// Handle error or panic
			panic(fmt.Sprintf("failed to compute original tensor value before GetElement: %v", err))
		}
	}

	if row < 0 || row >= bv.rows || col < 0 || col >= bv.cols {
		panic(fmt.Sprintf("Index out of bounds in batch view: row=%d, col=%d, batch shape=[%d, %d]", row, col, bv.rows, bv.cols))
	}
	// Calculate the flat index in the original tensor
	flatIndex := bv.batchIndex*bv.rows*bv.cols + row*bv.cols + col
	return bv.originalTensor.Data[flatIndex]
}

// GetCurrentRowIndex returns the index of the current row within the batch.
func (bri *BatchRowIterator) GetCurrentRowIndex() int {
	return bri.currentRow
}
func (bv *BatchView) BatchColumnIterator() *BatchColumnIterator {
	return &BatchColumnIterator{
		batchView:  bv, // Use 'batchView' here, matching the struct field
		currentCol: -1,
	}
}

// NextColumn moves the iterator to the next column and returns it.
// It returns nil and false if there are no more columns.
func (bci *BatchColumnIterator) NextColumn() (Column, bool) {
	bci.currentCol++
	var err error
	if bci.currentCol >= bci.batchView.cols { // Access the 'cols' field from batchView
		return nil, false
	}
	// Create a new slice for the column data
	columnSlice := make([]float64, bci.batchView.rows) // Access the 'rows' field from batchView
	for i := 0; i < bci.batchView.rows; i++ {          // Access the 'rows' field from batchView
		columnSlice[i] = bci.batchView.GetElement(i, bci.currentCol) // Call GetElement on batchView
		fmt.Println(err)                                             // This might need to be handled differently with lazy execution
	}
	return Column(columnSlice), true
}

// GetCurrentColumnIndex returns the index of the current column within the batch.
func (bci *BatchColumnIterator) GetCurrentColumnIndex() int {
	return bci.currentCol
}

// Reset resets the column iterator to the beginning.
func (bci *BatchColumnIterator) Reset() {
	bci.currentCol = -1
}

// Define the TransposeIndexIterator struct and methods *outside* the Transpose function

// TransposeIndexIterator iterates through the flattened indices of the transposed tensor
// and provides the corresponding original flattened index.
type TransposeIndexIterator struct {
	newShape     []int
	oldShape     []int
	oldStrides   []int
	newStrides   []int
	axisA, axisB int

	currentNewFlatIndex int
	currentOldFlatIndex int // Added this field

	totalElements int
	start, end    int // Range for this iterator instance

	// Temporary slices to avoid re-allocation in Next()
	tempNewIndices []int
	tempOldIndices []int
}

func NewTransposeIndexIterator(t *Tensor, newShape []int, oldStrides, newStrides []int, axisA, axisB int, start, end int) *TransposeIndexIterator {
	return &TransposeIndexIterator{
		newShape:   newShape,
		oldShape:   t.Shape,
		oldStrides: oldStrides,
		newStrides: newStrides,
		axisA:      axisA,
		axisB:      axisB,

		currentNewFlatIndex: start - 1, // Start before the first element
		currentOldFlatIndex: 0,         // Initialize the new field

		totalElements: (end - start), // Only iterate over the chunk size
		start:         start,
		end:           end,

		tempNewIndices: make([]int, len(newShape)),
		tempOldIndices: make([]int, len(t.Shape)),
	}
}

// Next moves the iterator to the next index.
// It returns true if there are more indices in the range, false otherwise.
func (it *TransposeIndexIterator) Next() bool {
	it.currentNewFlatIndex++
	if it.currentNewFlatIndex >= it.end {
		return false
	}

	// Calculate multi-dimensional indices in the new shape
	flattenedIndex := it.currentNewFlatIndex
	for j := 0; j < len(it.newShape); j++ {
		if it.newStrides[j] > 0 {
			it.tempNewIndices[j] = flattenedIndex / it.newStrides[j]
			flattenedIndex %= it.newStrides[j]
		} else {
			it.tempNewIndices[j] = 0
		}
	}

	// Calculate multi-dimensional indices in the original shape and then the flattened index
	calculatedOldFlatIndex := 0 // Calculate and store
	for j := 0; j < len(it.oldShape); j++ {
		if j == it.axisA {
			it.tempOldIndices[j] = it.tempNewIndices[it.axisB]
		} else if j == it.axisB {
			it.tempOldIndices[j] = it.tempNewIndices[it.axisA]
		} else {
			it.tempOldIndices[j] = it.tempNewIndices[j]
		}
		if j < len(it.oldStrides) && it.oldStrides[j] > 0 {
			calculatedOldFlatIndex += it.tempOldIndices[j] * it.oldStrides[j]
		} else {
			// This case might happen for dimensions with size 1
		}
	}
	// Store the calculated old flattened index in the struct field
	it.currentOldFlatIndex = calculatedOldFlatIndex

	return true
}

// Current returns the current new flattened index and the corresponding old flattened index.
func (it *TransposeIndexIterator) Current() (int, int) {
	return it.currentNewFlatIndex, it.currentOldFlatIndex
}

// Step updates the model parameters based on their gradients and the learning rate.
func (o *Optimizer) Step() {
	for _, parameter := range o.Parameters {
		// Only update parameters that have computed gradients
		if parameter.Grad != nil && len(parameter.Data) == len(parameter.Grad.Data) {
			// Simple SGD update: parameter = parameter - learning_rate * gradient
			for i := range parameter.Data {
				parameter.Data[i] -= o.LearningRate * parameter.Grad.Data[i]
			}
		}
	}
}
