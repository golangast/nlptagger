package bartsimple

import (
	"fmt"
	"runtime"
	"sync"
	"time"
)

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
	// Move to the next column chunk
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

// Current returns the current indices (batch, head, starting row of chunk, starting col of chunk) being tracked by the iterator.
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

// Row represents a single row of a tensor (as a slice).
type Row []float64

// Column represents a single column of a tensor (as a slice).
type Column []float64

func (t *Tensor) MatMul(other *Tensor) (*Tensor, error) {
	start := time.Now()
	if len(t.Shape) == 3 && len(other.Shape) == 2 {
		batchSize := t.Shape[0] // Get the batch size
		rowsA, colsA := t.Shape[1], t.Shape[2]
		rowsB, colsB := other.Shape[0], other.Shape[1]

		// Check for compatible shapes for batched multiplication
		if colsA != rowsB {
			return nil, fmt.Errorf("incompatible shapes for 3D batched matrix multiplication: %v and %v", t.Shape, other.Shape)
		}

		resultRows := rowsA
		resultCols := colsB

		// Create the result tensor with the batch dimension
		// Calculate the total size of the tensor data
		dataSize := batchSize * resultRows * resultCols
		// Create a slice of float64 with the calculated size and initialize it with zeros
		data := make([]float64, dataSize)
		result := NewTensor(data, []int{batchSize, resultRows, resultCols}, false)

		// Get a batch iterator for tensor t
		batchIter, err := t.BatchIterator()
		if err != nil {
			return nil, err
		}
		var wg sync.WaitGroup
		numGoroutines := runtime.NumCPU() // Or a fixed number

		// Iterate over batches sequentially
		for batchView, ok := batchIter.NextBatch(); ok; batchView, ok = batchIter.NextBatch() {
			currentBatchIndex := batchIter.GetCurrentBatchIndex() // Get the current batch index

			// Get a row iterator for the current batch view
			batchRowIter := batchView.BatchRowIterator() // Use the BatchRowIterator for the view

			// Iterate over rows of the current batch view
			for rowA, okA := batchRowIter.NextRow(); okA; rowA, okA = batchRowIter.NextRow() {
				currentRowIndex := batchRowIter.GetCurrentRowIndex()

				// Call the helper function to perform matrix multiplication for this row
				err := MatMulRow(rowA, other, result, currentBatchIndex, currentRowIndex, colsA, resultCols)
				if err != nil {
					return nil, fmt.Errorf("MatMulRow failed: %w", err)
				}
			}
		}

		// Channel to send batch views to worker goroutines
		batchViewChan := make(chan *BatchView, numGoroutines)

		// Start worker goroutines
		for i := 0; i < numGoroutines; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for batchView := range batchViewChan {
					currentBatchIndex := batchView.batchIndex
					batchRowIter := batchView.BatchRowIterator()

					for rowA, okA := batchRowIter.NextRow(); okA; rowA, okA = batchRowIter.NextRow() {
						currentRowIndex := batchRowIter.GetCurrentRowIndex()

						err := MatMulRow(rowA, other, result, currentBatchIndex, currentRowIndex, colsA, resultCols)
						if err != nil {
							fmt.Printf("Error in goroutine processing batch %d, row %d: %v\\n", currentBatchIndex, currentRowIndex, err)
						}
					}
				}
			}()
		}
		for batchView, ok := batchIter.NextBatch(); ok; batchView, ok = batchIter.NextBatch() {
			batchViewChan <- batchView
		}
		elapsed := time.Since(start)
		fmt.Println("loop took ", elapsed)
		return result, nil
		// } else if len(t.Shape) == 4 && len(other.Shape) == 4 {
	} else if len(t.Shape) == 4 && len(other.Shape) == 4 {
		// 4D x 4D logic for batched matrix multiplication:
		// [batch, dim1, dim2, dim3] * [batch, dim1, dim3, dim4]
		// Batch size and dim1 must match, and dim3 must match.
		if t.Shape[0] != other.Shape[0] || t.Shape[1] != other.Shape[1] || t.Shape[3] != other.Shape[2] {
			return nil, fmt.Errorf("incompatible shapes for 4D batched matrix multiplication: %v and %v", t.Shape, other.Shape)
		}

		// Add print statements to show tensor shapes
		//fmt.Printf("4D MatMul: t.Shape = %v, other.Shape = %v\n", t.Shape, other.Shape)

		batchSize := t.Shape[0]
		numHeads := t.Shape[1]
		dim2_t := t.Shape[2]         // Corresponds to seq_length in Q
		dim3_t := t.Shape[3]         // Corresponds to head_dim in Q and K
		dim3_other := other.Shape[3] // Corresponds to seq_length in K^T (dim4_other)

		// Expected result shape: [batch, num_heads, dim2_t, dim3_other]
		resultShape := []int{batchSize, numHeads, dim2_t, dim3_other}
		// Calculate the total size of the tensor data from the resultShape
		dataSize := 1
		for _, dim := range resultShape {
			dataSize *= dim
		}
		// Create a slice of float64 with the calculated size and initialize it with zeros
		data := make([]float64, dataSize)
		result := NewTensor(data, resultShape, false) // Create the result tensor

		// Create the 4D matrix multiplication iterator
		// Add print statements to inspect input tensor data before multiplication
		// fmt.Printf("4D MatMul: t.Data (first %d elements): %v\n", min(20, len(t.Data)), t.Data[:min(20, len(t.Data))])
		// fmt.Printf("4D MatMul: other.Data (first %d elements): %v\n", min(20, len(other.Data)), other.Data[:min(20, len(other.Data))])

		iter := NewMatMul4DIterator(batchSize, numHeads, dim2_t, dim3_other)

		// Parallelize the outer loops (batch and head) or the chunk iterations
		for iter.Next() { // This iterator now iterates over chunks
			b, h, startRow, startCol := iter.Current()
			chunkRows, chunkCols := iter.CurrentChunkSize()

			// Iterate through the elements within the current chunk
			for i := 0; i < chunkRows; i++ {
				for j := 0; j < chunkCols; j++ { // This is the outer loop over columns of the result chunk
					// Calculate the absolute row and column indices within the 2D slice
					MatMulElement4D(t, other, result, b, h, startRow+i, startCol+j, dim3_t) // Call the helper function
				}
			}
		}

		// Add print statements to inspect the result tensor data after multiplication
		//fmt.Printf("4D MatMul: result.Data (first %d elements): %v\n", min(20, len(result.Data)), result.Data[:min(20, len(result.Data))])

		elapsed := time.Since(start)
		fmt.Printf("2D x 2D MatMul (no goroutines) took %s\n", elapsed)
		return result, nil
	} else if len(t.Shape) == 2 && len(other.Shape) == 2 {
		for i := 0; i < len(t.Shape)-2; i++ {
			if t.Shape[i] != other.Shape[i] {
				return nil, fmt.Errorf("batch dimensions incompatible for matrix multiplication: %v and %v", t.Shape, other.Shape)
			}
		}

		// Calculate output shape
		outputShape := make([]int, max(len(t.Shape), len(other.Shape))-1)
		// Copy batch dimensions
		if len(t.Shape) > len(other.Shape) {
			copy(outputShape, t.Shape[:len(t.Shape)-2])
		} else {
			copy(outputShape, other.Shape[:len(other.Shape)-2])
		}
		outputShape[len(outputShape)-2] = t.Shape[len(t.Shape)-2]         // M
		outputShape[len(outputShape)-1] = other.Shape[len(other.Shape)-1] // N

		// Create a placeholder output tensor
		outputTensor := NewTensor(nil, outputShape, false) // Data will be filled in Forward

		// Create the matMulOperation
		operation := &matMulOperation{input1: t, input2: other, output: outputTensor}

		// Set the creator of the output tensor
		outputTensor.creator = operation

		return outputTensor, nil
		// Existing 2D x 2D logic
		// result, err := t.MatMulFromScratch(other)
		// if err != nil {
		// 	return nil, fmt.Errorf("2D matrix multiplication failed: %w", err)
		// }

		// return result, nil

	}
	elapsed := time.Since(start)
	fmt.Printf("2D x 2D MatMul (no goroutines) took %s\n", elapsed)
	return nil, fmt.Errorf("unsupported tensor shapes for matrix multiplication: %v and %v", t.Shape, other.Shape)
}

// MatMulRow4d performs the matrix multiplication of a row from a 4D tensor (t)
// with a 2D slice from another 4D tensor (other), writing the result into the
// specified row of a 4D result tensor. It avoids creating temporary slices for
// the 2D slice of 'other' by calculating indices directly. The 'other' tensor // Corrected: Added result *Tensor
// should be the original 4D tensor.
func MatMulRow4d(
	rowA Row, // A single row from tensor A (as a slice view)
	other *Tensor, // The second tensor B (4D - original)
	result *Tensor, // The result tensor (4D)
	currentBatchIndex int,
	currentHeadIndex int,
	currentRowIndex int, // Row index within the 2D slice of the batch/head
	commonDim int, // Number of columns in rowA (dim3_t) and rows in the 2D slice of other (dim3_t) // Corrected: Removed result *Tensor, as it's part of the main MatMul function scope
	otherDim3 int, // Add other.Shape[3] (dim4_other) as explicit argument// Corrected: Removed result *Tensor, as it's part of the main MatMul function scope
) error {
	// 'other' is now the original 4D tensor: [batch, num_heads, dim3_t, dim4_other]
	// Calculate the base offset for the current batch and head in tensor 'other'
	// other shape: [batch, num_heads, dim3_t, dim4_other]
	otherBaseOffset := currentBatchIndex*other.Shape[1]*other.Shape[2]*other.Shape[3] + currentHeadIndex*other.Shape[2]*other.Shape[3] // Corrected other shape reference	// The stride to move down a row in the 2D slice of 'other' within its 4D structure is other.Shape[3].

	colDimOther := otherDim3 // Define colDimOther here
	unrollFactor := 30       // Define the unrolling factor, adjust as needed

	sum := 0.0 // Declare sum here, outside of the inner loop
	// Iterate through columns of the result row with unrolling
	j := 0                                                       // result.Shape[3] will be replaced by resultCols
	for ; j <= result.Shape[3]-unrollFactor; j += unrollFactor { // Added check against resultCols

		// Calculate sums for the unrolled columns
		sum1, sum2, sum3, sum4 := 0.0, 0.0, 0.0, 0.0
		for k := 0; k < commonDim; k++ { // Use commonDim

			a := rowA[k] // Access element from rowA (view)
			// Calculate direct flat index for other.Data at [currentBatchIndex, currentHeadIndex, k, col] other shape: [batch, num_heads, dim3_t, dim4_other] colDimOther := otherDim3 // Use the passed argument
			otherFlatIndex1 := otherBaseOffset + k*otherDim3 + j
			otherFlatIndex2 := otherBaseOffset + k*colDimOther + (j + 1)
			otherFlatIndex3 := otherBaseOffset + k*colDimOther + (j + 2)
			otherFlatIndex4 := otherBaseOffset + k*colDimOther + (j + 3)

			// Debugging print statements
			// fmt.Printf("Batch: %d, Head: %d, Row: %d, k: %d, j: %d, ColDimOther: %d, OtherBaseOffset: %d\n",
			// currentBatchIndex, currentHeadIndex, currentRowIndex, k, j, colDimOther, otherBaseOffset)
			// fmt.Printf("Flat Indices: %d, %d, %d, %d\n",
			// otherFlatIndex1, otherFlatIndex2, otherFlatIndex3, otherFlatIndex4)
			// fmt.Printf("Length of other.Data: %d\n", len(other.Data))

			if otherFlatIndex1 >= len(other.Data) || otherFlatIndex2 >= len(other.Data) || otherFlatIndex3 >= len(other.Data) || otherFlatIndex4 >= len(other.Data) {
				fmt.Printf("PANIC SOON: Index out of bounds detected before access.\n")
				// You might want to panic here or return an error to stop execution immediately
			}

			sum1 += a * other.Data[otherFlatIndex1]
			sum2 += a * other.Data[otherFlatIndex2]
			sum3 += a * other.Data[otherFlatIndex3]
			sum4 += a * other.Data[otherFlatIndex4]
		}

		// Store the final sums for the unrolled columns
		resultBaseIndex := currentBatchIndex*result.Shape[1]*result.Shape[2]*result.Shape[3] + currentHeadIndex*result.Shape[2]*result.Shape[3] + currentRowIndex*result.Shape[3]
		result.Data[resultBaseIndex+j] = sum1
		result.Data[resultBaseIndex+(j+1)] = sum2
		result.Data[resultBaseIndex+(j+2)] = sum3
		result.Data[resultBaseIndex+(j+3)] = sum4
	}

	// Handle remaining iterations (cleanup loop for j)
	for ; j < otherDim3; j++ { // Use otherDim3 which represents resultCols for this 2D slice

		sum = 0.0 // Reset sum for each cleanup column
		for k := 0; k < commonDim; k++ {
			colDimOther := otherDim3 // Use the passed argument
			otherFlatIndex := otherBaseOffset + k*colDimOther + j
			sum += rowA[k] * other.Data[otherFlatIndex]
		}
		// Store the final sum for the current column (within the main loop)
		resultIndex := currentBatchIndex*result.Shape[1]*result.Shape[2]*result.Shape[3] + currentHeadIndex*result.Shape[2]*result.Shape[3] + currentRowIndex*result.Shape[3] + j

		// Store the final sum for the current column (within the main loop)
		result.Data[resultIndex] = sum

	}
	return nil

}

// AutogradMatMul performs matrix multiplication with autograd support.
func (t *Tensor) AutogradMatMul(other *Tensor) (*Tensor, error) {
	fmt.Printf("AutogradMatMul: t.Shape = %v, other.Shape = %v\n", t.Shape, other.Shape)
	// Create a MatMulOperation.
	// Assuming MatMulOperation is defined elsewhere and accessible within the 'bart' package.
	op := &MatMulOperation{Input1: t, Input2: other}
	// Perform the forward pass using the MatMul function.
	// The MatMul function should be defined elsewhere in the 'bart' package (e.g., in matmul.go).
	result, err := t.MatMul(other) // Assuming MatMul method is defined for *Tensor in the 'bart' package
	if err != nil {
		return nil, fmt.Errorf("error during AutogradMatMul forward pass: %w", err)
	}
	// Set the creator of the result tensor to this operation if gradient is required.
	if t.requiresGrad || other.requiresGrad {
		result.requiresGrad = true
		result.creator = op // Set the operation as the creator
	}
	return result, nil
}

// MatMulFromScratch implements a basic 2D matrix multiplication from scratch.
// It does not use external libraries and is intended for understanding the process
// or for very small tensors where overhead might matter.
// This function will be called during the Compute() phase of a MatMulOperation
// or directly by the 2D MatMul case.
func (t *Tensor) MatMulFromScratch(other *Tensor) (*Tensor, error) {
	// Ensure input tensors have their data computed.
	if t.creator != nil && t.Data == nil {
		_, err := t.Compute() // Assuming Compute() exists and fills t.Data
		if err != nil {
			return nil, fmt.Errorf("failed to compute left tensor value before MatMulFromScratch: %w", err)
		}
	}
	if other.creator != nil && other.Data == nil {
		_, err := other.Compute() // Assuming Compute() exists and fills other.Data
		if err != nil {
			return nil, fmt.Errorf("failed to compute right tensor value before MatMulFromScratch: %w", err)
		}
	}

	// Basic shape checks for 2D matrix multiplication
	if len(t.Shape) != 2 || len(other.Shape) != 2 {
		return nil, fmt.Errorf("MatMulFromScratch only supports 2D tensors")
	}

	rowsA, colsA := t.Shape[0], t.Shape[1]
	rowsB, colsB := other.Shape[0], other.Shape[1]

	if colsA != rowsB {
		return nil, fmt.Errorf("incompatible shapes for matrix multiplication: %v and %v", t.Shape, other.Shape)
	}

	// Result tensor shape
	resultRows := rowsA
	resultCols := colsB
	resultShape := []int{resultRows, resultCols}
	// Calculate the total size of the tensor data from the resultShape
	dataSize := 1
	for _, dim := range resultShape {
		dataSize *= dim
	}
	// Create a slice of float64 with the calculated size and initialize it with zeros
	data := make([]float64, dataSize)
	result := NewTensor(data, resultShape, false) // Assuming NewTensor allocates the Data slice

	// Pre-calculate strides for efficiency
	strideA := colsA
	strideB := colsB
	strideResult := resultCols

	// Manual matrix multiplication using nested loops
	for i := 0; i < resultRows; i++ { // Iterate over rows of the result
		for j := 0; j < resultCols; j++ { // Iterate over columns of the result
			sum := 0.0
			// Iterate over the common dimension (colsA or rowsB)
			for k := 0; k < colsA; k++ {
				// Access elements using calculated flat indices
				sum += t.Data[i*strideA+k] * other.Data[k*strideB+j]

			}
			// Store the result in the result tensor
			result.Data[i*strideResult+j] = sum

		}
	}
	// Inside the k loop in MatMulFromScratch

	return result, nil
}

// MatMulElement4D calculates a single element in the 4D matrix multiplication result.
// It performs the dot product for result[b, h, row, col] = sum(t[b, h, row, k] * other[b, h, k, col] for k).
// This function is called by the parallelized 4D MatMul.
func MatMulElement4D(t, other, result *Tensor, b, h, row, col, commonDim int) {
	sum := 0.0
	// Perform the dot product for the current element (b, h, row, col)
	// Iterate over the common dimension (dim3_t in the MatMul function)
	for k := 0; k < commonDim; k++ {
		// Calculate the flat index for element t[b, h, row, k]
		// t shape: [batch, num_heads, dim2_t, dim3_t]
		tFlatIndex := b*t.Shape[1]*t.Shape[2]*t.Shape[3] +
			h*t.Shape[2]*t.Shape[3] +
			row*t.Shape[3] +
			k

		// Calculate the flat index for element other[b, h, k, col]
		// other shape: [batch, num_heads, dim3_t, dim4_other]
		otherFlatIndex := b*other.Shape[1]*other.Shape[2]*other.Shape[3] +
			h*other.Shape[2]*other.Shape[3] +
			k*other.Shape[3] +
			col
		// Inside the k loop in MatMulElement4D
		// fmt.Printf("  Accessing t.Data[%d] (%f) and other.Data[%d] (%f)\n", tFlatIndex, t.Data[tFlatIndex], otherFlatIndex, other.Data[otherFlatIndex])

		// // After the k loop in MatMulElement4D
		// fmt.Printf("  Calculated sum for result[%d, %d, %d, %d]: %f\n", b, h, row, col, sum)

		sum += t.Data[tFlatIndex] * other.Data[otherFlatIndex]
	}

	// Store the calculated dot product in the result tensor
	result.SetElement(sum, b, h, row, col) // Assuming SetElement handles 4D indices
}

// MatMulRow performs the matrix multiplication of a row from a 3D tensor (t)
// with a 2D tensor (other), writing the result into the specified row of a 3D
// result tensor. This is a helper for the 3D x 2D MatMul case.
func MatMulRow(
	rowA Row, // A single row from tensor A (as a slice view)
	other *Tensor, // The second tensor B (2D)
	result *Tensor, // The result tensor (3D)
	currentBatchIndex int,
	currentRowIndex int,
	colsA int, // Number of columns in tensor A (and rows in tensor B)
	resultCols int, // Number of columns in the result tensor (and tensor B)
) error {

	// Calculate the base flat index for the current row in the result tensor
	resultBaseIndex := currentBatchIndex*result.Shape[1]*result.Shape[2] + currentRowIndex*result.Shape[2]

	// Calculate the stride for accessing elements in a column of 'other' (assuming row-major)
	otherColStride := other.Shape[1]

	// Iterate through columns of the result (which is colsB)
	for j := 0; j < resultCols; j++ {
		sum := 0.0
		// Perform dot product of rowA and column j of tensor other
		for k := 0; k < colsA; k++ {
			a := rowA[k]
			// Calculate direct index for other.Data (k, j) assuming row-major
			otherIndex := k*otherColStride + j
			b := other.Data[otherIndex]
			sum += a * b
		}
		// Store the final sum for the current column
		resultIndex := resultBaseIndex + j
		result.Data[resultIndex] = sum
	}
	return nil
}

// BatchIterator returns a new BatchIterator for the tensor.
func (t *Tensor) BatchIterator() (*BatchIterator, error) {
	if len(t.Shape) != 3 {
		return nil, fmt.Errorf("BatchIterator only supports 3D tensors")
	}
	return &BatchIterator{
		tensor:       t,
		currentBatch: -1, // Start before the first batch
	}, nil
}
func (bi *BatchIterator) NextBatch() (*BatchView, bool) {
	bi.currentBatch++
	if bi.currentBatch >= bi.tensor.Shape[0] {
		return nil, false // No more batches
	}

	rows, cols := bi.tensor.Shape[1], bi.tensor.Shape[2]
	batchView := &BatchView{
		originalTensor: bi.tensor,
		batchIndex:     bi.currentBatch,
		rows:           rows,
		cols:           cols,
	}

	return batchView, true // CORRECTED: Return a *BatchView
}

// GetCurrentBatchIndex returns the index of the current batch.
func (bi *BatchIterator) GetCurrentBatchIndex() int {
	return bi.currentBatch
}
