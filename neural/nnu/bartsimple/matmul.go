package bartsimple

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
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

// Row represents a single row of a tensor (as a slice).
type Row []float64

// Column represents a single column of a tensor (as a slice).
type Column []float64

// matMul2DViewWithUnroll performs an optimized 2D matrix multiplication on slice views.
// It calculates resultView = tView @ otherView.
// Shapes: tView (M, K), otherView (K, N), resultView (M, N).
const unrollFactor4D = 4 // Unroll factor for the inner loop

func matMul2DViewWithUnroll(resultView, tView, otherView []float64, M, K, N int) {
	// Initialize the result view to zeros.
	// This is important as we are accumulating results (+=).
	for i := range resultView {
		resultView[i] = 0.0
	}

	// Use a cache-friendly loop order (i, k, j).
	// This improves performance by maximizing sequential memory access.
	for i := 0; i < M; i++ { // Iterate over rows of the output/tView
		for k := 0; k < K; k++ { // Iterate over the common dimension
			// For each element in the i-th row of tView, we will use it
			// to scale the k-th row of otherView and add it to the i-th row of the result.
			t_ik := tView[i*K+k]

			// Unrolled loop over columns (j).
			// This processes multiple elements of the row in a single loop iteration,
			// reducing loop overhead and allowing for better instruction-level parallelism.
			j := 0
			for ; j <= N-unrollFactor4D; j += unrollFactor4D {
				resultView[i*N+j] += t_ik * otherView[k*N+j]
				resultView[i*N+j+1] += t_ik * otherView[k*N+j+1]
				resultView[i*N+j+2] += t_ik * otherView[k*N+j+2]
				resultView[i*N+j+3] += t_ik * otherView[k*N+j+3]
			}

			// Cleanup loop for remaining columns that don't fit in the unrolled part.
			for ; j < N; j++ {
				resultView[i*N+j] += t_ik * otherView[k*N+j]
			}
		}
	}
}

func (t *Tensor) MatMul(other *Tensor) (*Tensor, error) {
	// start := time.Now()
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
		// Close the channel to signal that no more batches will be sent.
		close(batchViewChan)

		// Wait for all goroutines to finish their work.
		wg.Wait()
		// elapsed := time.Since(start)
		// fmt.Println("loop took ", elapsed)
		return result, nil
		// } else if len(t.Shape) == 4 && len(other.Shape) == 4 {
	} else if len(t.Shape) == 4 && len(other.Shape) == 4 {
		// 4D x 4D logic for batched matrix multiplication:
		// [batch, head, M, K] @ [batch, head, K, N] -> [batch, head, M, N]
		// Batch size and dim1 must match, and dim3 must match.
		if t.Shape[0] != other.Shape[0] || t.Shape[1] != other.Shape[1] || t.Shape[3] != other.Shape[2] {
			return nil, fmt.Errorf("incompatible shapes for 4D batched matrix multiplication: %v and %v", t.Shape, other.Shape)
		}

		batchSize := t.Shape[0]
		numHeads := t.Shape[1]
		M := t.Shape[2]
		K := t.Shape[3]
		N := other.Shape[3]

		resultShape := []int{batchSize, numHeads, M, N}
		dataSize := batchSize * numHeads * M * N
		data := make([]float64, dataSize)
		result := NewTensor(data, resultShape, false) // Create the result tensor

		// Use a worker pool to parallelize the 2D matrix multiplications.
		// This is more efficient than iterating sequentially, especially on multi-core systems.
		var wg sync.WaitGroup
		numWorkers := runtime.NumCPU()
		workChan := make(chan struct{ b, h int }, numWorkers)

		// Start worker goroutines
		for i := 0; i < numWorkers; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for work := range workChan {
					b, h := work.b, work.h
					// Get slice views for the current 2D matrices
					tView := t.Data[b*numHeads*M*K+h*M*K : b*numHeads*M*K+(h+1)*M*K]
					otherView := other.Data[b*numHeads*K*N+h*K*N : b*numHeads*K*N+(h+1)*K*N]
					resultView := result.Data[b*numHeads*M*N+h*M*N : b*numHeads*M*N+(h+1)*M*N]

					// Perform optimized 2D matrix multiplication on the views
					matMul2DViewWithUnroll(resultView, tView, otherView, M, K, N)
				}
			}()
		}

		// Use the iterator to send work to the workers
		iter := NewBatchHeadIterator(batchSize, numHeads)
		for iter.Next() {
			b, h := iter.Current()
			workChan <- struct{ b, h int }{b, h}
		}
		close(workChan)
		wg.Wait()

		// elapsed := time.Since(start)
		// fmt.Printf("4D x 4D MatMul (with goroutines) took %s\n", elapsed)
		return result, nil
	} else if len(t.Shape) == 2 && len(other.Shape) == 2 {
		
		// The original implementation for 2D x 2D multiplication was designed for
		// lazy execution, which doesn't work for the current inference path.
		// We'll call MatMulFromScratch to perform the calculation immediately.
		return t.MatMulFromScratch(other)
	}
	// elapsed := time.Since(start)
	// fmt.Printf("2D x 2D MatMul (no goroutines) took %s\n", elapsed)
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
	unrollFactor := 50       // Define the unrolling factor, adjust as needed

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

// Forward performs the forward pass of the matrix multiplication operation.
func (op *MatMulOperation) Forward(inputs ...*Tensor) (*Tensor, error) {
	if len(inputs) != 2 {
		return nil, fmt.Errorf("matMulOperation requires exactly two input tensors, got %d", len(inputs))
	}
	op.input1 = inputs[0]
	op.input2 = inputs[1]

	shape1 := op.input1.Shape
	shape2 := op.input2.Shape

	if len(shape1) == 0 || len(shape2) == 0 {
		return nil, errors.New("matrix multiplication requires at least one dimension")
	}
	if shape1[len(shape1)-1] != shape2[len(shape2)-2] {
		return nil, fmt.Errorf("matrix multiplication incompatible shapes: %v and %v", shape1, shape2)
	}

	// Determine output shape (handle broadcasting of batch dimensions)
	// This part needs to be robust to different numbers of batch dimensions and broadcasting.
	// A more complete implementation would involve padding shapes with 1s to the left
	// to match the longest shape, and then applying broadcasting rules.

	// Simplified broadcasting for now: assuming compatible batch shapes or one is 1
	maxBatchDims := max(len(shape1)-2, len(shape2)-2)
	outputShape := make([]int, maxBatchDims+2)

	// Copy and broadcast batch dimensions
	for i := 0; i < maxBatchDims; i++ {
		d1 := 1
		if len(shape1)-2 > i {
			d1 = shape1[i]
		}
		d2 := 1
		if len(shape2)-2 > i {
			d2 = shape2[i]
		}

		if d1 != d2 && d1 != 1 && d2 != 1 {
			return nil, fmt.Errorf("batch dimensions incompatible for matrix multiplication with broadcasting: %v and %v", shape1[:len(shape1)-2], shape2[:len(shape2)-2])
		}
		outputShape[i] = max(d1, d2)
	}

	outputShape[maxBatchDims] = shape1[len(shape1)-2]
	outputShape[maxBatchDims+1] = shape2[len(shape2)-1]

	outputSize := 1
	for _, dim := range outputShape {
		outputSize *= dim
	}

	// Create the output tensor
	op.output = NewTensor(make([]float64, outputSize), outputShape, op.input1.requiresGrad || op.input2.requiresGrad)

	// Perform matrix multiplication based on shapes
	switch {
	case len(shape1) == 2 && len(shape2) == 2:
		// 2D matrix multiplication computation
		M := shape1[0]
		K := shape1[1]
		N := shape2[1]

		if K != shape2[0] {
			return nil, fmt.Errorf("internal error: shape mismatch in 2D matmul logic during Forward: K (%d) != shape2[0] (%d)", K, shape2[0])
		}

		// Ensure output tensor data slice is allocated
		if op.output.Data == nil || len(op.output.Data) != M*N {
			op.output.Data = make([]float64, M*N)
		}

		// Iterate over each row of the first matrix (A) and compute the corresponding output row.
		// This row-by-row approach is analogous to the batch processing in the 3D case and is cache-friendly.
		for i := 0; i < M; i++ {
			// Get slice views for the current row of A and the corresponding row of the output.
			// These are lightweight views and do not copy the underlying data.
			rowA := op.input1.Data[i*K : (i+1)*K]
			outputRow := op.output.Data[i*N : (i+1)*N]

			// Call the optimized row multiplication function.
			matMulRow2DWithUnroll(outputRow, rowA, op.input2, K, N)
		}
		return op.output, nil
	case len(shape1) == 3 && len(shape2) == 2:
		// 3D x 2D batched matrix multiplication
		// This requires iterating over batches and performing 2D MatMul for each batch.
		// You can adapt your existing 3D x 2D logic from Tensor.MatMul here.
		// Ensure you fill op.output.Data.
		batchSize := shape1[0]
		rowsA, colsA := shape1[1], shape1[2]
		rowsB, colsB := shape2[0], shape2[1] // shape2 is 2D [K, N]

		if colsA != rowsB { // This check should ideally be earlier with shape compatibility
			return nil, fmt.Errorf("incompatible shapes for 3D x 2D matrix multiplication: %v and %v", shape1, shape2)
		}

		resultRows := rowsA
		resultCols := colsB

		// Iterate over batches
		for b := 0; b < batchSize; b++ {
			// Perform 2D MatMul for the current batch of input1 [rowsA, colsA] and input2 [rowsB, colsB]
			// Access slices of data for current batch
			input1BatchData := op.input1.Data[b*rowsA*colsA : (b+1)*rowsA*colsA]
			input2Data := op.input2.Data // input2 is 2D, applied to all batches

			// Calculate output data slice for current batch
			outputBatchData := op.output.Data[b*resultRows*resultCols : (b+1)*resultRows*resultCols]

			// Perform 2D MatMul: input1BatchData [rowsA, colsA] @ input2Data [rowsB, colsB] -> outputBatchData [resultRows, resultCols]
			for i := 0; i < resultRows; i++ {
				for j := 0; j < resultCols; j++ {
					sum := 0.0
					for k := 0; k < colsA; k++ {
						// Access elements in slices
						sum += input1BatchData[i*colsA+k] * input2Data[k*colsB+j]
					}
					outputBatchData[i*resultCols+j] = sum
				}
			}
		}

	case len(shape1) == 4 && len(shape2) == 4:
		// 4D x 4D batched matrix multiplication (e.g., in multi-head attention)
		// [batch, head, seq_len_A, dim_A] @ [batch, head, dim_A, seq_len_B] -> [batch, head, seq_len_A, seq_len_B]
		// This requires iterating over batch and head dimensions and performing 2D MatMul for each.
		// You can adapt your existing 4D x 4D logic from Tensor.MatMul here.
		// Ensure you fill op.output.Data.

		batchSize := shape1[0]
		numHeads := shape1[1]
		rowsA := shape1[2]
		colsA := shape1[3] // Inner dimension
		rowsB := shape2[2] // Inner dimension
		colsB := shape2[3]

		if colsA != rowsB {
			return nil, fmt.Errorf("incompatible shapes for 4D x 4D matrix multiplication: %v and %v", shape1, shape2)
		}

		resultRows := rowsA
		resultCols := colsB

		// Iterate over batches and heads
		for b := 0; b < batchSize; b++ {
			for h := 0; h < numHeads; h++ {
				// Access slices of data for current batch and head
				input1Slice := op.input1.Data[b*numHeads*rowsA*colsA+h*rowsA*colsA : b*numHeads*rowsA*colsA+(h+1)*rowsA*colsA]
				input2Slice := op.input2.Data[b*numHeads*rowsB*colsB+h*rowsB*colsB : b*numHeads*rowsB*colsB+(h+1)*rowsB*colsB]

				// Calculate output data slice for current batch and head
				outputSlice := op.output.Data[b*numHeads*resultRows*resultCols+h*resultRows*resultCols : b*numHeads*resultRows*resultCols+(h+1)*resultRows*resultCols]

				// Perform 2D MatMul: input1Slice [rowsA, colsA] @ input2Slice [rowsB, colsB] -> outputSlice [resultRows, resultCols]
				for i := 0; i < resultRows; i++ {
					for j := 0; j < resultCols; j++ {
						sum := 0.0
						for k := 0; k < colsA; k++ {
							// Access elements in slices
							sum += input1Slice[i*colsA+k] * input2Slice[k*colsB+j]
						}
						outputSlice[i*resultCols+j] = sum
					}
				}
			}
		}

	default:
		return nil, fmt.Errorf("unsupported tensor shapes for matrix multiplication: %v and %v", shape1, shape2)
	}

	// Set the creator of the output tensor
	if op.output.requiresGrad {
		op.output.creator = op // Set the creator to the matMulOperation itself
	}

	return op.output, nil
}

const unrollFactor2D = 4

// matMulRow2DWithUnroll computes one row of the output matrix C for C = A @ B,
// where A and B are 2D matrices. It takes a single row from A and the entire B matrix.
// It uses a cache-friendly loop order and loop unrolling for performance.
func matMulRow2DWithUnroll(
	outputRow []float64, // slice view into the output tensor's data for the current row
	rowA []float64,      // slice view for the current row of tensor A
	tensorB *Tensor,     // the entire tensor B
	K, N int,            // K is common dimension, N is columns of B
) {
	// This function calculates: outputRow = rowA @ tensorB
	// outputRow has length N, rowA has length K, tensorB is KxN.

	// Initialize the output row to zeros.
	for i := range outputRow {
		outputRow[i] = 0.0
	}

	// The (i, k, j) loop order is generally more cache-friendly than (i, j, k).
	// We iterate through each element of rowA (k loop), and for each element,
	// we iterate through the corresponding row in B (j loop).
	for k := 0; k < K; k++ { // k is the common dimension
		a_k := rowA[k]
		rowB := tensorB.Data[k*N : (k+1)*N] // Get a view of the k-th row of B

		// Unrolled loop over columns of B
		j := 0
		for ; j <= N-unrollFactor2D; j += unrollFactor2D {
			outputRow[j] += a_k * rowB[j]
			outputRow[j+1] += a_k * rowB[j+1]
			outputRow[j+2] += a_k * rowB[j+2]
			outputRow[j+3] += a_k * rowB[j+3]
		}

		// Cleanup loop for remaining columns that didn't fit in the unrolled part.
		for ; j < N; j++ {
			outputRow[j] += a_k * rowB[j]
		}
	}
}

// Backward performs the backward pass for the matrix multiplication operation.
func (op *MatMulOperation) Backward(grad *Tensor) error {
	// ... (Implement backward pass logic for 2D, 3D, 4D cases) ...
	// This will involve matrix multiplications of the incoming gradient with the inputs or their transposes.
	// Remember to accumulate gradients to op.input1.Grad and op.input2.Grad.
	return errors.New("matmulOperation.Backward not yet fully implemented") // Placeholder
}

// Inputs returns the input tensors of the matrix multiplication operation.
func (op *MatMulOperation) Inputs() []*Tensor {
	// ... (Return input tensors) ...
	return []*Tensor{op.input1, op.input2} // Assuming two inputs
}

type MatMulOperation struct {
	input1 *Tensor // A
	input2 *Tensor // B
	output *Tensor // C = A @ B
}

// AutogradMatMul performs matrix multiplication with autograd support.
// MOVED THIS FUNCTION TO HERE (matmul.go)
func (t *Tensor) AutogradMatMul(other *Tensor) (*Tensor, error) {
	fmt.Printf("AutogradMatMul: t.Shape = %v, other.Shape = %v\n", t.Shape, other.Shape)
	// Perform the forward pass using the MatMul function.
	// The MatMul function should create and return the output of the MatMulOperation.
	result, err := t.MatMul(other) // Tensor.MatMul should now create and run the operation
	if err != nil {
		return nil, fmt.Errorf("error during AutogradMatMul forward pass: %w", err)
	}
	// The MatMul function (which uses the MatMulOperation's Forward)
	// should already set the creator if requiresGrad is true.
	// This part might be redundant if MatMul's internal operation.Forward sets the creator.
	// Double-check if result.creator is already set correctly by result, err := t.MatMul(other)
	// if result != nil && (t.requiresGrad || other.requiresGrad) {
	//     result.requiresGrad = true // Should be set by op.Forward
	//     result.creator = op        // Should be set by op.Forward
	// }

	return result, nil
}

// MatMulFromScratch implements a basic 2D matrix multiplication from scratch.
// It does not use external libraries and is intended for understanding the process
// or for very small tensors where overhead might matter.
// This function will be called during the Compute() phase of a

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
