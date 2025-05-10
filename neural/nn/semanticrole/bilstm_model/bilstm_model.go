// Package bilstm_model provides a Bidirectional LSTM for semantic role labeling.
// It processes sentences forward and backward to understand word roles.
package bilstm_model

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
)

type LSTMWeights struct {
	InputWeights  [][]float64
	HiddenWeights [][]float64
	Bias          []float64
}

type LSTMState struct {
	HiddenStates   [][]float64
	CellStates     [][]float64
	InputGates     [][]float64
	ForgetGates    [][]float64
	OutputGates    [][]float64
	CandidateCells [][]float64
}

type BiLSTMModel struct {
	ForwardLSTMWeights LSTMWeights

	BackwardLSTMWeights LSTMWeights
	OutputWeights       [][]float64
	OutputBias          []float64
	HiddenSize          int
	VocabSize           int
	OutputSize          int
	Gradients           *Gradients
	ForwardLSTMState    LSTMState
	BackwardLSTMState   LSTMState
}

type Gradients struct {
	OutputWeightGradients         [][]float64
	OutputBiasGradients           []float64
	ForwardInputWeightGradients   [][]float64
	ForwardHiddenWeightGradients  [][]float64
	ForwardBiasGradients          []float64
	BackwardInputWeightGradients  [][]float64
	BackwardHiddenWeightGradients [][]float64
	BackwardBiasGradients         []float64
}

// WeightIterator is used to iterate over the weights.
type WeightIterator struct {
	i       int
	j       int
	maxI    int
	maxJ    int
	current *BiLSTMModel
}

func NewWeightIterator(m *BiLSTMModel) *WeightIterator {
	if m == nil || m.ForwardLSTMWeights.InputWeights == nil {
		return &WeightIterator{maxI: 0, maxJ: 0, current: m}
	}
	return &WeightIterator{
		i:       0,
		j:       0,
		maxI:    len(m.ForwardLSTMWeights.InputWeights),
		maxJ:    len(m.ForwardLSTMWeights.InputWeights[0]),
		current: m,
	}
}

// BackwardWeightIterator is used to iterate over the weights.
type BackwardWeightIterator struct {
	i       int
	j       int
	maxI    int
	maxJ    int
	current *BiLSTMModel
}

func NewBackwardWeightIterator(m *BiLSTMModel) *BackwardWeightIterator {
	if m == nil || m.BackwardLSTMWeights.InputWeights == nil {
		return &BackwardWeightIterator{maxI: 0, maxJ: 0, current: m}
	}
	return &BackwardWeightIterator{
		i:       0,
		j:       0,
		maxI:    len(m.BackwardLSTMWeights.InputWeights),
		maxJ:    len(m.BackwardLSTMWeights.InputWeights[0]),
		current: m,
	}
}

func NewBiLSTMModel(vocabSize, hiddenSize, outputSize int) *BiLSTMModel {
	if outputSize <= 0 {
		panic(fmt.Sprintf("invalid outputSize: must be greater than 0"))
	}
	model := &BiLSTMModel{
		ForwardLSTMWeights:  newLSTMWeights(vocabSize, hiddenSize),
		BackwardLSTMWeights: newLSTMWeights(vocabSize, hiddenSize),
		HiddenSize:          hiddenSize,
		VocabSize:           vocabSize,
		OutputSize:          outputSize,
		OutputWeights:       make([][]float64, outputSize),
		OutputBias:          make([]float64, outputSize),

		Gradients: &Gradients{
			OutputWeightGradients:         make2D(outputSize, 2*hiddenSize),
			OutputBiasGradients:           make([]float64, outputSize),
			ForwardInputWeightGradients:   make2D(4*hiddenSize, vocabSize),
			ForwardHiddenWeightGradients:  make2D(4*hiddenSize, hiddenSize),
			ForwardBiasGradients:          make([]float64, 4*hiddenSize),
			BackwardInputWeightGradients:  make2D(4*hiddenSize, vocabSize),
			BackwardHiddenWeightGradients: make2D(4*hiddenSize, hiddenSize),
			BackwardBiasGradients:         make([]float64, 4*hiddenSize),
		},
	}
	model.InitializeOutputLayer(hiddenSize)
	initializeWeights(model)
	return model
}

func (iter *WeightIterator) Next() bool {
	if iter.current == nil || iter.current.ForwardLSTMWeights.InputWeights == nil {
		return false
	}
	if iter.i >= iter.maxI {
		return false
	}
	if iter.j < iter.maxJ-1 {
		iter.j++
	} else {
		iter.j = 0
		iter.i++
	}
	if iter.i >= iter.maxI {
		return false
	}
	return true
}
func (iter *WeightIterator) Value() (int, int) {
	return iter.i, iter.j
}

func (iter *BackwardWeightIterator) Next() bool {
	if iter.current == nil || iter.current.BackwardLSTMWeights.InputWeights == nil {
		return false
	}
	if iter.i >= iter.maxI {
		return false
	}
	if iter.j < iter.maxJ-1 {
		iter.j++
	} else {
		iter.j = 0
		iter.i++
	}
	if iter.i >= iter.maxI {
		return false
	}
	return true
}
func init() {
	rand.Seed(42)
}
func (m *BiLSTMModel) InitializeOutputLayer(hiddenSize int) {
	for i := range m.OutputWeights {
		m.OutputWeights[i] = make([]float64, 2*hiddenSize)
	}
	m.OutputBias = make([]float64, m.OutputSize)
	for i := range m.OutputWeights {
		for j := range m.OutputWeights[i] {
			m.OutputWeights[i][j] = rand.NormFloat64() * 0.01
		}
	}

}

func newLSTMWeights(vocabSize, hiddenSize int) LSTMWeights {
	return LSTMWeights{
		InputWeights:  make2D(4*hiddenSize, vocabSize),
		HiddenWeights: make2D(4*hiddenSize, hiddenSize),
		Bias:          make([]float64, 4*hiddenSize),
	}
}

func make2D(rows, cols int) [][]float64 {
	matrix := make([][]float64, rows)
	for i := range matrix {
		if matrix[i] == nil {
			matrix[i] = make([]float64, cols)
		}
		matrix[i] = make([]float64, cols)
	}
	return matrix
}

func initializeWeights(model *BiLSTMModel) {
	// Initialize LSTM weights with small random values
	for _, weights := range []LSTMWeights{model.ForwardLSTMWeights, model.BackwardLSTMWeights} {
		for i := range weights.InputWeights {
			for j := range weights.InputWeights[i] {
				weights.InputWeights[i][j] = rand.NormFloat64() * 0.01
			}
		}
		for i := range weights.HiddenWeights {
			for j := range weights.HiddenWeights[i] {
				weights.HiddenWeights[i][j] = rand.NormFloat64() * 0.01
			}
		}
	}

	// Initialize output weights
	for i := range model.OutputWeights {
		for j := range model.OutputWeights[i] {
			model.OutputWeights[i][j] = rand.NormFloat64() * 0.1
		}
	}
}

func embed(tokenId int, vocabSize int) []float64 {
	// Placeholder for embedding function. Replace this with a real implementation.

	embedding := make([]float64, vocabSize)
	for i := range embedding {
		embedding[i] = 0.01
	}
	embedding[tokenId%vocabSize] = 1.0
	return embedding
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func tanh(x float64) float64 {
	return math.Tanh(x)
}

func dot(a, b []float64) float64 {
	var result float64
	for i := range a {
		result += a[i] * b[i]
	}
	return result
}

func matVecMul(mat [][]float64, vec []float64) []float64 {
	result := make([]float64, len(mat))
	for i := range mat {
		result[i] = dot(mat[i], vec)
	}
	return result
}

func add(a, b []float64) []float64 {
	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] + b[i]
	}
	return result
}

func vecSigmoid(vec []float64) []float64 {
	result := make([]float64, len(vec))
	for i, v := range vec {
		result[i] = sigmoid(v)
	}
	return result
}

func vecTanh(vec []float64) []float64 {
	result := make([]float64, len(vec))
	for i, v := range vec {
		result[i] = tanh(v)
	}
	return result
}

func vecMul(a, b []float64) []float64 {
	result := make([]float64, len(a))
	for i := range a {
		result[i] = a[i] * b[i]
	}
	return result
}

func (m *BiLSTMModel) lstmStep(t int, tokenIds []int, weights LSTMWeights, state *LSTMState, hidden, cell []float64) ([]float64, []float64) {
	inputEmbedding := embed(tokenIds[t], m.VocabSize)
	lstmInput := matVecMul(weights.InputWeights, inputEmbedding)
	lstmHidden := matVecMul(weights.HiddenWeights, hidden)
	combined := add(add(lstmInput, lstmHidden), weights.Bias)

	i_t := vecSigmoid(combined[:m.HiddenSize])
	f_t := vecSigmoid(combined[m.HiddenSize : 2*m.HiddenSize])
	g_t := vecTanh(combined[2*m.HiddenSize : 3*m.HiddenSize])
	o_t := vecSigmoid(combined[3*m.HiddenSize:])

	cell = add(vecMul(f_t, cell), vecMul(i_t, g_t))
	hidden = vecMul(o_t, vecTanh(cell))

	state.HiddenStates[t] = make([]float64, m.HiddenSize)
	copy(state.HiddenStates[t], hidden)
	state.CellStates[t] = make([]float64, m.HiddenSize)
	copy(state.CellStates[t], cell)
	state.InputGates[t] = i_t
	state.ForgetGates[t] = f_t
	state.OutputGates[t] = o_t
	state.CandidateCells[t] = g_t
	return hidden, cell
}

func (m *BiLSTMModel) Forward(tokenIds []int) [][]float64 {
	sequenceLength := len(tokenIds)
	hiddenStates := make([][]float64, sequenceLength)

	m.ForwardLSTMState = newLSTMState(sequenceLength)
	m.BackwardLSTMState = newLSTMState(sequenceLength)

	forwardHidden := make([]float64, m.HiddenSize)
	forwardCell := make([]float64, m.HiddenSize)
	backwardHidden := make([]float64, m.HiddenSize)
	backwardCell := make([]float64, m.HiddenSize)

	for t := 0; t < sequenceLength; t++ {
		forwardHidden, forwardCell = m.lstmStep(t, tokenIds, m.ForwardLSTMWeights, &m.ForwardLSTMState, forwardHidden, forwardCell)
	}

	for t := sequenceLength - 1; t >= 0; t-- {
		backwardHidden, backwardCell = m.lstmStep(t, tokenIds, m.BackwardLSTMWeights, &m.BackwardLSTMState, backwardHidden, backwardCell)
	}

	for t := 0; t < sequenceLength; t++ {
		hiddenStates[t] = append(m.ForwardLSTMState.HiddenStates[t], m.BackwardLSTMState.HiddenStates[t]...)
	}
	return hiddenStates
}

func (m *BiLSTMModel) computeHiddenStates(tokenIds []int) (forwardHiddenStates, backwardHiddenStates [][]float64) {
	sequenceLength := len(tokenIds)
	forwardHiddenStates = make([][]float64, sequenceLength)
	backwardHiddenStates = make([][]float64, sequenceLength)

	m.ForwardLSTMState = newLSTMState(sequenceLength)
	m.BackwardLSTMState = newLSTMState(sequenceLength)

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		forwardHidden := make([]float64, m.HiddenSize)
		forwardCell := make([]float64, m.HiddenSize)
		for t := 0; t < sequenceLength; t++ {
			forwardHidden, forwardCell = m.lstmStep(t, tokenIds, m.ForwardLSTMWeights, &m.ForwardLSTMState, forwardHidden, forwardCell)
			forwardHiddenStates[t] = make([]float64, m.HiddenSize)
			copy(forwardHiddenStates[t], forwardHidden)
		}
	}()
	go func() {
		defer wg.Done()
		backwardHidden := make([]float64, m.HiddenSize)
		backwardCell := make([]float64, m.HiddenSize)
		for t := sequenceLength - 1; t >= 0; t-- {
			backwardHidden, backwardCell = m.lstmStep(t, tokenIds, m.BackwardLSTMWeights, &m.BackwardLSTMState, backwardHidden, backwardCell)
			backwardHiddenStates[t] = make([]float64, m.HiddenSize)
			copy(backwardHiddenStates[t], backwardHidden)
		}
	}()
	wg.Wait()

	return forwardHiddenStates, backwardHiddenStates
}
func combineHiddenStates(forwardHiddenStates, backwardHiddenStates [][]float64) [][]float64 {
	sequenceLength := len(forwardHiddenStates)
	combinedHiddenStates := make([][]float64, sequenceLength)
	for t := 0; t < sequenceLength; t++ {
		combinedHiddenStates[t] = combineSingleTimeStepHiddenStates(forwardHiddenStates[t], backwardHiddenStates[t])
	}
	return combinedHiddenStates
}

func combineSingleTimeStepHiddenStates(forwardHiddenState, backwardHiddenState []float64) []float64 {
	combined := make([]float64, len(forwardHiddenState)+len(backwardHiddenState))
	copy(combined[:len(forwardHiddenState)], forwardHiddenState)
	copy(combined[len(forwardHiddenState):], backwardHiddenState)
	return combined

}

func (m *BiLSTMModel) ForwardAndCalculateProbabilities(wordIDs []int) [][]float64 {
	hiddenStates := m.Forward(wordIDs)
	return m.CalculateProbabilities(hiddenStates)
}

func (m *BiLSTMModel) CalculateProbabilities(hiddenStates [][]float64) [][]float64 {
	probabilities := make([][]float64, len(hiddenStates))
	for i, hiddenState := range hiddenStates {
		// Add dimension check here
		if len(hiddenState) != 2*m.HiddenSize {
			panic(fmt.Sprintf("Dimension mismatch: len(hiddenState) = %d, expected %d", len(hiddenState), 2*m.HiddenSize))
		}
		output := matVecMul(m.OutputWeights, hiddenState)
		output = add(output, m.OutputBias)
		probabilities[i] = softmax(output)
	}
	return probabilities
}
func (m *BiLSTMModel) Predict(hiddenStates [][]float64) []int {
	predictions := make([]int, len(hiddenStates))
	for i, hiddenState := range hiddenStates {
		// Add dimension check here
		if len(hiddenState) != 2*m.HiddenSize {
			panic(fmt.Sprintf("Dimension mismatch: len(hiddenState) = %d, expected %d", len(hiddenState), 2*m.HiddenSize))
		}
		output := matVecMul(m.OutputWeights, hiddenState)
		output = add(output, m.OutputBias)
		probabilities := softmax(output)
		predictions[i] = argmax(probabilities)
	}

	return predictions
}

func newLSTMState(sequenceLength int) LSTMState {
	return LSTMState{
		HiddenStates:   make([][]float64, sequenceLength),
		CellStates:     make([][]float64, sequenceLength),
		InputGates:     make([][]float64, sequenceLength),
		ForgetGates:    make([][]float64, sequenceLength),
		OutputGates:    make([][]float64, sequenceLength),
		CandidateCells: make([][]float64, sequenceLength),
	}
}

// softmax calculates the softmax of the input vector x with temperature scaling.
func softmax(x []float64, T ...float64) []float64 {
	temperature := 0.5 // Default temperature value
	if len(T) > 0 {
		temperature = T[0]
	}
	exps := make([]float64, len(x))
	sum := 0.0
	for i, val := range x {
		exps[i] = math.Exp(val / temperature)
		sum += exps[i]
	}
	for i := range exps {
		exps[i] /= sum
	}
	return exps
}
func argmax(x []float64) int {
	if len(x) == 0 {
		return -1 // Handle empty slice case
	}
	maxIdx := 0
	maxVal := x[0]
	if len(x) > 1 {
		for _, val := range x {
			if val > maxVal {
				if len(x) == 0 {
					return -1 // Handle empty slice case
				}
				maxIdx = 0
				maxVal := x[0]
				for i, val := range x {
					if val > maxVal {
						maxVal = val
						maxIdx = i
					}
				}
			}
		}
	}
	return maxIdx
}

func (m *BiLSTMModel) Backpropagate(probabilities [][]float64, roleIDs []int, tokenIds []int) {

	sequenceLength := len(probabilities)

	m.resetGradients()

	outputWeightsT := transpose(m.OutputWeights)
	forwardHiddenStates, backwardHiddenStates := m.computeHiddenStates(tokenIds) // [][]float64
	hiddenStates := combineHiddenStates(forwardHiddenStates, backwardHiddenStates)
	for t := sequenceLength - 1; t >= 0; t-- {
		outputErrors := m.calculateOutputErrors(probabilities[t], roleIDs, t)
		m.updateOutputLayerGradients(outputErrors, hiddenStates[t])
		combinedHiddenStateError := matVecMul(outputWeightsT, outputErrors)

		forwardHiddenError := combinedHiddenStateError[:m.HiddenSize]
		backwardHiddenError := combinedHiddenStateError[m.HiddenSize:]

		var wg sync.WaitGroup
		wg.Add(2)
		go func() {
			defer wg.Done()
			m.backpropagateBackwardLSTM(t, tokenIds, backwardHiddenError)
		}()
		go func() {
			defer wg.Done()
			m.backpropagateForwardLSTM(t, tokenIds, forwardHiddenError)
		}()
		wg.Wait()
	}
}

func (m *BiLSTMModel) resetGradients() {
	m.Gradients.OutputWeightGradients = make2D(m.OutputSize, 2*m.HiddenSize)
	m.Gradients.OutputBiasGradients = make([]float64, m.OutputSize)
	m.Gradients.ForwardInputWeightGradients = make2D(len(m.ForwardLSTMWeights.InputWeights), len(m.ForwardLSTMWeights.InputWeights[0]))
	m.Gradients.ForwardHiddenWeightGradients = make2D(len(m.ForwardLSTMWeights.HiddenWeights), len(m.ForwardLSTMWeights.HiddenWeights[0]))
	m.Gradients.ForwardBiasGradients = make([]float64, len(m.ForwardLSTMWeights.Bias))
	m.Gradients.BackwardInputWeightGradients = make2D(len(m.BackwardLSTMWeights.InputWeights), len(m.BackwardLSTMWeights.InputWeights[0]))
	m.Gradients.BackwardHiddenWeightGradients = make2D(len(m.BackwardLSTMWeights.HiddenWeights), len(m.BackwardLSTMWeights.HiddenWeights[0]))
	m.Gradients.BackwardBiasGradients = make([]float64, len(m.BackwardLSTMWeights.Bias))
}

func (m *BiLSTMModel) calculateOutputErrors(probabilities []float64, roleIDs []int, t int) []float64 {
	outputErrors := make([]float64, len(probabilities))
	for i := range probabilities {
		outputErrors[i] = probabilities[i]
		// Treat as "no role" (ID 0) if roleIDs is shorter than t
		var targetRoleID int
		if t < len(roleIDs) {
			targetRoleID = roleIDs[t]
		}
		if i == targetRoleID {
			outputErrors[i] -= 1
		}
	}
	return outputErrors
}

func (m *BiLSTMModel) updateOutputLayerGradients(outputErrors []float64, hiddenState []float64) {
	for i := range m.OutputWeights {
		for j := range hiddenState {
			m.Gradients.OutputWeightGradients[i][j] += outputErrors[i] * hiddenState[j]
		}
		m.Gradients.OutputBiasGradients[i] += outputErrors[i]
	}
}

func (m *BiLSTMModel) backpropagateBackwardLSTM(t int, tokenIds []int, backwardHiddenError []float64) {
	sequenceLength := len(tokenIds)
	backwardCellError := make([]float64, m.HiddenSize)

	var nextCellState float64
	var wg sync.WaitGroup
	wg.Add(m.HiddenSize)
	for i := range backwardHiddenError {
		go func(i int) {
			defer wg.Done()
			if t < sequenceLength-1 {
				nextCellState = m.BackwardLSTMState.CellStates[t+1][i]
			} else {
				nextCellState = 0
			}

			backwardCellError[i] = backwardHiddenError[i] * m.BackwardLSTMState.OutputGates[t][i] * (1 - math.Pow(math.Tanh(nextCellState), 2))
		}(i)
	}
	wg.Wait()

	inputError := make([]float64, m.HiddenSize)
	forgetError := make([]float64, m.HiddenSize)
	outputError := make([]float64, m.HiddenSize)
	candidateError := make([]float64, m.HiddenSize)

	wg.Add(m.HiddenSize)
	for i := range backwardHiddenError {
		go func(i int) {
			defer wg.Done()
			inputError[i], forgetError[i], outputError[i], candidateError[i] = m.calculateBackwardGateErrors(t, backwardHiddenError[i], backwardCellError[i], i)
		}(i)
	}
	wg.Wait()

	wg.Add(m.HiddenSize)

	for i := range backwardHiddenError {
		go func(i int) {
			defer wg.Done()
			m.Gradients.BackwardBiasGradients[i] += inputError[i] + forgetError[i] + outputError[i] + candidateError[i]
		}(i)
	}

	wg.Wait()

	// Precompute gate errors
	weightErrors := []([]float64){inputError, forgetError, outputError, candidateError}

	// Precompute often used data
	inputEmbedding := embed(tokenIds[t], m.VocabSize)
	var previousHiddenState []float64
	if t > 0 {
		previousHiddenState = m.BackwardLSTMState.HiddenStates[t-1]
	}

	// Iterate over each set of weights (input, forget, output, candidate)
	for weightSetIndex, gateErrorSet := range weightErrors {
		// Iterate over each hidden unit's weights in the current set
		for hiddenUnitIndex, weightError := range gateErrorSet {

			if weightSetIndex < len(m.BackwardLSTMWeights.InputWeights)/m.HiddenSize {
				for inputIndex, inputValue := range inputEmbedding {
					m.Gradients.BackwardInputWeightGradients[hiddenUnitIndex+weightSetIndex*m.HiddenSize][inputIndex] += weightError * inputValue
				}
			} else if t > 0 {
				for previousHiddenUnitIndex, previousHiddenValue := range previousHiddenState {
					m.Gradients.BackwardHiddenWeightGradients[hiddenUnitIndex+weightSetIndex*m.HiddenSize][previousHiddenUnitIndex] += weightError * previousHiddenValue
				}
			}
		}
	}

}

func (m *BiLSTMModel) calculateBackwardGateErrors(t int, hiddenError, cellError float64, i int) (inputError, forgetError, outputError, candidateError float64) {
	var prevCellState float64
	if t > 0 {
		prevCellState = m.BackwardLSTMState.CellStates[t-1][i]
	}

	outputError = hiddenError * math.Tanh(m.BackwardLSTMState.CellStates[t][i]) * m.BackwardLSTMState.OutputGates[t][i] * (1 - m.BackwardLSTMState.OutputGates[t][i])
	candidateError = cellError * m.BackwardLSTMState.InputGates[t][i] * (1 - math.Pow(m.BackwardLSTMState.CandidateCells[t][i], 2))
	if t > 0 {

		forgetError = cellError * prevCellState * m.BackwardLSTMState.ForgetGates[t][i] * (1 - m.BackwardLSTMState.ForgetGates[t][i])
	}
	inputError = cellError * m.BackwardLSTMState.CandidateCells[t][i] * m.BackwardLSTMState.InputGates[t][i] * (1 - m.BackwardLSTMState.InputGates[t][i])
	return
}

func (m *BiLSTMModel) backpropagateForwardLSTM(t int, tokenIds []int, forwardHiddenError []float64) {
	sequenceLength := len(tokenIds)
	forwardCellError := make([]float64, m.HiddenSize)

	// Calculate forwardCellError for each element
	for i := range forwardHiddenError {
		forwardCellError[i] = m.calculateForwardCellError(t, forwardHiddenError[i], sequenceLength)
	}

	// Calculate forwardGateForget, forwardGateOutputError, and forwardGateCandidateError for each element
	forwardGateErrors := make([]float64, 4*m.HiddenSize)
	for i := 0; i < m.HiddenSize; i++ {
		forwardGateErrors[i] = m.calculateForwardGateInputError(t, forwardCellError[i], i)
		forwardGateErrors[m.HiddenSize+i] = m.calculateForwardGateForgetError(t, forwardCellError[i], i)
		forwardGateErrors[2*m.HiddenSize+i] = m.calculateForwardGateCandidateError(t, forwardCellError[i], i)
		forwardGateErrors[3*m.HiddenSize+i] = m.calculateForwardGateOutputError(t, forwardHiddenError[i], i)
		m.Gradients.ForwardBiasGradients[i] += forwardGateErrors[i] + forwardGateErrors[m.HiddenSize+i] + forwardGateErrors[2*m.HiddenSize+i] + forwardGateErrors[3*m.HiddenSize+i]
	}

	// Okay, so first, we're getting a special number called inputEmbedding!
	// It's like a secret code for each word. We get it by using the word's ID (tokenId) and a big list of words (VocabSize)!
	inputEmbedding := embed(tokenIds[t], m.VocabSize)
	// Now, we need to see what happened in the last step (if there was one)!
	// That's where the prevHiddenState comes in.
	var prevHiddenState []float64
	// If this is NOT the first step (t > 0), then we grab the prevHiddenState from a list of states!
	if t > 0 {
		prevHiddenState = m.ForwardLSTMState.HiddenStates[t-1]
	}

	// Next up is a weightIterator, it's like a magical helper that helps us go through all the weights!
	weightIterator := NewWeightIterator(m)
	// Now, we loop through each weight, getting two numbers, i and j!
	for weightIterator.Next() {
		i, j := weightIterator.Value()

		// We figure out a special number called linearIndex using i.
		linearIndex := i
		// Here's the cool part! We're updating something called
		// ForwardInputWeightGradients with some calculations. It's like
		// teaching the computer to understand words better!
		// forwardGateErrors are like errors or things that went wrong in our understanding.
		// inputEmbedding is like our special code for the current word.
		// We are telling the computer to understand the word better by looking at the forward gate error!
		m.Gradients.ForwardInputWeightGradients[linearIndex][j] += forwardGateErrors[linearIndex] * inputEmbedding[j]
	}

	for j := range m.ForwardLSTMWeights.HiddenWeights[0] {
		for i := range m.ForwardLSTMWeights.HiddenWeights {
			if t > 0 {
				m.Gradients.ForwardHiddenWeightGradients[i][j] += forwardGateErrors[i] * prevHiddenState[j]
			}
		}
	}
}
func (m *BiLSTMModel) calculateForwardCellError(t int, forwardHiddenError float64, sequenceLength int) float64 {
	if t < sequenceLength-1 {
		return forwardHiddenError * m.ForwardLSTMState.OutputGates[t+1][0] * (1 - math.Pow(math.Tanh(m.ForwardLSTMState.CellStates[t+1][0]), 2))
	}
	return 0
}
func (m *BiLSTMModel) calculateForwardGateInputError(t int, forwardCellError float64, i int) float64 {
	return forwardCellError * m.ForwardLSTMState.CandidateCells[t][i] * m.ForwardLSTMState.InputGates[t][i] * (1 - m.ForwardLSTMState.InputGates[t][i])
}
func (m *BiLSTMModel) calculateForwardGateForgetError(t int, forwardCellError float64, i int) float64 {
	if t > 0 {
		return forwardCellError * m.ForwardLSTMState.CellStates[t-1][i] * m.ForwardLSTMState.ForgetGates[t][i] * (1 - m.ForwardLSTMState.ForgetGates[t][i])
	}
	return 0
}
func (m *BiLSTMModel) calculateForwardGateCandidateError(t int, forwardCellError float64, i int) float64 {
	return forwardCellError * m.ForwardLSTMState.InputGates[t][i] * (1 - math.Pow(m.ForwardLSTMState.CandidateCells[t][i], 2))
}
func (m *BiLSTMModel) calculateForwardGateOutputError(t int, forwardHiddenError float64, i int) float64 {
	return forwardHiddenError * math.Tanh(m.ForwardLSTMState.CellStates[t][i]) * m.ForwardLSTMState.OutputGates[t][i] * (1 - m.ForwardLSTMState.OutputGates[t][i])
}
func (m *BiLSTMModel) UpdateWeights(learningRate float64) {

	// Update output layer weights and biases
	for j := range m.OutputWeights[0] {
		for i := range m.OutputWeights {
			// if i == 0 && j == 0 {
			// 	fmt.Printf("Before update: learningRate=%.6f, gradient=%.6f, weight=%.6f\n", learningRate, m.Gradients.OutputWeightGradients[i][j], m.OutputWeights[i][j])
			// }
			m.OutputWeights[i][j] -= learningRate * m.Gradients.OutputWeightGradients[i][j]
			// if i == 0 && j == 0 {
			// 	fmt.Printf("After update: learningRate=%.6f, gradient=%.6f, weight=%.6f\n", learningRate, m.Gradients.OutputWeightGradients[i][j], m.OutputWeights[i][j])
			// }
		}
	}
	for i := range m.OutputBias {
		m.OutputBias[i] -= learningRate * m.Gradients.OutputBiasGradients[i]
	}

	// Update forward LSTM weights and biases
	for j := range m.ForwardLSTMWeights.InputWeights[0] { // Assuming all rows have the same length
		for i := range m.ForwardLSTMWeights.InputWeights {
			m.ForwardLSTMWeights.InputWeights[i][j] -= learningRate * m.Gradients.ForwardInputWeightGradients[i][j]
		}
	}
	for i := range m.ForwardLSTMWeights.Bias {
		m.ForwardLSTMWeights.Bias[i] -= learningRate * m.Gradients.ForwardBiasGradients[i]
	}

	for j := range m.ForwardLSTMWeights.HiddenWeights[0] { // Assuming all rows have the same length
		for i := range m.ForwardLSTMWeights.HiddenWeights {
			m.ForwardLSTMWeights.HiddenWeights[i][j] -= learningRate * m.Gradients.ForwardHiddenWeightGradients[i][j]
		}
	}

	// Update backward LSTM weights and biases
	for j := range m.BackwardLSTMWeights.InputWeights[0] { // Assuming all rows have the same length
		for i := range m.BackwardLSTMWeights.InputWeights {
			m.BackwardLSTMWeights.InputWeights[i][j] -= learningRate * m.Gradients.BackwardInputWeightGradients[i][j]
		}
	}
	for i := range m.BackwardLSTMWeights.Bias {
		m.BackwardLSTMWeights.Bias[i] -= learningRate * m.Gradients.BackwardBiasGradients[i]
	}

	for j := range m.BackwardLSTMWeights.HiddenWeights[0] { // Assuming all rows have the same length
		for i := range m.BackwardLSTMWeights.HiddenWeights {
			m.BackwardLSTMWeights.HiddenWeights[i][j] -= learningRate * m.Gradients.BackwardHiddenWeightGradients[i][j]
		}
	}
}

func (m *BiLSTMModel) ClipGradients(maxGrad float64) {
	// Clip output layer gradients
	for i := range m.Gradients.OutputWeightGradients {
		for j := range m.Gradients.OutputWeightGradients[i] {
			m.clipGradient(&m.Gradients.OutputWeightGradients[i][j], maxGrad)
		}
		m.clipGradient(&m.Gradients.OutputBiasGradients[i], maxGrad)
	}

	// Clip forward LSTM gradients
	for i := range m.Gradients.ForwardInputWeightGradients {
		for j := range m.Gradients.ForwardInputWeightGradients[i] {
			m.clipGradient(&m.Gradients.ForwardInputWeightGradients[i][j], maxGrad)
		}
		for j := range m.Gradients.ForwardHiddenWeightGradients[i] {
			m.clipGradient(&m.Gradients.ForwardHiddenWeightGradients[i][j], maxGrad)
		}

		m.clipGradient(&m.Gradients.ForwardBiasGradients[i], maxGrad)
	}

	for i := range m.Gradients.BackwardHiddenWeightGradients {
		for j := range m.Gradients.BackwardHiddenWeightGradients[i] {
			m.clipGradient(&m.Gradients.BackwardHiddenWeightGradients[i][j], maxGrad)
		}

	}

	for i := range m.Gradients.BackwardInputWeightGradients {
		for j := range m.Gradients.BackwardInputWeightGradients[i] {
			m.clipGradient(&m.Gradients.BackwardInputWeightGradients[i][j], maxGrad)
		}
		m.clipGradient(&m.Gradients.BackwardBiasGradients[i], maxGrad)

	}
}

func (m *BiLSTMModel) clipGradient(gradient *float64, maxGrad float64) {
	if math.Abs(*gradient) > maxGrad {
		*gradient = math.Copysign(maxGrad, *gradient)
	}
}
func transpose(matrix [][]float64) [][]float64 {
	rows := len(matrix)
	cols := len(matrix[0])
	transposed := make([][]float64, cols)
	for i := range transposed {
		transposed[i] = make([]float64, rows)
		for j := range matrix {
			transposed[i][j] = matrix[j][i]
		}
	}
	return transposed
}
