package bilstm_model

import (
	"fmt"
	"math"
	"math/rand"
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
	initializeWeights(model)
	return model
}
func init() {
	rand.Seed(42)
}
func (m *BiLSTMModel) InitializeOutputLayer(hiddenSize int) {
	for i := range m.OutputWeights {
		m.OutputWeights[i] = make([]float64, 2*hiddenSize)
	}
	m.OutputBias = make([]float64, m.OutputSize)
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
			model.OutputWeights[i][j] = rand.NormFloat64() * 0.01
		}
	}
}

func embed(tokenId int, vocabSize int) []float64 {
	// Placeholder for embedding function. Replace this with a real implementation.
	embedding := make([]float64, vocabSize)
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

func (m *BiLSTMModel) lstmStep(t int, tokenIds []int, weights LSTMWeights, state *LSTMState, hidden, cell []float64, isForward bool) ([]float64, []float64) {
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
		forwardHidden, forwardCell = m.lstmStep(t, tokenIds, m.ForwardLSTMWeights, &m.ForwardLSTMState, forwardHidden, forwardCell, true)
	}

	for t := sequenceLength - 1; t >= 0; t-- {
		backwardHidden, backwardCell = m.lstmStep(t, tokenIds, m.BackwardLSTMWeights, &m.BackwardLSTMState, backwardHidden, backwardCell, false)
	}

	for t := 0; t < sequenceLength; t++ {
		hiddenStates[t] = append(m.ForwardLSTMState.HiddenStates[t], m.BackwardLSTMState.HiddenStates[t]...)
	}
	return hiddenStates
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

func softmax(x []float64) []float64 {
	exps := make([]float64, len(x))
	sum := 0.0
	for i, val := range x {
		exps[i] = math.Exp(val)
		sum += exps[i]
	}
	result := make([]float64, len(x))
	for i, exp := range exps {
		result[i] = exp / sum
	}
	return result
}
func argmax(x []float64) int {
	maxIdx := 0
	for i, val := range x {
		if val > x[maxIdx] {
			maxIdx = i
		}
	}
	return maxIdx
}

func (m *BiLSTMModel) Backpropagate(probabilities [][]float64, roleIDs []int, hiddenStates [][]float64, tokenIds []int) {
	sequenceLength := len(probabilities)
	m.resetGradients()
	outputWeightsT := transpose(m.OutputWeights)

	for t := sequenceLength - 1; t >= 0; t-- {
		outputErrors := m.calculateOutputErrors(probabilities[t], roleIDs, t)
		m.updateOutputLayerGradients(outputErrors, hiddenStates[t])

		combinedHiddenStateError := matVecMul(outputWeightsT, outputErrors)
		forwardHiddenError := combinedHiddenStateError[:m.HiddenSize]
		backwardHiddenError := combinedHiddenStateError[m.HiddenSize:]

		m.backpropagateBackwardLSTM(t, tokenIds, backwardHiddenError)
		m.backpropagateForwardLSTM(t, tokenIds, forwardHiddenError)
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
		if t < len(roleIDs) {
			outputErrors[i] = probabilities[i]
			if i == roleIDs[t] {
				outputErrors[i] -= 1
			}
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

func (m *BiLSTMModel) calculateGateErrors(t int, hiddenError, cellError []float64) (inputError, forgetError, outputError, candidateError []float64) {
	inputError = make([]float64, m.HiddenSize)
	forgetError = make([]float64, m.HiddenSize)
	outputError = make([]float64, m.HiddenSize)
	candidateError = make([]float64, m.HiddenSize)
	for i := range hiddenError {
		var prevCellState float64
		if t > 0 {
			prevCellState = m.BackwardLSTMState.CellStates[t-1][i]
		}

		outputError[i] = hiddenError[i] * math.Tanh(m.BackwardLSTMState.CellStates[t][i]) * m.BackwardLSTMState.OutputGates[t][i] * (1 - m.BackwardLSTMState.OutputGates[t][i])
		candidateError[i] = cellError[i] * m.BackwardLSTMState.InputGates[t][i] * (1 - math.Pow(m.BackwardLSTMState.CandidateCells[t][i], 2))
		if t > 0 {

			forgetError[i] = cellError[i] * prevCellState * m.BackwardLSTMState.ForgetGates[t][i] * (1 - m.BackwardLSTMState.ForgetGates[t][i])
		}
		inputError[i] = cellError[i] * m.BackwardLSTMState.CandidateCells[t][i] * m.BackwardLSTMState.InputGates[t][i] * (1 - m.BackwardLSTMState.InputGates[t][i])

		m.Gradients.BackwardBiasGradients[i] += inputError[i] + forgetError[i] + outputError[i] + candidateError[i]
	}
	return
}

func (m *BiLSTMModel) backpropagateBackwardLSTM(t int, tokenIds []int, backwardHiddenError []float64) {
	sequenceLength := len(tokenIds)
	backwardCellError := make([]float64, m.HiddenSize)

	for i := range backwardHiddenError {
		var nextCellState float64
		if t < sequenceLength-1 {
			nextCellState = m.BackwardLSTMState.CellStates[t+1][i]
		}
		backwardCellError[i] = backwardHiddenError[i] * m.BackwardLSTMState.OutputGates[t][i] * (1 - math.Pow(math.Tanh(nextCellState), 2))
	}
	inputError, forgetError, outputError, candidateError := m.calculateGateErrors(t, backwardHiddenError, backwardCellError)
	inputEmbedding := embed(tokenIds[t], m.VocabSize)
	var prevHiddenState []float64
	if t > 0 {
		prevHiddenState = m.BackwardLSTMState.HiddenStates[t-1]
	}
	weightErrors := [][]float64{inputError, forgetError, outputError, candidateError}

	for i := range m.BackwardLSTMWeights.InputWeights {
		for j := range m.BackwardLSTMWeights.InputWeights[i] {
			m.Gradients.BackwardInputWeightGradients[i][j] += weightErrors[i/m.HiddenSize][i%m.HiddenSize] * inputEmbedding[j]
		}
	}

	for i := range m.BackwardLSTMWeights.HiddenWeights {
		for j := range m.BackwardLSTMWeights.HiddenWeights[i] {
			if t > 0 {
				m.Gradients.BackwardHiddenWeightGradients[i][j] += weightErrors[i/m.HiddenSize][i%m.HiddenSize] * prevHiddenState[j]
			}
		}
	}

}

func (m *BiLSTMModel) backpropagateForwardLSTM(t int, tokenIds []int, forwardHiddenError []float64) {
	sequenceLength := len(tokenIds)
	forwardCellError := make([]float64, m.HiddenSize)
	for i := range forwardHiddenError {
		if t < sequenceLength-1 {
			forwardCellError[i] = forwardHiddenError[i] * m.ForwardLSTMState.OutputGates[t+1][i] * (1 - math.Pow(math.Tanh(m.ForwardLSTMState.CellStates[t+1][i]), 2))
		} else {
			forwardCellError[i] = 0
		}

		var forwardGateForget float64 = 0.0
		if t > 0 {
			forwardGateForget = forwardCellError[i] * m.ForwardLSTMState.CellStates[t-1][i] * m.ForwardLSTMState.ForgetGates[t][i] * (1 - m.ForwardLSTMState.ForgetGates[t][i])
		}
		forwardGateOutputError := forwardHiddenError[i] * math.Tanh(m.ForwardLSTMState.CellStates[t][i]) * m.ForwardLSTMState.OutputGates[t][i] * (1 - m.ForwardLSTMState.OutputGates[t][i])
		forwardGateCandidateError := forwardCellError[i] * m.ForwardLSTMState.InputGates[t][i] * (1 - math.Pow(m.ForwardLSTMState.CandidateCells[t][i], 2))
		m.Gradients.ForwardBiasGradients[i] += forwardGateForget + forwardGateOutputError + forwardGateCandidateError
	}
	for i := range m.ForwardLSTMWeights.InputWeights {
		for j := range m.ForwardLSTMWeights.InputWeights[i] {
			inputEmbedding := embed(tokenIds[t], m.VocabSize)
			var inputError float64
			switch {
			case i < m.HiddenSize:
				inputError = forwardCellError[i] * m.ForwardLSTMState.CandidateCells[t][i] * m.ForwardLSTMState.InputGates[t][i] * (1 - m.ForwardLSTMState.InputGates[t][i])
			case i < 2*m.HiddenSize && t > 0:
				inputError = forwardCellError[i-m.HiddenSize] * m.ForwardLSTMState.CellStates[t-1][i-m.HiddenSize] * m.ForwardLSTMState.ForgetGates[t][i-m.HiddenSize] * (1 - m.ForwardLSTMState.ForgetGates[t][i-m.HiddenSize])
			case i < 3*m.HiddenSize:
				if i-2*m.HiddenSize >= 0 {
					inputError = forwardCellError[i-2*m.HiddenSize] * m.ForwardLSTMState.InputGates[t][i-2*m.HiddenSize] * (1 - math.Pow(m.ForwardLSTMState.CandidateCells[t][i-2*m.HiddenSize], 2)) // Corrected index here
				}
			default:
				inputError = forwardHiddenError[i-3*m.HiddenSize] * math.Tanh(m.ForwardLSTMState.CellStates[t][i-3*m.HiddenSize]) * m.ForwardLSTMState.OutputGates[t][i-3*m.HiddenSize] * (1 - m.ForwardLSTMState.OutputGates[t][i-3*m.HiddenSize])
			}
			m.Gradients.ForwardInputWeightGradients[i][j] += inputError * inputEmbedding[j]
		}
	}
	for i := range m.ForwardLSTMWeights.HiddenWeights {
		for j := range m.ForwardLSTMWeights.HiddenWeights[i] {
			hiddenError := 0.0
			if t > 0 {
				switch {
				case i < m.HiddenSize:
					hiddenError = forwardCellError[i] * m.ForwardLSTMState.CandidateCells[t][i] * m.ForwardLSTMState.InputGates[t][i] * (1 - m.ForwardLSTMState.InputGates[t][i])
				case i < 2*m.HiddenSize:
					hiddenError = forwardCellError[i-m.HiddenSize] * m.ForwardLSTMState.CellStates[t-1][i-m.HiddenSize] * m.ForwardLSTMState.ForgetGates[t][i-m.HiddenSize] * (1 - m.ForwardLSTMState.ForgetGates[t][i-m.HiddenSize])
				case i < 3*m.HiddenSize:
					if i-2*m.HiddenSize >= 0 {
						hiddenError = forwardCellError[i-2*m.HiddenSize] * m.ForwardLSTMState.InputGates[t][i-2*m.HiddenSize] * (1 - math.Pow(m.ForwardLSTMState.CandidateCells[t][i-2*m.HiddenSize], 2))
					}
				default:
					hiddenError = forwardHiddenError[i-3*m.HiddenSize] * math.Tanh(m.ForwardLSTMState.CellStates[t][i-3*m.HiddenSize]) * m.ForwardLSTMState.OutputGates[t][i-3*m.HiddenSize] * (1 - m.ForwardLSTMState.OutputGates[t][i-3*m.HiddenSize])
				}
				m.Gradients.ForwardHiddenWeightGradients[i][j] += hiddenError * m.ForwardLSTMState.HiddenStates[t-1][j]
			}
		}
	}
}

func (m *BiLSTMModel) UpdateWeights(learningRate float64) {
	// Update output layer weights and biases
	for i := range m.OutputWeights {
		for j := range m.OutputWeights[i] {
			m.OutputWeights[i][j] -= learningRate * m.Gradients.OutputWeightGradients[i][j]
		}
		m.OutputBias[i] -= learningRate * m.Gradients.OutputBiasGradients[i]

	}

	// Update forward LSTM weights and biases
	for i := range m.ForwardLSTMWeights.InputWeights {
		for j := range m.ForwardLSTMWeights.InputWeights[i] {
			m.ForwardLSTMWeights.InputWeights[i][j] -= learningRate * m.Gradients.ForwardInputWeightGradients[i][j]
		}
		m.ForwardLSTMWeights.Bias[i] -= learningRate * m.Gradients.ForwardBiasGradients[i]

		for j := range m.ForwardLSTMWeights.HiddenWeights[i] {
			m.ForwardLSTMWeights.HiddenWeights[i][j] -= learningRate * m.Gradients.ForwardHiddenWeightGradients[i][j]
		}
	}

	for i := range m.BackwardLSTMWeights.InputWeights {
		for j := range m.BackwardLSTMWeights.InputWeights[i] {
			m.BackwardLSTMWeights.InputWeights[i][j] -= learningRate * m.Gradients.BackwardInputWeightGradients[i][j]
		}
		m.BackwardLSTMWeights.Bias[i] -= learningRate * m.Gradients.BackwardBiasGradients[i]
	}
	for i := range m.BackwardLSTMWeights.HiddenWeights {
		for j := range m.BackwardLSTMWeights.HiddenWeights[i] {
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
