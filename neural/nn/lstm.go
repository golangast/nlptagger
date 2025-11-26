package nn

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"

	. "nlptagger/neural/tensor"
)

func init() {
	log.SetOutput(os.Stderr)
	log.SetFlags(log.LstdFlags | log.Lshortfile)
}

// applyDropout applies dropout to a tensor during training.
// During training, randomly sets dropoutRate fraction of values to 0 and scales remaining by 1/(1-dropoutRate).
// During inference (training=false), returns the tensor unchanged.
func applyDropout(tensor *Tensor, dropoutRate float64, training bool) *Tensor {
	if !training || dropoutRate == 0.0 {
		return tensor
	}

	// Create dropout mask
	mask := NewTensor(tensor.Shape, make([]float64, len(tensor.Data)), false)
	scale := 1.0 / (1.0 - dropoutRate)

	for i := range mask.Data {
		if rand.Float64() < dropoutRate {
			mask.Data[i] = 0.0
		} else {
			mask.Data[i] = scale
		}
	}

	// Apply mask
	output := NewTensor(tensor.Shape, make([]float64, len(tensor.Data)), tensor.RequiresGrad)
	for i := range output.Data {
		output.Data[i] = tensor.Data[i] * mask.Data[i]
	}

	return output
}

// LSTMCell represents a single LSTM cell.
type LSTMCell struct {
	InputSize  int
	HiddenSize int

	// Weight matrices
	Wf, Wi, Wc, Wo *Tensor
	// Bias vectors
	Bf, Bi, Bc, Bo *Tensor

	// Stored for backward pass
	InputTensor    *Tensor
	PrevHidden     *Tensor
	PrevCell       *Tensor
	ft, it, ct, ot *Tensor
	cct            *Tensor
}

// NewLSTMCell creates a new LSTMCell.
func NewLSTMCell(inputSize, hiddenSize int) (*LSTMCell, error) {
	// Initialize weights
	wf, err := NewLinear(inputSize+hiddenSize, hiddenSize)
	if err != nil {
		return nil, err
	}
	wi, err := NewLinear(inputSize+hiddenSize, hiddenSize)
	if err != nil {
		return nil, err
	}
	wc, err := NewLinear(inputSize+hiddenSize, hiddenSize)
	if err != nil {
		return nil, err
	}
	wo, err := NewLinear(inputSize+hiddenSize, hiddenSize)
	if err != nil {
		return nil, err
	}

	return &LSTMCell{
		InputSize:  inputSize,
		HiddenSize: hiddenSize,
		Wf:         wf.Weights,
		Wi:         wi.Weights,
		Wc:         wc.Weights,
		Wo:         wo.Weights,
		Bf:         wf.Biases,
		Bi:         wi.Biases,
		Bc:         wc.Biases,
		Bo:         wo.Biases,
	}, nil
}

// Parameters returns all learnable parameters of the LSTMCell.
func (c *LSTMCell) Parameters() []*Tensor {
	return []*Tensor{c.Wf, c.Wi, c.Wc, c.Wo, c.Bf, c.Bi, c.Bc, c.Bo}
}

// Forward performs the forward pass of the LSTMCell.
func (c *LSTMCell) Forward(inputs ...*Tensor) (*Tensor, *Tensor, error) {
	if len(inputs) != 3 {
		return nil, nil, fmt.Errorf("LSTMCell.Forward expects 3 inputs (input, prevHidden, prevCell), got %d", len(inputs))
	}
	input, prevHidden, prevCell := inputs[0], inputs[1], inputs[2]

	// Store inputs for backward pass
	c.InputTensor = input
	c.PrevHidden = prevHidden
	c.PrevCell = prevCell

	// Concatenate input and previous hidden state
	combined, err := Concat([]*Tensor{input, prevHidden}, 1)
	if err != nil {
		return nil, nil, err
	}

	// Forget gate
	ft, err := combined.MatMul(c.Wf)
	if err != nil {
		return nil, nil, err
	}
	ft, err = ft.AddWithBroadcast(c.Bf)
	if err != nil {
		return nil, nil, err
	}
	ft, err = ft.Sigmoid()
	if err != nil {
		return nil, nil, err
	}
	c.ft = ft

	// Input gate
	it, err := combined.MatMul(c.Wi)
	if err != nil {
		return nil, nil, err
	}
	it, err = it.AddWithBroadcast(c.Bi)
	if err != nil {
		return nil, nil, err
	}
	it, err = it.Sigmoid()
	if err != nil {
		return nil, nil, err
	}
	c.it = it

	// Candidate cell state
	cct, err := combined.MatMul(c.Wc)
	if err != nil {
		return nil, nil, err
	}
	cct, err = cct.AddWithBroadcast(c.Bc)
	if err != nil {
		return nil, nil, err
	}
	cct, err = cct.Tanh()
	if err != nil {
		return nil, nil, err
	}
	c.cct = cct

	// New cell state
	ct, err := ft.Mul(prevCell)
	if err != nil {
		return nil, nil, err
	}
	it_cct, err := it.Mul(cct)
	if err != nil {
		return nil, nil, err
	}
	ct, err = ct.Add(it_cct)
	if err != nil {
		return nil, nil, err
	}
	c.ct = ct

	// Output gate
	ot, err := combined.MatMul(c.Wo)
	if err != nil {
		return nil, nil, err
	}
	ot, err = ot.AddWithBroadcast(c.Bo)
	if err != nil {
		return nil, nil, err
	}
	ot, err = ot.Sigmoid()
	if err != nil {
		return nil, nil, err
	}
	c.ot = ot

	// New hidden state
	ct_tanh, err := ct.Tanh()
	if err != nil {
		return nil, nil, fmt.Errorf("LSTMCell.Forward: Tanh operation failed: %w", err)
	}
	ht, err := ot.Mul(ct_tanh)
	if err != nil {
		return nil, nil, fmt.Errorf("LSTMCell.Forward: Mul operation failed for hidden state: %w", err)
	}

	return ht, ct, nil
}

// Backward performs the backward pass for the LSTMCell.
func (c *LSTMCell) Backward(gradHt, gradCt *Tensor) error {
	// Gradients for outputs
	gradOt := NewTensor(c.ot.Shape, make([]float64, len(c.ot.Data)), false)
	gradCtTanh := NewTensor(c.ct.Shape, make([]float64, len(c.ct.Data)), false)
	for i := range gradHt.Data {
		ct_tanh_val, _ := c.ct.Tanh()
		gradOt.Data[i] = gradHt.Data[i] * ct_tanh_val.Data[i]
		gradCtTanh.Data[i] = gradHt.Data[i] * c.ot.Data[i]
	}

	// Gradient for ct
	gradCt_from_ht := NewTensor(c.ct.Shape, make([]float64, len(c.ct.Data)), false)
	for i := range gradCtTanh.Data {
		ct_tanh_val, _ := c.ct.Tanh()
		gradCt_from_ht.Data[i] = gradCtTanh.Data[i] * (1 - math.Pow(ct_tanh_val.Data[i], 2))
	}
	gradCt.Add(gradCt_from_ht)

	// Gradients for gates
	gradFt := NewTensor(c.ft.Shape, make([]float64, len(c.ft.Data)), false)
	gradIt := NewTensor(c.it.Shape, make([]float64, len(c.it.Data)), false)
	gradCct := NewTensor(c.cct.Shape, make([]float64, len(c.cct.Data)), false)
	for i := range gradCt.Data {
		gradFt.Data[i] = gradCt.Data[i] * c.PrevCell.Data[i]
		gradIt.Data[i] = gradCt.Data[i] * c.cct.Data[i]
		gradCct.Data[i] = gradCt.Data[i] * c.it.Data[i]
	}

	// Backprop through activations
	gradOt_linear := NewTensor(gradOt.Shape, make([]float64, len(gradOt.Data)), false)
	for i := range gradOt.Data {
		gradOt_linear.Data[i] = gradOt.Data[i] * c.ot.Data[i] * (1 - c.ot.Data[i])
	}
	gradFt_linear := NewTensor(gradFt.Shape, make([]float64, len(gradFt.Data)), false)
	for i := range gradFt.Data {
		gradFt_linear.Data[i] = gradFt.Data[i] * c.ft.Data[i] * (1 - c.ft.Data[i])
	}
	gradIt_linear := NewTensor(gradIt.Shape, make([]float64, len(gradIt.Data)), false)
	for i := range gradIt.Data {
		gradIt_linear.Data[i] = gradIt.Data[i] * c.it.Data[i] * (1 - c.it.Data[i])
	}
	gradCct_linear := NewTensor(gradCct.Shape, make([]float64, len(gradCct.Data)), false)
	for i := range gradCct.Data {
		gradCct_linear.Data[i] = gradCct.Data[i] * (1 - math.Pow(c.cct.Data[i], 2))
	}

	// Gradients for weights and biases
	combined, _ := Concat([]*Tensor{c.InputTensor, c.PrevHidden}, 1)
	combinedT, _ := combined.Transpose(0, 1)

	gradWf, _ := combinedT.MatMul(gradFt_linear)
	gradWi, _ := combinedT.MatMul(gradIt_linear)
	gradWc, _ := combinedT.MatMul(gradCct_linear)
	gradWo, _ := combinedT.MatMul(gradOt_linear)

	gradBf, err := gradFt_linear.Sum(0)
	if err != nil {
		return err
	}
	gradBi, err := gradIt_linear.Sum(0)
	if err != nil {
		return err
	}
	gradBc, err := gradCct_linear.Sum(0)
	if err != nil {
		return err
	}
	gradBo, err := gradOt_linear.Sum(0)
	if err != nil {
		return err
	}

	// Accumulate gradients
	c.Wf.Grad.Add(gradWf)
	c.Wi.Grad.Add(gradWi)
	c.Wc.Grad.Add(gradWc)
	c.Wo.Grad.Add(gradWo)
	c.Bf.Grad.Add(gradBf)
	c.Bi.Grad.Add(gradBi)
	c.Bc.Grad.Add(gradBc)
	c.Bo.Grad.Add(gradBo)

	// Gradients for inputs
	transposedWf, err := c.Wf.Transpose(0, 1)
	if err != nil {
		return err
	}
	gradCombined_f, err := gradFt_linear.MatMul(transposedWf)
	if err != nil {
		return err
	}

	transposedWi, err := c.Wi.Transpose(0, 1)
	if err != nil {
		return err
	}
	gradCombined_i, err := gradIt_linear.MatMul(transposedWi)
	if err != nil {
		return err
	}

	transposedWc, err := c.Wc.Transpose(0, 1)
	if err != nil {
		return err
	}
	gradCombined_c, err := gradCct_linear.MatMul(transposedWc)
	if err != nil {
		return err
	}

	transposedWo, err := c.Wo.Transpose(0, 1)
	if err != nil {
		return err
	}
	gradCombined_o, err := gradOt_linear.MatMul(transposedWo)
	if err != nil {
		return err
	}

	gradCombined, _ := gradCombined_f.Add(gradCombined_i)
	gradCombined, _ = gradCombined.Add(gradCombined_c)
	gradCombined, _ = gradCombined.Add(gradCombined_o)

	// Split gradCombined into gradInput and gradPrevHidden
	gradInput, _ := gradCombined.Slice(1, 0, c.InputSize)
	gradPrevHidden, _ := gradCombined.Slice(1, c.InputSize, c.InputSize+c.HiddenSize)

	// Gradient for prevCell
	gradPrevCell, _ := gradCt.Mul(c.ft)

	// Accumulate gradients for inputs
	if c.InputTensor.RequiresGrad {
		if c.InputTensor.Grad == nil {
			c.InputTensor.Grad = NewTensor(c.InputTensor.Shape, make([]float64, len(c.InputTensor.Data)), false)
		}
		for i := range c.InputTensor.Grad.Data {
			c.InputTensor.Grad.Data[i] += gradInput.Data[i]
		}
	}
	if c.PrevHidden.RequiresGrad {
		if c.PrevHidden.Grad == nil {
			c.PrevHidden.Grad = NewTensor(c.PrevHidden.Shape, make([]float64, len(c.PrevHidden.Data)), false)
		}
		for i := range c.PrevHidden.Grad.Data {
			c.PrevHidden.Grad.Data[i] += gradPrevHidden.Data[i]
		}
	}
	if c.PrevCell.RequiresGrad {
		if c.PrevCell.Grad == nil {
			c.PrevCell.Grad = NewTensor(c.PrevCell.Shape, make([]float64, len(c.PrevCell.Data)), false)
		}
		for i := range c.PrevCell.Grad.Data {
			c.PrevCell.Grad.Data[i] += gradPrevCell.Data[i]
		}
	}

	return nil
}

// LSTM represents a multi-layer LSTM.
type LSTM struct {
	InputSize   int
	HiddenSize  int
	NumLayers   int
	Cells       [][]*LSTMCell
	DropoutRate float64 // Dropout rate between layers (0.0 = no dropout)
	Training    bool    // Whether model is in training mode (dropout active)
}

// NewLSTM creates a new LSTM.
func NewLSTM(inputSize, hiddenSize, numLayers int) (*LSTM, error) {
	cells := make([][]*LSTMCell, numLayers)
	for i := 0; i < numLayers; i++ {
		layerInputSize := inputSize
		if i > 0 {
			layerInputSize = hiddenSize
		}
		cells[i] = make([]*LSTMCell, 1) // Assuming single cell per layer for now
		cell, err := NewLSTMCell(layerInputSize, hiddenSize)
		if err != nil {
			return nil, err
		}
		cells[i][0] = cell
	}
	return &LSTM{
		InputSize:  inputSize,
		HiddenSize: hiddenSize,
		NumLayers:  numLayers,
		Cells:      cells,
	}, nil
}

// Parameters returns all learnable parameters of the LSTM.
func (l *LSTM) Parameters() []*Tensor {
	params := []*Tensor{}
	for _, layer := range l.Cells {
		for _, cell := range layer {
			params = append(params, cell.Parameters()...)
		}
	}
	return params
}

// Forward performs the forward pass of the LSTM.
func (l *LSTM) Forward(inputs ...*Tensor) (*Tensor, *Tensor, error) {
	if len(inputs) != 3 {
		return nil, nil, fmt.Errorf("LSTM.Forward expects 3 inputs (input, prevHidden, prevCell), got %d", len(inputs))
	}
	input, initialHidden, initialCell := inputs[0], inputs[1], inputs[2] // Renamed for clarity

	var currentHidden, currentCell *Tensor = initialHidden, initialCell // Initialize with initial states
	var layerOutput *Tensor = input                                     // Input to the first layer

	for i := 0; i < l.NumLayers; i++ {
		// For the first layer, layerInput is the original input.
		// For subsequent layers, layerInput is the output (ht) of the previous layer.
		if i > 0 {
			layerOutput = currentHidden
			// Apply dropout between layers (not on the last layer output)
			if i < l.NumLayers {
				layerOutput = applyDropout(layerOutput, l.DropoutRate, l.Training)
			}
		}

		ht, ct, err := l.Cells[i][0].Forward(layerOutput, currentHidden, currentCell)
		if err != nil {
			log.Printf("LSTMCell.Forward in LSTM.Forward failed: %+v", err)
			return nil, nil, err
		}
		currentHidden = ht
		currentCell = ct
	}
	return currentHidden, currentCell, nil // Return the final hidden and cell states
}

// Backward performs the backward pass for the LSTM.
func (l *LSTM) Backward(gradHt, gradCt *Tensor) error {
	for i := l.NumLayers - 1; i >= 0; i-- {
		err := l.Cells[i][0].Backward(gradHt, gradCt)
		if err != nil {
			return err
		}
		// Gradients for the previous layer's hidden and cell states
		// are now in the Grad fields of the input tensors of the current layer's cell.
		gradHt = l.Cells[i][0].PrevHidden.Grad
		gradCt = l.Cells[i][0].PrevCell.Grad
	}
	return nil
}
