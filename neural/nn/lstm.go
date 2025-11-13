package nn

import (
	"fmt"
	"log"
	"os"

	. "github.com/zendrulat/nlptagger/neural/tensor"
)

func init() {
	log.SetOutput(os.Stderr)
	log.SetFlags(log.LstdFlags | log.Lshortfile)
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
	InputTensor *Tensor
	PrevHidden  *Tensor
	PrevCell    *Tensor
	ft, it, ct, ot *Tensor
	cct         *Tensor
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
	// gradHt is dL/dht, gradCt is dL/dct from next timestep

	// 1. dL/dot and dL/d(tanh(ct))
	// ht = ot * tanh(ct)
	ct_tanh, err := c.ct.Tanh()
	if err != nil {
		return err
	}
	gradOt, err := gradHt.Mul(ct_tanh)
	if err != nil {
		return err
	}
	grad_ct_tanh, err := gradHt.Mul(c.ot)
	if err != nil {
		return err
	}

	// 2. dL/dct (total)
	// tanh'(x) = 1 - tanh(x)^2
	grad_ct_from_ht, err := grad_ct_tanh.OneMinusSquareTanh(c.ct)
	if err != nil {
		return err
	}
	gradCt, err = gradCt.Add(grad_ct_from_ht)
	if err != nil {
		return err
	}

	// 3. dL/d(prev_c), dL/dft, dL/dit, dL/dcct
	// ct = ft * prev_c + it * cct
	gradPrevCell, err := gradCt.Mul(c.ft)
	if err != nil {
		return err
	}
	gradFt, err := gradCt.Mul(c.PrevCell)
	if err != nil {
		return err
	}
	gradIt, err := gradCt.Mul(c.cct)
	if err != nil {
		return err
	}
	gradCct, err := gradCt.Mul(c.it)
	if err != nil {
		return err
	}

	// 4. Backprop through activations for gates
	// sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
	gradOt_linear, err := gradOt.SigmoidBackward(c.ot)
	if err != nil {
		return err
	}
	gradFt_linear, err := gradFt.SigmoidBackward(c.ft)
	if err != nil {
		return err
	}
	gradIt_linear, err := gradIt.SigmoidBackward(c.it)
	if err != nil {
		return err
	}
	// tanh'(x) = 1 - tanh(x)^2
	gradCct_linear, err := gradCct.OneMinusSquareTanh(c.cct)
	if err != nil {
		return err
	}

	// 5. Gradients for weights and biases
	combined, err := Concat([]*Tensor{c.InputTensor, c.PrevHidden}, 1)
	if err != nil {
		return err
	}
	combinedT, err := combined.Transpose(0, 1)
	if err != nil {
		return err
	}

	gradWf, err := combinedT.MatMul(gradFt_linear)
	if err != nil {
		return err
	}
	gradWi, err := combinedT.MatMul(gradIt_linear)
	if err != nil {
		return err
	}
	gradWc, err := combinedT.MatMul(gradCct_linear)
	if err != nil {
		return err
	}
	gradWo, err := combinedT.MatMul(gradOt_linear)
	if err != nil {
		return err
	}

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

	// 6. Accumulate gradients for weights and biases
	if c.Wf.Grad == nil { c.Wf.Grad = NewTensor(c.Wf.Shape, nil, false) }
	c.Wf.Grad, err = c.Wf.Grad.Add(gradWf)
	if err != nil { return err }

	if c.Wi.Grad == nil { c.Wi.Grad = NewTensor(c.Wi.Shape, nil, false) }
	c.Wi.Grad, err = c.Wi.Grad.Add(gradWi)
	if err != nil { return err }

	if c.Wc.Grad == nil { c.Wc.Grad = NewTensor(c.Wc.Shape, nil, false) }
	c.Wc.Grad, err = c.Wc.Grad.Add(gradWc)
	if err != nil { return err }

	if c.Wo.Grad == nil { c.Wo.Grad = NewTensor(c.Wo.Shape, nil, false) }
	c.Wo.Grad, err = c.Wo.Grad.Add(gradWo)
	if err != nil { return err }

	if c.Bf.Grad == nil { c.Bf.Grad = NewTensor(c.Bf.Shape, nil, false) }
	c.Bf.Grad, err = c.Bf.Grad.Add(gradBf)
	if err != nil { return err }

	if c.Bi.Grad == nil { c.Bi.Grad = NewTensor(c.Bi.Shape, nil, false) }
	c.Bi.Grad, err = c.Bi.Grad.Add(gradBi)
	if err != nil { return err }

	if c.Bc.Grad == nil { c.Bc.Grad = NewTensor(c.Bc.Shape, nil, false) }
	c.Bc.Grad, err = c.Bc.Grad.Add(gradBc)
	if err != nil { return err }

	if c.Bo.Grad == nil { c.Bo.Grad = NewTensor(c.Bo.Shape, nil, false) }
	c.Bo.Grad, err = c.Bo.Grad.Add(gradBo)
	if err != nil { return err }


	// 7. Gradients for combined input
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

	gradCombined, err := gradCombined_f.Add(gradCombined_i)
	if err != nil {
		return err
	}
	gradCombined, err = gradCombined.Add(gradCombined_c)
	if err != nil {
		return err
	}
	gradCombined, err = gradCombined.Add(gradCombined_o)
	if err != nil {
		return err
	}

	// 8. Split gradCombined into gradInput and gradPrevHidden
	gradInput, err := gradCombined.Slice(1, 0, c.InputSize)
	if err != nil {
		return err
	}
	gradPrevHidden, err := gradCombined.Slice(1, c.InputSize, c.InputSize+c.HiddenSize)
	if err != nil {
		return err
	}

	// 9. Accumulate gradients for inputs
	if c.InputTensor.RequiresGrad {
		if c.InputTensor.Grad == nil {
			c.InputTensor.Grad = NewTensor(c.InputTensor.Shape, make([]float64, len(c.InputTensor.Data)), false)
		}
		c.InputTensor.Grad, err = c.InputTensor.Grad.Add(gradInput)
		if err != nil {
			return err
		}
	}
	if c.PrevHidden.RequiresGrad {
		if c.PrevHidden.Grad == nil {
			c.PrevHidden.Grad = NewTensor(c.PrevHidden.Shape, make([]float64, len(c.PrevHidden.Data)), false)
		}
		c.PrevHidden.Grad, err = c.PrevHidden.Grad.Add(gradPrevHidden)
		if err != nil {
			return err
		}
	}
	if c.PrevCell.RequiresGrad {
		if c.PrevCell.Grad == nil {
			c.PrevCell.Grad = NewTensor(c.PrevCell.Shape, make([]float64, len(c.PrevCell.Data)), false)
		}
		c.PrevCell.Grad, err = c.PrevCell.Grad.Add(gradPrevCell)
		if err != nil {
			return err
		}
	}

	return nil
}

// LSTM represents a multi-layer LSTM.
type LSTM struct {
	InputSize  int
	HiddenSize int
	NumLayers  int
	Cells      [][]*LSTMCell
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
	var layerOutput *Tensor = input                                    // Input to the first layer

	for i := 0; i < l.NumLayers; i++ {
		// For the first layer, layerInput is the original input.
		// For subsequent layers, layerInput is the output (ht) of the previous layer.
		if i > 0 {
			layerOutput = currentHidden
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

// Backward performs the backward pass for the entire LSTM layer.
func (l *LSTM) Backward(gradNextHidden, gradNextCell *Tensor) error {
	gradH := gradNextHidden
	gradC := gradNextCell
	var err error

	for i := l.NumLayers - 1; i >= 0; i-- {
		cell := l.Cells[i][0] // Assuming one cell per layer

		// The backward pass for the cell computes gradients for its inputs.
		err = cell.Backward(gradH, gradC)
		if err != nil {
			return fmt.Errorf("failed to backpropagate through LSTM cell in layer %d: %w", i, err)
		}

		// The input to this layer (for i>0) was the hidden state of the previous layer.
		// The gradient for the output of the previous layer is the sum of the gradients
		// for the 'input' and 'prevHidden' of this layer's cell.
		if i > 0 {
			// cell.InputTensor.Grad is grad for layerOutput
			// cell.PrevHidden.Grad is grad for currentHidden
			// cell.PrevCell.Grad is grad for currentCell
			gradH, err = cell.InputTensor.Grad.Add(cell.PrevHidden.Grad)
			if err != nil {
				return err
			}
			gradC = cell.PrevCell.Grad
		}
	}
	return nil
}
