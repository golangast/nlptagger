package nn

import (
	"fmt"
	"math"
	"math/rand"

	. "nlptagger/neural/tensor"
)

// LSTM represents a Long Short-Term Memory layer.
type LSTM struct {
	// Weights for input gate, forget gate, cell gate, and output gate
	W_ii, W_hi *Tensor // Input gate weights (input, hidden)
	W_if, W_hf *Tensor // Forget gate weights (input, hidden)
	W_ig, W_hg *Tensor // Cell gate weights (input, hidden)
	W_io, W_ho *Tensor // Output gate weights (input, hidden)

	// Biases for input gate, forget gate, cell gate, and output gate
	b_i, b_f, b_g, b_o *Tensor

	InputSize  int
	HiddenSize int

	// Stored intermediate values for backward pass
	inputTensor *Tensor
	prevHidden  *Tensor
	prevCell    *Tensor
	// Gate activations and cell states for backward pass
	i_t, f_t, g_t, o_t *Tensor // Gate activations
	cell_t_candidate   *Tensor // Candidate cell state
	cell_t             *Tensor // Current cell state
	hidden_t           *Tensor // Current hidden state
}

// NewLSTM creates a new LSTM layer.
func NewLSTM(inputSize, hiddenSize int) *LSTM {
	// He initialization for weights
	stdDev_i := math.Sqrt(2.0 / float64(inputSize))
	stdDev_h := math.Sqrt(2.0 / float64(hiddenSize))

	// Initialize weights and biases
	// Input gate
	W_ii := NewTensor([]int{inputSize, hiddenSize}, randomData(inputSize*hiddenSize, stdDev_i), true)
	W_hi := NewTensor([]int{hiddenSize, hiddenSize}, randomData(hiddenSize*hiddenSize, stdDev_h), true)
	b_i := NewTensor([]int{hiddenSize}, make([]float64, hiddenSize), true)

	// Forget gate
	W_if := NewTensor([]int{inputSize, hiddenSize}, randomData(inputSize*hiddenSize, stdDev_i), true)
	W_hf := NewTensor([]int{hiddenSize, hiddenSize}, randomData(hiddenSize*hiddenSize, stdDev_h), true)
	b_f := NewTensor([]int{hiddenSize}, make([]float64, hiddenSize), true)

	// Cell gate
	W_ig := NewTensor([]int{inputSize, hiddenSize}, randomData(inputSize*hiddenSize, stdDev_i), true)
	W_hg := NewTensor([]int{hiddenSize, hiddenSize}, randomData(hiddenSize*hiddenSize, stdDev_h), true)
	b_g := NewTensor([]int{hiddenSize}, make([]float64, hiddenSize), true)

	// Output gate
	W_io := NewTensor([]int{inputSize, hiddenSize}, randomData(inputSize*hiddenSize, stdDev_i), true)
	W_ho := NewTensor([]int{hiddenSize, hiddenSize}, randomData(hiddenSize*hiddenSize, stdDev_h), true)
	b_o := NewTensor([]int{hiddenSize}, make([]float64, hiddenSize), true)

	return &LSTM{
		W_ii: W_ii, W_hi: W_hi, b_i: b_i,
		W_if: W_if, W_hf: W_hf, b_f: b_f,
		W_ig: W_ig, W_hg: W_hg, b_g: b_g,
		W_io: W_io, W_ho: W_ho, b_o: b_o,
		InputSize:  inputSize,
		HiddenSize: hiddenSize,
	}
}

// randomData generates a slice of random float64 values with a given standard deviation.
func randomData(size int, stdDev float64) []float64 {
	data := make([]float64, size)
	for i := range data {
		data[i] = rand.NormFloat64() * stdDev
	}
	return data
}

// Parameters returns all learnable parameters of the LSTM layer.
func (l *LSTM) Parameters() []*Tensor {
	return []*Tensor{
		l.W_ii, l.W_hi, l.b_i,
		l.W_if, l.W_hf, l.b_f,
		l.W_ig, l.W_hg, l.b_g,
		l.W_io, l.W_ho, l.b_o,
	}
}

// Forward performs the forward pass of the LSTM layer.
// input: [batch_size, sequence_length, input_size]
// initialHidden: [batch_size, hidden_size] (optional, if nil, zeros are used)
// initialCell: [batch_size, hidden_size] (optional, if nil, zeros are used)
// Returns: output [batch_size, sequence_length, hidden_size], finalHidden [batch_size, hidden_size], finalCell [batch_size, hidden_size]
func (l *LSTM) Forward(input *Tensor, initialHidden, initialCell *Tensor) (*Tensor, *Tensor, *Tensor, error) {
	if len(input.Shape) != 3 {
		return nil, nil, nil, fmt.Errorf("LSTM input must be 3D [batch_size, sequence_length, input_size], got %v", input.Shape)
	}
	batchSize := input.Shape[0]
	seqLength := input.Shape[1]
	inputSize := input.Shape[2]

	if inputSize != l.InputSize {
		return nil, nil, nil, fmt.Errorf("LSTM input_size mismatch: expected %d, got %d", l.InputSize, inputSize)
	}

	l.inputTensor = input // Store input for backward pass

	// Initialize hidden and cell states if not provided
	h := initialHidden
	c := initialCell
	if h == nil {
		h = NewTensor([]int{batchSize, l.HiddenSize}, make([]float64, batchSize*l.HiddenSize), true)
	}
	if c == nil {
		c = NewTensor([]int{batchSize, l.HiddenSize}, make([]float64, batchSize*l.HiddenSize), true)
	}
	l.prevHidden = h // Store initial hidden for backward
	l.prevCell = c   // Store initial cell for backward

	// Tensors to store intermediate gate activations and cell states for backward pass
	l.i_t = NewTensor([]int{seqLength, batchSize, l.HiddenSize}, nil, true)
	l.f_t = NewTensor([]int{seqLength, batchSize, l.HiddenSize}, nil, true)
	l.g_t = NewTensor([]int{seqLength, batchSize, l.HiddenSize}, nil, true)
	l.o_t = NewTensor([]int{seqLength, batchSize, l.HiddenSize}, nil, true)
	l.cell_t_candidate = NewTensor([]int{seqLength, batchSize, l.HiddenSize}, nil, true)
	l.cell_t = NewTensor([]int{seqLength, batchSize, l.HiddenSize}, nil, true)
	l.hidden_t = NewTensor([]int{seqLength, batchSize, l.HiddenSize}, nil, true)

	outputs := NewTensor([]int{batchSize, seqLength, l.HiddenSize}, nil, true)

	for t := 0; t < seqLength; t++ {
		// Extract current input slice: input[:, t, :]
		input_t_data := make([]float64, batchSize*inputSize)
		for b := 0; b < batchSize; b++ {
			copy(input_t_data[b*inputSize:(b+1)*inputSize], input.Data[b*seqLength*inputSize+t*inputSize:b*seqLength*inputSize+(t+1)*inputSize])
		}
		input_t := NewTensor([]int{batchSize, inputSize}, input_t_data, true)

		// Input gate: i_t = sigmoid(W_ii * input_t + b_i + W_hi * h_{t-1} + b_i)
		lin_ii, err := input_t.MatMul(l.W_ii)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("LSTM W_ii matmul failed: %w", err)
		}
		lin_ii, err = lin_ii.AddWithBroadcast(l.b_i)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("LSTM b_i add failed: %w", err)
		}
		lin_hi, err := h.MatMul(l.W_hi)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("LSTM W_hi matmul failed: %w", err)
		}
		i_t_pre_act, err := lin_ii.Add(lin_hi)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("LSTM i_t pre-activation add failed: %w", err)
		}
		i_t, err := i_t_pre_act.Sigmoid()
		if err != nil {
			return nil, nil, nil, fmt.Errorf("LSTM i_t sigmoid failed: %w", err)
		}
		l.i_t.SetSlice(0, t, i_t) // Store for backward

		// Forget gate: f_t = sigmoid(W_if * input_t + b_f + W_hf * h_{t-1} + b_f)
		lin_if, err := input_t.MatMul(l.W_if)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("LSTM W_if matmul failed: %w", err)
		}
		lin_if, err = lin_if.AddWithBroadcast(l.b_f)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("LSTM b_f add failed: %w", err)
		}
		lin_hf, err := h.MatMul(l.W_hf)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("LSTM W_hf matmul failed: %w", err)
		}
		f_t_pre_act, err := lin_if.Add(lin_hf)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("LSTM f_t pre-activation add failed: %w", err)
		}
		f_t, err := f_t_pre_act.Sigmoid()
		if err != nil {
			return nil, nil, nil, fmt.Errorf("LSTM f_t sigmoid failed: %w", err)
		}
		l.f_t.SetSlice(0, t, f_t) // Store for backward

		// Cell gate: g_t = tanh(W_ig * input_t + b_g + W_hg * h_{t-1} + b_g)
		lin_ig, err := input_t.MatMul(l.W_ig)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("LSTM W_ig matmul failed: %w", err)
		}
		lin_ig, err = lin_ig.AddWithBroadcast(l.b_g)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("LSTM b_g add failed: %w", err)
		}
		lin_hg, err := h.MatMul(l.W_hg)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("LSTM W_hg matmul failed: %w", err)
		}
		g_t_pre_act, err := lin_ig.Add(lin_hg)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("LSTM g_t pre-activation add failed: %w", err)
		}
		g_t, err := g_t_pre_act.Tanh()
		if err != nil {
			return nil, nil, nil, fmt.Errorf("LSTM g_t tanh failed: %w", err)
		}
		l.g_t.SetSlice(0, t, g_t) // Store for backward

		// Output gate: o_t = sigmoid(W_io * input_t + b_o + W_ho * h_{t-1} + b_o)
		lin_io, err := input_t.MatMul(l.W_io)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("LSTM W_io matmul failed: %w", err)
		}
		lin_io, err = lin_io.AddWithBroadcast(l.b_o)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("LSTM b_o add failed: %w", err)
		}
		lin_ho, err := h.MatMul(l.W_ho)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("LSTM W_ho matmul failed: %w", err)
		}
		o_t_pre_act, err := lin_io.Add(lin_ho)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("LSTM o_t pre-activation add failed: %w", err)
		}
		o_t, err := o_t_pre_act.Sigmoid()
		if err != nil {
			return nil, nil, nil, fmt.Errorf("LSTM o_t sigmoid failed: %w", err)
		}
		l.o_t.SetSlice(0, t, o_t) // Store for backward

		// Cell state: c_t = f_t * c_{t-1} + i_t * g_t
		term1, err := f_t.Mul(c)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("LSTM f_t mul c failed: %w", err)
		}
		term2, err := i_t.Mul(g_t)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("LSTM i_t mul g_t failed: %w", err)
		}
		c_t, err := term1.Add(term2)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("LSTM c_t add failed: %w", err)
		}
		l.cell_t.SetSlice(0, t, c_t) // Store for backward

		// Hidden state: h_t = o_t * tanh(c_t)
		c_t_tanh, err := c_t.Tanh()
		if err != nil {
			return nil, nil, nil, fmt.Errorf("LSTM c_t tanh failed: %w", err)
		}
		h_t, err := o_t.Mul(c_t_tanh)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("LSTM o_t mul c_t_tanh failed: %w", err)
		}
		l.hidden_t.SetSlice(0, t, h_t) // Store for backward

		// Update for next time step
	h = h_t
	c = c_t

		// Store output for this time step
		outputs.SetSlice(1, t, h_t) // outputs[batch_size, seq_length, hidden_size]
	}

	return outputs, h, c, nil
}

// Backward performs the backward pass for the LSTM layer.
// gradOutput: Gradient from the next layer w.r.t. the output (h_t) of this LSTM.
// gradHidden: Gradient from the next layer w.r.t. the final hidden state (h_final) of this LSTM.
// gradCell: Gradient from the next layer w.r.t. the final cell state (c_final) of this LSTM.
func (l *LSTM) Backward(gradOutput, gradHidden, gradCell *Tensor) error {
	batchSize := l.inputTensor.Shape[0]
	seqLength := l.inputTensor.Shape[1]
	inputSize := l.inputTensor.Shape[2]

	// Initialize gradients for parameters
	params := l.Parameters()
	for _, p := range params {
		if p.RequiresGrad && p.Grad == nil {
			p.Grad = NewTensor(p.Shape, make([]float64, len(p.Data)), false)
		}
	}

	// Initialize gradients for input, prevHidden, prevCell
	if l.inputTensor.RequiresGrad && l.inputTensor.Grad == nil {
		l.inputTensor.Grad = NewTensor(l.inputTensor.Shape, make([]float64, len(l.inputTensor.Data)), false)
	}
	if l.prevHidden.RequiresGrad && l.prevHidden.Grad == nil {
		l.prevHidden.Grad = NewTensor(l.prevHidden.Shape, make([]float64, len(l.prevHidden.Data)), false)
	}
	if l.prevCell.RequiresGrad && l.prevCell.Grad == nil {
		l.prevCell.Grad = NewTensor(l.prevCell.Shape, make([]float64, len(l.prevCell.Data)), false)
	}

	// Initialize dh_next and dc_next with gradients from the final hidden/cell states
	dh_next := gradHidden
	dc_next := gradCell

	// Loop backward through the sequence
	for t := seqLength - 1; t >= 0; t-- {
		// Get stored intermediate values for current time step t
		input_t_data := make([]float64, batchSize*inputSize)
		for b := 0; b < batchSize; b++ {
			copy(input_t_data[b*inputSize:(b+1)*inputSize], l.inputTensor.Data[b*seqLength*inputSize+t*inputSize:b*seqLength*inputSize+(t+1)*inputSize])
		}
		input_t := NewTensor([]int{batchSize, inputSize}, input_t_data, true)

		h_t, err := l.hidden_t.GetSlice(0, t)
		if err != nil {
			return fmt.Errorf("LSTM backward GetSlice hidden_t failed: %w", err)
		}
		c_t, err := l.cell_t.GetSlice(0, t)
		if err != nil {
			return fmt.Errorf("LSTM backward GetSlice cell_t failed: %w", err)
		}
		
		i_t, err := l.i_t.GetSlice(0, t)
		if err != nil {
			return fmt.Errorf("LSTM backward GetSlice i_t failed: %w", err)
		}
		f_t, err := l.f_t.GetSlice(0, t)
		if err != nil {
			return fmt.Errorf("LSTM backward GetSlice f_t failed: %w", err)
		}
		g_t, err := l.g_t.GetSlice(0, t)
		if err != nil {
			return fmt.Errorf("LSTM backward GetSlice g_t failed: %w", err)
		}
		o_t, err := l.o_t.GetSlice(0, t)
		if err != nil {
			return fmt.Errorf("LSTM backward GetSlice o_t failed: %w", err)
		}

		var h_prev, c_prev *Tensor
		if t > 0 {
			h_prev, err = l.hidden_t.GetSlice(0, t-1)
			if err != nil {
				return fmt.Errorf("LSTM backward GetSlice h_prev failed: %w", err)
			}
			c_prev, err = l.cell_t.GetSlice(0, t-1)
			if err != nil {
				return fmt.Errorf("LSTM backward GetSlice c_prev failed: %w", err)
			}
		} else {
			h_prev = l.prevHidden
			c_prev = l.prevCell
		}

		// Add gradient from output of this time step
		dh, err := gradOutput.GetSlice(1, t)
		if err != nil {
			return fmt.Errorf("LSTM backward GetSlice gradOutput failed: %w", err)
		}
		dh, err = dh.Add(dh_next)
		if err != nil {
			return fmt.Errorf("LSTM dh add failed: %w", err)
		}

		// Gradient for c_t_tanh
		c_t_tanh_grad, err := h_t.Mul(dh)
		if err != nil {
			return fmt.Errorf("LSTM c_t_tanh_grad mul failed: %w", err)
		}
		c_t_tanh_grad, err = c_t_tanh_grad.Mul(o_t)
		if err != nil {
			return fmt.Errorf("LSTM c_t_tanh_grad mul o_t failed: %w", err)
		}
		c_t_tanh_grad, err = c_t_tanh_grad.OneMinusSquareTanh(c_t)
		if err != nil {
			return fmt.Errorf("LSTM c_t_tanh_grad one minus square tanh failed: %w", err)
		}

		// Gradient for c_t
		dc, err := dc_next.Add(c_t_tanh_grad)
		if err != nil {
			return fmt.Errorf("LSTM dc add failed: %w", err)
		}

		// Gradient for o_t
		c_t_tanh, err := c_t.Tanh()
		if err != nil {
			return fmt.Errorf("LSTM c_t.Tanh() failed: %w", err)
		}
		do_t, err := c_t_tanh.Mul(dh)
		if err != nil {
			return fmt.Errorf("LSTM do_t mul dh failed: %w", err)
		}
		do_t, err = do_t.SigmoidBackward(o_t)
		if err != nil {
			return fmt.Errorf("LSTM do_t sigmoid backward failed: %w", err)
		}

		// Gradient for g_t
		dg_t, err := i_t.Mul(dc)
		if err != nil {
			return fmt.Errorf("LSTM dg_t mul dc failed: %w", err)
		}
		dg_t, err = dg_t.TanhBackward(g_t)
		if err != nil {
			return fmt.Errorf("LSTM dg_t tanh backward failed: %w", err)
		}

		// Gradient for i_t
		di_t, err := g_t.Mul(dc)
		if err != nil {
			return fmt.Errorf("LSTM di_t mul dc failed: %w", err)
		}
		di_t, err = di_t.SigmoidBackward(i_t)
		if err != nil {
			return fmt.Errorf("LSTM di_t sigmoid backward failed: %w", err)
		}

		// Gradient for f_t
		df_t, err := c_prev.Mul(dc)
		if err != nil {
			return fmt.Errorf("LSTM df_t mul c_prev failed: %w", err)
		}
		df_t, err = df_t.SigmoidBackward(f_t)
		if err != nil {
			return fmt.Errorf("LSTM df_t sigmoid backward failed: %w", err)
		}

		// Gradients for W_io, W_ho, b_o
		input_t_transposed, err := input_t.Transpose(0, 1)
		if err != nil {
			return fmt.Errorf("input_t.Transpose failed: %w", err)
		}
		grad_W_io, err := input_t_transposed.MatMul(do_t)
		if err != nil {
			return fmt.Errorf("LSTM grad_W_io matmul failed: %w", err)
		}
		if _, err := l.W_io.Grad.Add(grad_W_io); err != nil {
			return fmt.Errorf("l.W_io.Grad.Add failed: %w", err)
		}
		h_prev_transposed, err := h_prev.Transpose(0, 1)
		if err != nil {
			return fmt.Errorf("h_prev.Transpose failed: %w", err)
		}
		grad_W_ho, err := h_prev_transposed.MatMul(do_t)
		if err != nil {
			return fmt.Errorf("LSTM grad_W_ho matmul failed: %w", err)
		}
		if _, err := l.W_ho.Grad.Add(grad_W_ho); err != nil {
			return fmt.Errorf("l.W_ho.Grad.Add failed: %w", err)
		}
		sum, err := do_t.Sum(0)
		if err != nil {
			return fmt.Errorf("do_t.Sum failed: %w", err)
		}
		if _, err := l.b_o.Grad.Add(sum); err != nil {
			return fmt.Errorf("l.b_o.Grad.Add failed: %w", err)
		}

		// Gradients for W_ig, W_hg, b_g
		grad_W_ig, err := input_t_transposed.MatMul(dg_t)
		if err != nil {
			return fmt.Errorf("LSTM grad_W_ig matmul failed: %w", err)
		}
		if _, err := l.W_ig.Grad.Add(grad_W_ig); err != nil {
			return fmt.Errorf("l.W_ig.Grad.Add failed: %w", err)
		}
		grad_W_hg, err := h_prev_transposed.MatMul(dg_t)
		if err != nil {
			return fmt.Errorf("LSTM grad_W_hg matmul failed: %w", err)
		}
		if _, err := l.W_hg.Grad.Add(grad_W_hg); err != nil {
			return fmt.Errorf("l.W_hg.Grad.Add failed: %w", err)
		}
		sum, err = dg_t.Sum(0)
		if err != nil {
			return fmt.Errorf("dg_t.Sum failed: %w", err)
		}
		if _, err := l.b_g.Grad.Add(sum); err != nil {
			return fmt.Errorf("l.b_g.Grad.Add failed: %w", err)
		}

		// Gradients for W_if, W_hf, b_f
		grad_W_if, err := input_t_transposed.MatMul(df_t)
		if err != nil {
			return fmt.Errorf("LSTM grad_W_if matmul failed: %w", err)
		}
		if _, err := l.W_if.Grad.Add(grad_W_if); err != nil {
			return fmt.Errorf("l.W_if.Grad.Add failed: %w", err)
		}
		grad_W_hf, err := h_prev_transposed.MatMul(df_t)
		if err != nil {
			return fmt.Errorf("LSTM grad_W_hf matmul failed: %w", err)
		}
		if _, err := l.W_hf.Grad.Add(grad_W_hf); err != nil {
			return fmt.Errorf("l.W_hf.Grad.Add failed: %w", err)
		}
		sum, err = df_t.Sum(0)
		if err != nil {
			return fmt.Errorf("df_t.Sum failed: %w", err)
		}
		if _, err := l.b_f.Grad.Add(sum); err != nil {
			return fmt.Errorf("l.b_f.Grad.Add failed: %w", err)
		}

		// Gradients for W_ii, W_hi, b_i
		grad_W_ii, err := input_t_transposed.MatMul(di_t)
		if err != nil {
			return fmt.Errorf("LSTM grad_W_ii matmul failed: %w", err)
		}
		if _, err := l.W_ii.Grad.Add(grad_W_ii); err != nil {
			return fmt.Errorf("l.W_ii.Grad.Add failed: %w", err)
		}
		grad_W_hi, err := h_prev_transposed.MatMul(di_t)
		if err != nil {
			return fmt.Errorf("LSTM grad_W_hi matmul failed: %w", err)
		}
		if _, err := l.W_hi.Grad.Add(grad_W_hi); err != nil {
			return fmt.Errorf("l.W_hi.Grad.Add failed: %w", err)
		}
		sum, err = di_t.Sum(0)
		if err != nil {
			return fmt.Errorf("di_t.Sum failed: %w", err)
		}
		if _, err := l.b_i.Grad.Add(sum); err != nil {
			return fmt.Errorf("l.b_i.Grad.Add failed: %w", err)
		}

		// Gradients for input_t
		w_io_transposed, err := l.W_io.Transpose(0, 1)
		if err != nil {
			return fmt.Errorf("l.W_io.Transpose failed: %w", err)
		}
		dinput_t_from_i, err := do_t.MatMul(w_io_transposed)
		if err != nil {
			return fmt.Errorf("LSTM dinput_t_from_i matmul failed: %w", err)
		}
		w_if_transposed, err := l.W_if.Transpose(0, 1)
		if err != nil {
			return fmt.Errorf("l.W_if.Transpose failed: %w", err)
		}
		dinput_t_from_f, err := df_t.MatMul(w_if_transposed)
		if err != nil {
			return fmt.Errorf("LSTM dinput_t_from_f matmul failed: %w", err)
		}
		w_ig_transposed, err := l.W_ig.Transpose(0, 1)
		if err != nil {
			return fmt.Errorf("l.W_ig.Transpose failed: %w", err)
		}
		dinput_t_from_g, err := dg_t.MatMul(w_ig_transposed)
		if err != nil {
			return fmt.Errorf("LSTM dinput_t_from_g matmul failed: %w", err)
		}
		w_ii_transposed, err := l.W_ii.Transpose(0, 1)
		if err != nil {
			return fmt.Errorf("l.W_ii.Transpose failed: %w", err)
		}
		dinput_t_from_o, err := di_t.MatMul(w_ii_transposed)
		if err != nil {
			return fmt.Errorf("LSTM dinput_t_from_o matmul failed: %w", err)
		}

		dinput_t, err := dinput_t_from_i.Add(dinput_t_from_f)
		if err != nil {
			return fmt.Errorf("LSTM dinput_t add failed: %w", err)
		}
		dinput_t, err = dinput_t.Add(dinput_t_from_g)
		if err != nil {
			return fmt.Errorf("LSTM dinput_t add failed: %w", err)
		}
		dinput_t, err = dinput_t.Add(dinput_t_from_o)
		if err != nil {
			return fmt.Errorf("LSTM dinput_t add failed: %w", err)
		}

		// Accumulate dinput_t to l.inputTensor.Grad
		if l.inputTensor.RequiresGrad {
			for b := 0; b < batchSize; b++ {
				for i := 0; i < inputSize; i++ {
					l.inputTensor.Grad.Data[b*seqLength*inputSize+t*inputSize+i] += dinput_t.Data[b*inputSize+i]
				}
			}
		}

		// Gradients for h_prev
		w_ho_transposed, err := l.W_ho.Transpose(0, 1)
		if err != nil {
			return fmt.Errorf("l.W_ho.Transpose failed: %w", err)
		}
		dh_prev_from_i, err := do_t.MatMul(w_ho_transposed)
		if err != nil {
			return fmt.Errorf("LSTM dh_prev_from_i matmul failed: %w", err)
		}
		w_hf_transposed, err := l.W_hf.Transpose(0, 1)
		if err != nil {
			return fmt.Errorf("l.W_hf.Transpose failed: %w", err)
		}
		dh_prev_from_f, err := df_t.MatMul(w_hf_transposed)
		if err != nil {
			return fmt.Errorf("LSTM dh_prev_from_f matmul failed: %w", err)
		}
		w_hg_transposed, err := l.W_hg.Transpose(0, 1)
		if err != nil {
			return fmt.Errorf("l.W_hg.Transpose failed: %w", err)
		}
		dh_prev_from_g, err := dg_t.MatMul(w_hg_transposed)
		if err != nil {
			return fmt.Errorf("LSTM dh_prev_from_g matmul failed: %w", err)
		}
		w_hi_transposed, err := l.W_hi.Transpose(0, 1)
		if err != nil {
			return fmt.Errorf("l.W_hi.Transpose failed: %w", err)
		}
		dh_prev_from_o, err := di_t.MatMul(w_hi_transposed)
		if err != nil {
			return fmt.Errorf("LSTM dh_prev_from_o matmul failed: %w", err)
		}

		dh_prev, err := dh_prev_from_i.Add(dh_prev_from_f)
		if err != nil {
			return fmt.Errorf("LSTM dh_prev add failed: %w", err)
		}
		dh_prev, err = dh_prev.Add(dh_prev_from_g)
		if err != nil {
			return fmt.Errorf("LSTM dh_prev add failed: %w", err)
		}
		dh_prev, err = dh_prev.Add(dh_prev_from_o)
		if err != nil {
			return fmt.Errorf("LSTM dh_prev add failed: %w", err)
		}

		// Gradients for c_prev
		dc_prev, err := dc.Mul(f_t)
		if err != nil {
			return fmt.Errorf("LSTM dc_prev mul failed: %w", err)
		}
		dc_prev_from_f, err := df_t.Mul(c_prev)
		if err != nil {
			return fmt.Errorf("LSTM dc_prev_from_f mul failed: %w", err)
		}
		dc_prev, err = dc_prev.Add(dc_prev_from_f)
		if err != nil {
			return fmt.Errorf("LSTM dc_prev add failed: %w", err)
		}

		dh_next = dh_prev
		dc_next = dc_prev
	}

	// Accumulate final dh_next and dc_next to l.prevHidden.Grad and l.prevCell.Grad
	if l.prevHidden.RequiresGrad {
		if _, err := l.prevHidden.Grad.Add(dh_next); err != nil {
			return fmt.Errorf("l.prevHidden.Grad.Add failed: %w", err)
		}
	}
	if l.prevCell.RequiresGrad {
		if _, err := l.prevCell.Grad.Add(dc_next); err != nil {
			return fmt.Errorf("l.prevCell.Grad.Add failed: %w", err)
		}
	}

	return nil
}

// Inputs returns the input tensors of the LSTM operation.
func (l *LSTM) Inputs() []*Tensor {
	inputs := []*Tensor{l.inputTensor}
	if l.prevHidden != nil {
		inputs = append(inputs, l.prevHidden)
	}
	if l.prevCell != nil {
		inputs = append(inputs, l.prevCell)
	}
	return inputs
}
