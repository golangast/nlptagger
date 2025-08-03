package bert

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/golangast/nlptagger/neural/nnu/bartsimple"
)

// init seeds the random number generator.
func init() {
	rand.Seed(time.Now().UnixNano())
}

// --- Custom Tensor and Autodiff Engine ---

// Tensor is a multi-dimensional matrix that supports automatic differentiation.
type Tensor struct {
	Data      []float64
	Shape     []int
	Grad      []float64
	creator   *op
	visited   bool // For topological sort
}

// op represents an operation that created a tensor, for building the computation graph.
type op struct {
	inputs   []*Tensor
	backward func()
}

// NewTensor creates a new tensor, optionally initializing its gradient.
func NewTensor(data []float64, shape []int, requiresGrad bool) *Tensor {
	size := tensorSize(shape)
	if data == nil {
		data = make([]float64, size)
	}
	t := &Tensor{Data: data, Shape: shape}
	if requiresGrad {
		t.Grad = make([]float64, size)
	}
	return t
}

// tensorSize calculates the total number of elements in a tensor.
func tensorSize(shape []int) int {
	if len(shape) == 0 {
		return 1 // A scalar
	}
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	return size
}

// Backward performs backpropagation starting from this tensor.
func (t *Tensor) Backward() {
	// Build the computation graph in reverse topological order
	topo := []*Tensor{}
	var buildTopo func(*Tensor)
	buildTopo = func(v *Tensor) {
		if !v.visited {
			v.visited = true
			if v.creator != nil {
				for _, input := range v.creator.inputs {
					buildTopo(input)
				}
			}
			topo = append(topo, v)
		}
	}
	buildTopo(t)

	// Initialize the gradient of the output tensor to 1.
	for i := range t.Grad {
		t.Grad[i] = 1
	}

	// Propagate gradients backward through the graph.
	for i := len(topo) - 1; i >= 0; i-- {
		if topo[i].creator != nil {
			topo[i].creator.backward()
		}
	}
}

// ZeroGrad resets the gradient of the tensor.
func (t *Tensor) ZeroGrad() {
	if t.Grad != nil {
		for i := range t.Grad {
			t.Grad[i] = 0
		}
	}
}

// Reshape changes the shape of the tensor. The total number of elements must remain the same.
func (t *Tensor) Reshape(newShape []int) (*Tensor, error) {
	currentSize := tensorSize(t.Shape)
	newSize := tensorSize(newShape)

	if currentSize != newSize {
		return nil, fmt.Errorf("cannot reshape tensor of size %d to shape %v (size %d)", currentSize, newShape, newSize)
	}

	// In a real implementation, we might want to avoid copying data.
	// For simplicity, we create a new tensor with copied data.
	newData := make([]float64, len(t.Data))
	copy(newData, t.Data)

	// The reshaped tensor should propagate gradients if the original does.
	reshaped := NewTensor(newData, newShape, t.Grad != nil)

	// This part is tricky for autograd. A simple reshape might break gradient flow
	// if not handled carefully. For now, we won't link the creator.
	// A more robust solution would require a reshape operation in the computation graph.

	return reshaped, nil
}

// Transpose swaps two dimensions of the tensor.
func (t *Tensor) Transpose(axis1, axis2 int) (*Tensor, error) {
    if axis1 < 0 || axis1 >= len(t.Shape) || axis2 < 0 || axis2 >= len(t.Shape) {
        return nil, fmt.Errorf("invalid axes for transpose: %d, %d", axis1, axis2)
    }

    newShape := make([]int, len(t.Shape))
    copy(newShape, t.Shape)
    newShape[axis1], newShape[axis2] = newShape[axis2], newShape[axis1]

    newData := make([]float64, len(t.Data))

    // Strides for the source tensor
    sourceStrides := make([]int, len(t.Shape))
    sourceStrides[len(t.Shape)-1] = 1
    for i := len(t.Shape) - 2; i >= 0; i-- {
        sourceStrides[i] = sourceStrides[i+1] * t.Shape[i+1]
    }

    // Strides for the destination tensor
    destStrides := make([]int, len(newShape))
    destStrides[len(newShape)-1] = 1
    for i := len(newShape) - 2; i >= 0; i-- {
        destStrides[i] = destStrides[i+1] * newShape[i+1]
    }

    for i := 0; i < len(t.Data); i++ {
        // Calculate source multi-index
        sourceMultiIndex := make([]int, len(t.Shape))
        remainder := i
        for j := 0; j < len(t.Shape); j++ {
            sourceMultiIndex[j] = remainder / sourceStrides[j]
            remainder %= sourceStrides[j]
        }

        // Create destination multi-index by swapping axes
        destMultiIndex := make([]int, len(newShape))
        copy(destMultiIndex, sourceMultiIndex)
        destMultiIndex[axis1], destMultiIndex[axis2] = sourceMultiIndex[axis2], sourceMultiIndex[axis1]

        // Calculate destination flat index
        destIndex := 0
        for j := 0; j < len(newShape); j++ {
            destIndex += destMultiIndex[j] * destStrides[j]
        }
        newData[destIndex] = t.Data[i]
    }

    transposed := NewTensor(newData, newShape, t.Grad != nil)
    // Note: A proper backward pass for transpose is non-trivial and is omitted here.
    return transposed, nil
}


// --- Tensor Operations ---

func (t1 *Tensor) MatMul(t2 *Tensor) (*Tensor, error) {
    // Handles 2D and 3D matrix multiplication.
    // Case 1: 2D @ 2D -> [a, b] @ [b, c] = [a, c]
    // Case 2: 3D @ 2D -> [batch, n, m] @ [m, p] = [batch, n, p]
    // Case 3: 3D @ 3D -> [batch, n, m] @ [batch, m, p] = [batch, n, p]

    if len(t1.Shape) == 2 && len(t2.Shape) == 2 {
        // Standard 2D matrix multiplication
        a, b1 := t1.Shape[0], t1.Shape[1]
        b2, c := t2.Shape[0], t2.Shape[1]
        if b1 != b2 {
            panic(fmt.Sprintf("MatMul shape mismatch: %v and %v", t1.Shape, t2.Shape))
        }

        resultData := make([]float64, a*c)
        for i := 0; i < a; i++ {
            for j := 0; j < c; j++ {
                for k := 0; k < b1; k++ {
                    resultData[i*c+j] += t1.Data[i*b1+k] * t2.Data[k*c+j]
                }
            }
        }
        result := NewTensor(resultData, []int{a, c}, t1.Grad != nil || t2.Grad != nil)
        // Backward pass for 2D MatMul (simplified)
        return result, nil

    } else if len(t1.Shape) == 3 && len(t2.Shape) == 2 {
        // Batched 3D @ 2D
        batch, n, m1 := t1.Shape[0], t1.Shape[1], t1.Shape[2]
        m2, p := t2.Shape[0], t2.Shape[1]
        if m1 != m2 {
            // If the inner dimensions don't match, check if t2 is implicitly transposed.
            // This handles cases like (batch, n, m) @ (p, m), which is treated as (batch, n, m) @ (m, p).
            if m1 == p {
                p_new := m2
                resultShape := []int{batch, n, p_new}
                resultData := make([]float64, batch*n*p_new)
                for i := 0; i < batch; i++ {
                    for j := 0; j < n; j++ {
                        for k := 0; k < p_new; k++ {
                            sum := 0.0
                            for l := 0; l < m1; l++ {
                                // t2 is indexed as (k, l) instead of (l, k)
                                sum += t1.Data[i*n*m1+j*m1+l] * t2.Data[k*m1+l]
                            }
                            resultData[i*n*p_new+j*p_new+k] = sum
                        }
                    }
                }
                return NewTensor(resultData, resultShape, t1.Grad != nil || t2.Grad != nil), nil
            }
            panic(fmt.Sprintf("MatMul shape mismatch: %v and %v", t1.Shape, t2.Shape))
        }

        resultShape := []int{batch, n, p}
        resultData := make([]float64, batch*n*p)
        for i := 0; i < batch; i++ {
            for j := 0; j < n; j++ {
                for k := 0; k < p; k++ {
                    sum := 0.0
                    for l := 0; l < m1; l++ {
                        sum += t1.Data[i*n*m1+j*m1+l] * t2.Data[l*p+k]
                    }
                    resultData[i*n*p+j*p+k] = sum
                }
            }
        }
        return NewTensor(resultData, resultShape, t1.Grad != nil || t2.Grad != nil), nil

    } else if len(t1.Shape) == 3 && len(t2.Shape) == 3 {
        // Batched 3D @ 3D
        batch1, n, m1 := t1.Shape[0], t1.Shape[1], t1.Shape[2]
        batch2, m2, p := t2.Shape[0], t2.Shape[1], t2.Shape[2]
        if batch1 != batch2 || m1 != m2 {
            panic(fmt.Sprintf("MatMul shape mismatch: %v and %v", t1.Shape, t2.Shape))
        }

        resultShape := []int{batch1, n, p}
        resultData := make([]float64, batch1*n*p)
        for i := 0; i < batch1; i++ {
            for j := 0; j < n; j++ {
                for k := 0; k < p; k++ {
                    sum := 0.0
                    for l := 0; l < m1; l++ {
                        sum += t1.Data[i*n*m1+j*m1+l] * t2.Data[i*m1*p+l*p+k]
                    }
                    resultData[i*n*p+j*p+k] = sum
                }
            }
        }
        return NewTensor(resultData, resultShape, t1.Grad != nil || t2.Grad != nil), nil

    } else if len(t1.Shape) == 4 && len(t2.Shape) == 4 {
        // Batched 4D @ 4D: [b1, b2, n, m] @ [b1, b2, m, p] = [b1, b2, n, p]
        b1_1, b2_1, n, m1 := t1.Shape[0], t1.Shape[1], t1.Shape[2], t1.Shape[3]
        b1_2, b2_2, m2, p := t2.Shape[0], t2.Shape[1], t2.Shape[2], t2.Shape[3]

        if b1_1 != b1_2 || b2_1 != b2_2 || m1 != m2 {
            panic(fmt.Sprintf("MatMul shape mismatch for 4D: %v and %v", t1.Shape, t2.Shape))
        }

        resultShape := []int{b1_1, b2_1, n, p}
        resultData := make([]float64, b1_1*b2_1*n*p)

        for i := 0; i < b1_1; i++ { // First batch dimension
            for j := 0; j < b2_1; j++ { // Second batch dimension
                for k := 0; k < n; k++ { // Rows of the matrix
                    for l := 0; l < p; l++ { // Columns of the matrix
                        sum := 0.0
                        for m := 0; m < m1; m++ { // Inner dimension
                            // Indexing for t1: [i, j, k, m]
                            t1_idx := i*b2_1*n*m1 + j*n*m1 + k*m1 + m
                            // Indexing for t2: [i, j, m, l]
                            t2_idx := i*b2_2*m2*p + j*m2*p + m*p + l
                            sum += t1.Data[t1_idx] * t2.Data[t2_idx]
                        }
                        // Indexing for result: [i, j, k, l]
                        result_idx := i*b2_1*n*p + j*n*p + k*p + l
                        resultData[result_idx] = sum
                    }
                }
            }
        }
        return NewTensor(resultData, resultShape, t1.Grad != nil || t2.Grad != nil), nil
    }
    return nil, fmt.Errorf("Unsupported MatMul shapes: %v and %v", t1.Shape, t2.Shape)
}

func (t1 *Tensor) Add(t2 *Tensor) *Tensor {
	// Simplified Add for broadcasting a vector to a matrix
	resultData := make([]float64, len(t1.Data))
	copy(resultData, t1.Data)
	numVectors := tensorSize(t1.Shape) / t2.Shape[0]
	for i := 0; i < numVectors; i++ {
		for j := 0; j < t2.Shape[0]; j++ {
			resultData[i*t2.Shape[0]+j] += t2.Data[j]
		}
	}

	result := NewTensor(resultData, t1.Shape, t1.Grad != nil || t2.Grad != nil)
	result.creator = &op{
		inputs: []*Tensor{t1, t2},
		backward: func() {
			if t1.Grad != nil {
				for i := range t1.Grad {
					t1.Grad[i] += result.Grad[i]
				}
			}
			if t2.Grad != nil {
				for i := 0; i < numVectors; i++ {
					for j := 0; j < t2.Shape[0]; j++ {
						t2.Grad[j] += result.Grad[i*t2.Shape[0]+j]
					}
				}
			}
		},
	}
	return result
}

func (t *Tensor) Softmax(axis int) *Tensor {
	// Simplified Softmax for the last dimension of a 2D tensor
	batchSize, numClasses := t.Shape[0], t.Shape[1]
	resultData := make([]float64, len(t.Data))

	for i := 0; i < batchSize; i++ {
		maxVal := -math.MaxFloat64
		offset := i * numClasses
		for j := 0; j < numClasses; j++ {
			if t.Data[offset+j] > maxVal {
				maxVal = t.Data[offset+j]
			}
		}
		sum := 0.0
		for j := 0; j < numClasses; j++ {
			expVal := math.Exp(t.Data[offset+j] - maxVal)
			resultData[offset+j] = expVal
			sum += expVal
		}
		for j := 0; j < numClasses; j++ {
			resultData[offset+j] /= sum
		}
	}

	result := NewTensor(resultData, t.Shape, t.Grad != nil)
	result.creator = &op{
		inputs: []*Tensor{t},
		backward: func() {
			if t.Grad != nil {
				for i := 0; i < batchSize*numClasses; i++ {
					for j := 0; j < numClasses; j++ {
						row := i / numClasses
						col := i % numClasses
						if col == j {
							t.Grad[i] += result.Grad[i] * result.Data[i] * (1 - result.Data[i])
						} else {
							t.Grad[i] -= result.Grad[row*numClasses+j] * result.Data[i] * result.Data[row*numClasses+j]
						}
					}
				}
			}
		},
	}
	return result
}

func (t *Tensor) Tanh() *Tensor {
	resultData := make([]float64, len(t.Data))
	for i, v := range t.Data {
		resultData[i] = math.Tanh(v)
	}
	result := NewTensor(resultData, t.Shape, t.Grad != nil)
	result.creator = &op{
		inputs: []*Tensor{t},
		backward: func() {
			if t.Grad != nil {
				for i := range t.Grad {
					t.Grad[i] += (1 - result.Data[i]*result.Data[i]) * result.Grad[i]
				}
			}
		},
	}
	return result
}

// gelu activation function
func gelu(x float64) float64 {
	return 0.5 * x * (1.0 + math.Tanh(math.Sqrt(2.0/math.Pi)*(x+0.044715*math.Pow(x, 3))))
}

// gelu derivative
func geluDerivative(x float64) float64 {
	const a = 0.044715
	const b = 0.7978845608 // sqrt(2/pi)
	tanh_arg := b * (x + a*math.Pow(x, 3))
	sech_sq := 1 - math.Pow(math.Tanh(tanh_arg), 2)
	return 0.5*(1+math.Tanh(tanh_arg)) + 0.5*x*sech_sq*b*(1+3*a*math.Pow(x, 2))
}

// BertConfig holds the configuration for the BERT model.
type BertConfig struct {
	VocabSize                 int
	HiddenSize                int
	NumHiddenLayers           int
	NumAttentionHeads         int
	IntermediateSize          int
	MaxPositionEmbeddings     int
	TypeVocabSize             int
	LayerNormEps              float64
	HiddenDropoutProb       float64
	AttentionProbsDropoutProb float64
	NumLabels                 int // Number of labels for classification
}

// --- Full BERT Model Components ---

// Linear implements a standard fully connected layer.
type Linear struct {
	Weight *Tensor
	Bias   *Tensor
}

func NewLinear(inFeatures, outFeatures int, initializerStdDev float64) *Linear {
    if outFeatures == 0 {
        panic("NewLinear: outFeatures must be > 0")
    }
    weight := NewTensor(nil, []int{inFeatures, outFeatures}, true)
    bias := NewTensor(nil, []int{outFeatures}, true)
    for i := range weight.Data {
        weight.Data[i] = rand.NormFloat64() * initializerStdDev
    }
    return &Linear{Weight: weight, Bias: bias}
}

func (l *Linear) Forward(x *Tensor) (*Tensor, error) {
	m, err := x.MatMul(l.Weight)
	if err != nil {
		return nil, err
	}
	if len(l.Bias.Shape) == 0 || l.Bias.Shape[0] == 0 {
		fmt.Printf("Linear.Forward: bias shape is %v\n", l.Bias.Shape)
	}
	mm := m.Add(l.Bias)
	return mm, err
}

func (l *Linear) Parameters() []*Tensor {
	return []*Tensor{l.Weight, l.Bias}
}

// BertPooler takes the output of the first token ([CLS]) for classification.
type BertPooler struct {
	Dense *Linear
}

func NewBertPooler(config BertConfig, initializerStdDev float64) *BertPooler {
	return &BertPooler{
		Dense: NewLinear(config.HiddenSize, config.HiddenSize, initializerStdDev),
	}
}

func (p *BertPooler) Forward(hiddenStates *Tensor) (*Tensor, error) {
    // hiddenStates: [batch, seq_len, hidden_size]
    batch := hiddenStates.Shape[0]
    hiddenSize := hiddenStates.Shape[2]
    firstTokenHiddenState := make([]float64, batch*hiddenSize)
    for i := 0; i < batch; i++ {
        copy(
            firstTokenHiddenState[i*hiddenSize:(i+1)*hiddenSize],
            hiddenStates.Data[i*hiddenStates.Shape[1]*hiddenSize:i*hiddenStates.Shape[1]*hiddenSize+hiddenSize],
        )
    }
    firstToken := NewTensor(firstTokenHiddenState, []int{batch, hiddenSize}, hiddenStates.Grad != nil)
    pp, err := p.Dense.Forward(firstToken)
    if err != nil {
        return nil, err
    }
    return pp.Tanh(), err
}

func (p *BertPooler) Parameters() []*Tensor {
	return p.Dense.Parameters()
}

// BertModel is the main BERT model structure.
type BertModel struct {
	Embeddings         *BertEmbeddings
	Encoder            *BertEncoder
	Pooler             *BertPooler
	ClassificationHead *Linear
}

func NewBertModel(config BertConfig) *BertModel {
	initializerStdDev := 0.02
	config.NumLabels = 3
	return &BertModel{
		Embeddings:         NewBertEmbeddings(config, initializerStdDev),
		Encoder:            NewBertEncoder(config, initializerStdDev),
		Pooler:             NewBertPooler(config, initializerStdDev),
		ClassificationHead: NewLinear(config.HiddenSize, config.NumLabels, initializerStdDev),
	}
}

func (m *BertModel) Forward(inputIDs, tokenTypeIDs *Tensor) (*Tensor, error) {
	embeddingOutput := m.Embeddings.Forward(inputIDs, tokenTypeIDs)
	sequenceOutput, err := m.Encoder.Forward(embeddingOutput)
	if err != nil {
		return nil, err
	}
	pooledOutput,err := m.Pooler.Forward(sequenceOutput)
	if err != nil {
		return nil, err
	}
	logits,err := m.ClassificationHead.Forward(pooledOutput)
	if err != nil {
		return nil, err
	}
	return logits, nil
}

func (m *BertModel) Parameters() []*Tensor {
	params := m.Embeddings.Parameters()
	params = append(params, m.Encoder.Parameters()...)
	params = append(params, m.Pooler.Parameters()...)
	params = append(params, m.ClassificationHead.Parameters()...)
	return params
}

// --- Optimizer ---

// Adam optimizer.
type Adam struct {
	Parameters []*Tensor
	LR         float64
	Beta1      float64
	Beta2      float64
	Epsilon    float64
	M          map[*Tensor][]float64
	V          map[*Tensor][]float64
	T          int
}

// NewAdam creates a new Adam optimizer.
func NewAdam(params []*Tensor, lr float64) *Adam {
	m := make(map[*Tensor][]float64)
	v := make(map[*Tensor][]float64)
	for _, p := range params {
		m[p] = make([]float64, tensorSize(p.Shape))
		v[p] = make([]float64, tensorSize(p.Shape))
	}
	return &Adam{
		Parameters: params,
		LR:         lr,
		Beta1:      0.9,
		Beta2:      0.999,
		Epsilon:    1e-8,
		M:          m,
		V:          v,
		T:          0,
	}
}

// Step performs a single optimization step.
func (a *Adam) Step() {
	a.T++
	for _, p := range a.Parameters {
		grad := p.Grad
		m := a.M[p]
		v := a.V[p]

		for i := range grad {
			m[i] = a.Beta1*m[i] + (1-a.Beta1)*grad[i]
			v[i] = a.Beta2*v[i] + (1-a.Beta2)*(grad[i]*grad[i])
			mHat := m[i] / (1 - math.Pow(a.Beta1, float64(a.T)))
			vHat := v[i] / (1 - math.Pow(a.Beta2, float64(a.T)))
			p.Data[i] -= a.LR * mHat / (math.Sqrt(vHat) + a.Epsilon)
		}
	}
}

// ZeroGrad clears the gradients of all parameters.
func (a *Adam) ZeroGrad() {
	for _, p := range a.Parameters {
		p.ZeroGrad()
	}
}

// --- Loss Function ---

// CrossEntropyLoss computes the cross-entropy loss and returns the loss as a scalar tensor.
func CrossEntropyLoss(logits *Tensor, labels *Tensor) *Tensor {
	probs := logits.Softmax(-1)
	batchSize := logits.Shape[0]
	numClasses := logits.Shape[1]

	// Compute loss
	lossVal := 0.0
	for i := 0; i < batchSize; i++ {
		label := int(labels.Data[i])
		// Add a small epsilon to prevent log(0)
		safeProb := math.Max(1e-9, probs.Data[i*numClasses+label])
		lossVal += -math.Log(safeProb)
	}
	lossVal /= float64(batchSize)

	loss := NewTensor([]float64{lossVal}, []int{1}, true)

	// Define the backward pass for the loss
	loss.creator = &op{
		inputs: []*Tensor{logits, labels},
		backward: func() {
			// Gradient of loss with respect to logits
			gradLogits := make([]float64, len(probs.Data))
			copy(gradLogits, probs.Data)

			for i := 0; i < batchSize; i++ {
				label := int(labels.Data[i])
				gradLogits[i*numClasses+label] -= 1
			}
			for i := range gradLogits {
				logits.Grad[i] += gradLogits[i] / float64(batchSize)
			}
		},
	}

	return loss
}

func (m *BertModel) BertProcessCommand(command string, config BertConfig, tokenizer *bartsimple.Tokenizer) (string, error) {
	// 1. Tokenize the input command
	tokenIDs, err := tokenizer.Encode(command)
	if err != nil{
		return "", err
	}

	// 2. Create input tensors
	inputTensor := NewTensor(nil, []int{1, len(tokenIDs)}, false)
	for i, id := range tokenIDs {
		inputTensor.Data[i] = float64(id)
	}
	tokenTypeIDs := NewTensor(make([]float64, len(tokenIDs)), []int{1, len(tokenIDs)}, false) // All zeros for a single sentence

	// 3. Perform a forward pass
	logits, err := m.Forward(inputTensor, tokenTypeIDs)
	if err != nil {
		return "", err
	}

	// 4. Find the predicted label
	maxLogit := -1e9
	predictedLabel := -1
	for i, logit := range logits.Data {
		if logit > maxLogit {
			maxLogit = logit
			predictedLabel = i
		}
	}

	if predictedLabel == -1 {
		return "", fmt.Errorf("prediction failed")
	}

	// 5. Map label to intent string (hardcoded for now)
	intentMap := map[int]string{
		0: "CREATE_WEBSERVER",
		1: "CREATE_DATABASE",
		2: "CREATE_HANDLER",
	}

	if intent, ok := intentMap[predictedLabel]; ok {
		return intent, nil
	} else {
		return "UNKNOWN", nil
	}
}
