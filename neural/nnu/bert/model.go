package bert

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/golangast/nlptagger/tagger/nertagger"
	"github.com/golangast/nlptagger/tagger/postagger"
)

// Embedding layer for token embeddings.
type Embedding struct {
	Weight *Tensor
}

// NewEmbedding creates a new Embedding layer.
func NewEmbedding(numEmbeddings, embeddingDim int, initializerStdDev float64) *Embedding {
	weight := NewTensor(nil, []int{numEmbeddings, embeddingDim}, true)
	for i := range weight.Data {
		weight.Data[i] = rand.NormFloat64() * initializerStdDev
	}
	return &Embedding{Weight: weight}
}

// Forward pass for Embedding layer (lookup).
func (e *Embedding) Forward(inputIDs *Tensor) *Tensor {
	if e.Weight == nil {
		panic("Embedding.Forward: e.Weight is nil!")
	}
	if len(e.Weight.Shape) < 2 {
		panic(fmt.Sprintf("Embedding.Forward: e.Weight.Shape is too short: %v", e.Weight.Shape))
	}
	batchSize, seqLength := inputIDs.Shape[0], inputIDs.Shape[1]
	embeddingDim := e.Weight.Shape[1]


	outputData := make([]float64, batchSize*seqLength*embeddingDim)

	for i := 0; i < batchSize; i++ {
		for j := 0; j < seqLength; j++ {
			tokenID := int(inputIDs.Data[i*seqLength+j])
			tokenEmbedding := e.Weight.Data[tokenID*embeddingDim : (tokenID+1)*embeddingDim]
			copy(outputData[(i*seqLength+j)*embeddingDim:], tokenEmbedding)
		}
	}

	output := NewTensor(outputData, []int{batchSize, seqLength, embeddingDim}, e.Weight.Grad != nil)
	// Simplified backward pass for embedding lookup
	if e.Weight.Grad != nil {
		output.creator = &op{
			inputs: []*Tensor{inputIDs, e.Weight},
			backward: func() {
				for i := 0; i < batchSize; i++ {
					for j := 0; j < seqLength; j++ {
						tokenID := int(inputIDs.Data[i*seqLength+j])
						gradSlice := output.Grad[(i*seqLength+j)*embeddingDim : (i*seqLength+j+1)*embeddingDim]
						for k := 0; k < embeddingDim; k++ {
							e.Weight.Grad[tokenID*embeddingDim+k] += gradSlice[k]
						}
					}
				}
			},
		}
	}
	return output
}

func (e *Embedding) Parameters() []*Tensor {
	return []*Tensor{e.Weight}
}

// BertEmbeddings handles the input embeddings for the BERT model.
type BertEmbeddings struct {
	WordEmbeddings        *Embedding
	PositionEmbeddings    *Embedding
	TokenTypeEmbeddings   *Embedding
	PosTagEmbeddings      *Embedding
	NerTagEmbeddings      *Embedding
	LayerNorm             *LayerNorm
	Dropout               float64
	PositionIDs           *Tensor
	TokenTypeIDs          *Tensor
}

// NewBertEmbeddings creates a new BertEmbeddings layer.
func NewBertEmbeddings(config BertConfig, initializerStdDev float64, word2vecEmbeddings map[string][]float64) *BertEmbeddings {
	wordEmbeddings := NewEmbedding(config.VocabSize, config.HiddenSize, initializerStdDev)
	if word2vecEmbeddings != nil {
		// Initialize with Word2Vec embeddings
		for word, i := range config.Vocabulary.WordToToken {
			if vec, ok := word2vecEmbeddings[word]; ok {
				copy(wordEmbeddings.Weight.Data[i*config.HiddenSize:(i+1)*config.HiddenSize], vec)
			}
		}
	}
	positionEmbeddings := NewEmbedding(config.MaxPositionEmbeddings, config.HiddenSize, initializerStdDev)
	tokenTypeEmbeddings := NewEmbedding(config.TypeVocabSize, config.HiddenSize, initializerStdDev)
	posTagEmbeddings := NewEmbedding(postagger.PosTags(), config.HiddenSize, initializerStdDev)
	nerTagEmbeddings := NewEmbedding(nertagger.NerTags(), config.HiddenSize, initializerStdDev)
	layerNorm := NewLayerNorm(config.HiddenSize, config.LayerNormEps)


	// Create position IDs tensor
	posData := make([]float64, config.MaxPositionEmbeddings)
	for i := 0; i < config.MaxPositionEmbeddings; i++ {
		posData[i] = float64(i)
	}
	positionIDs := NewTensor(posData, []int{1, config.MaxPositionEmbeddings}, false)

	return &BertEmbeddings{
		WordEmbeddings:        wordEmbeddings,
		PositionEmbeddings:    positionEmbeddings,
		TokenTypeEmbeddings:   tokenTypeEmbeddings,
		PosTagEmbeddings:      posTagEmbeddings,
		NerTagEmbeddings:      nerTagEmbeddings,
		LayerNorm:             layerNorm,
		Dropout:               config.HiddenDropoutProb,
		PositionIDs:           positionIDs,
	}
}

// Forward pass for BertEmbeddings.
func (e *BertEmbeddings) Forward(inputIDs, tokenTypeIDs, posTagIDs, nerTagIDs *Tensor) *Tensor {
	seqLength := inputIDs.Shape[1]

	positionIDs := e.PositionIDs.slice(0, seqLength)

	words := e.WordEmbeddings.Forward(inputIDs)
	positions := e.PositionEmbeddings.Forward(positionIDs)
	types := e.TokenTypeEmbeddings.Forward(tokenTypeIDs)
	posTags := e.PosTagEmbeddings.Forward(posTagIDs)
	nerTags := e.NerTagEmbeddings.Forward(nerTagIDs)

	embeddings := words.Add(positions).Add(types).Add(posTags).Add(nerTags)
	normalizedEmbeddings := e.LayerNorm.Forward(embeddings)

	return normalizedEmbeddings
}

func (e *BertEmbeddings) Parameters() []*Tensor {
	params := e.WordEmbeddings.Parameters()
	params = append(params, e.PositionEmbeddings.Parameters()...)
	params = append(params, e.TokenTypeEmbeddings.Parameters()...)
	params = append(params, e.LayerNorm.Parameters()...)
	return params
}


// BertSelfAttention implements the self-attention mechanism.
type BertSelfAttention struct {
	Query *Linear
	Key   *Linear
	Value *Linear
	NumAttentionHeads int
	AttentionHeadSize int
	Dropout float64
}

func NewBertSelfAttention(config BertConfig, initializerStdDev float64) *BertSelfAttention {
	if config.HiddenSize % config.NumAttentionHeads != 0 {
		panic("Hidden size is not a multiple of the number of attention heads")
	}
	return &BertSelfAttention{
		Query:             NewLinear(config.HiddenSize, config.HiddenSize, initializerStdDev),
		Key:               NewLinear(config.HiddenSize, config.HiddenSize, initializerStdDev),
		Value:             NewLinear(config.HiddenSize, config.HiddenSize, initializerStdDev),
		NumAttentionHeads: config.NumAttentionHeads,
		AttentionHeadSize: config.HiddenSize / config.NumAttentionHeads,
		Dropout:           config.AttentionProbsDropoutProb,
	}
}

func (sa *BertSelfAttention) Forward(hiddenStates *Tensor) (*Tensor, error) {
    // Project hidden states to Q, K, V
    queryLayer, err := sa.Query.Forward(hiddenStates)
	if err != nil {
		return nil, err
	}
    keyLayer,err := sa.Key.Forward(hiddenStates)
	if err != nil {
		return nil, err
	}
    valueLayer,err := sa.Value.Forward(hiddenStates)
	if err != nil {
		return nil, err
	}

    // Transpose for multi-head attention
    queryLayer = queryLayer.transposeForScores(sa.NumAttentionHeads, sa.AttentionHeadSize)
    keyLayer = keyLayer.transposeForScores(sa.NumAttentionHeads, sa.AttentionHeadSize)
    valueLayer = valueLayer.transposeForScores(sa.NumAttentionHeads, sa.AttentionHeadSize)

    // Transpose key for matrix multiplication
    keyLayerTransposed, err := keyLayer.Transpose(2, 3)
    if err != nil {
        		return nil, err
    }

    // Calculate attention scores
    attentionScores, err := queryLayer.MatMul(keyLayerTransposed)
	if err != nil {
		return nil, err
	}
    attentionScores = attentionScores.DivScalar(math.Sqrt(float64(sa.AttentionHeadSize)))

    // Apply softmax to get probabilities
    attentionProbs := attentionScores.Softmax(-1)
    // Dropout is conceptual here

    // Weighted sum of values
    contextLayer, err := attentionProbs.MatMul(valueLayer)
	if err != nil {
		return nil, err
	}
    contextLayer, err = contextLayer.Transpose(1, 2) // Transpose back
    if err != nil {
		return nil, err
	}

    // Reshape to original dimensions
    batchSize, seqLength, _ := hiddenStates.Shape[0], hiddenStates.Shape[1], hiddenStates.Shape[2]
    hiddenSize := sa.NumAttentionHeads * sa.AttentionHeadSize
    contextLayer, err = contextLayer.Reshape([]int{batchSize, seqLength, hiddenSize})
    if err != nil {
		return nil, err
	}

    return contextLayer, nil
}

func (sa *BertSelfAttention) Parameters() []*Tensor {
	params := sa.Query.Parameters()
	params = append(params, sa.Key.Parameters()...)
	params = append(params, sa.Value.Parameters()...)
	return params
}


// BertSelfOutput handles the output of the BertSelfAttention layer.
type BertSelfOutput struct {
	Dense     *Linear
	LayerNorm *LayerNorm
	Dropout   float64
}

func NewBertSelfOutput(config BertConfig, initializerStdDev float64) *BertSelfOutput {
	return &BertSelfOutput{
		Dense:     NewLinear(config.HiddenSize, config.HiddenSize, initializerStdDev),
		LayerNorm: NewLayerNorm(config.HiddenSize, config.LayerNormEps),
		Dropout:   config.HiddenDropoutProb,
	}
}

func (o *BertSelfOutput) Forward(hiddenStates, inputTensor *Tensor) (*Tensor, error) {
	hiddenStates, err := o.Dense.Forward(hiddenStates)
	if err != nil {
		return nil, err
	}
	// Dropout is conceptual
	hiddenStates = o.LayerNorm.Forward(hiddenStates.Add(inputTensor))
	return hiddenStates, nil
}

func (o *BertSelfOutput) Parameters() []*Tensor {
	params := o.Dense.Parameters()
	params = append(params, o.LayerNorm.Parameters()...)
	return params
}

// BertAttention combines self-attention with an output layer.
type BertAttention struct {
	Self   *BertSelfAttention
	Output *BertSelfOutput
}

func NewBertAttention(config BertConfig, initializerStdDev float64) *BertAttention {
	return &BertAttention{
		Self:   NewBertSelfAttention(config, initializerStdDev),
		Output: NewBertSelfOutput(config, initializerStdDev),
	}
}

func (a *BertAttention) Forward(hiddenStates *Tensor) (*Tensor, error) {
	selfAttentionOutput, err := a.Self.Forward(hiddenStates)
	if err != nil {
		return nil, err
	}
	attentionOutput, err := a.Output.Forward(selfAttentionOutput, hiddenStates)
	if err != nil {
		return nil, err
	}
	return attentionOutput, nil
}

func (a *BertAttention) Parameters() []*Tensor {
	params := a.Self.Parameters()
	params = append(params, a.Output.Parameters()...)
	return params
}

// BertIntermediate is the feed-forward layer in the transformer encoder.
type BertIntermediate struct {
	Dense *Linear
}

func NewBertIntermediate(config BertConfig, initializerStdDev float64) *BertIntermediate {
	return &BertIntermediate{
		Dense: NewLinear(config.HiddenSize, config.IntermediateSize, initializerStdDev),
	}
}

func (i *BertIntermediate) Forward(hiddenStates *Tensor) (*Tensor, error) {
	// Apply GELU activation
	// This is a simplification. A proper GELU tensor operation should be implemented.
	intermediateOutput,err := i.Dense.Forward(hiddenStates)
	if err != nil {
		return nil, err
	}
	// Conceptually: intermediateOutput = gelu(intermediateOutput)
	return intermediateOutput, nil
}

func (i *BertIntermediate) Parameters() []*Tensor {
	return i.Dense.Parameters()
}


// BertOutput handles the output of the BertAttention and BertIntermediate layers.
type BertOutput struct {
	Dense     *Linear
	LayerNorm *LayerNorm
	Dropout   float64
}

func NewBertOutput(config BertConfig, initializerStdDev float64) *BertOutput {
	return &BertOutput{
		Dense:     NewLinear(config.IntermediateSize, config.HiddenSize, initializerStdDev),
		LayerNorm: NewLayerNorm(config.HiddenSize, config.LayerNormEps),
		Dropout:   config.HiddenDropoutProb,
	}
}

func (o *BertOutput) Forward(hiddenStates, inputTensor *Tensor) (*Tensor, error) {
	hiddenStates,err := o.Dense.Forward(hiddenStates)
	if err != nil {
		return nil, err
	}
	// Dropout is conceptual
	hiddenStates = o.LayerNorm.Forward(hiddenStates.Add(inputTensor))
	return hiddenStates, nil
}

func (o *BertOutput) Parameters() []*Tensor {
	params := o.Dense.Parameters()
	params = append(params, o.LayerNorm.Parameters()...)
	return params
}


// BertLayer is a single transformer encoder layer.
type BertLayer struct {
	Attention    *BertAttention
	Intermediate *BertIntermediate
	Output       *BertOutput // This is different from standard BERT, but let's stick to the user's code structure
}

func NewBertLayer(config BertConfig, initializerStdDev float64) *BertLayer {
	return &BertLayer{
		Attention:    NewBertAttention(config, initializerStdDev),
		Intermediate: NewBertIntermediate(config, initializerStdDev),
		Output:       NewBertOutput(config, initializerStdDev),
	}
}

func (l *BertLayer) Forward(hiddenStates *Tensor) (*Tensor, error) {
	attentionOutput, err := l.Attention.Forward(hiddenStates)
	if err != nil {
		return nil, err
	}
	intermediateOutput, err := l.Intermediate.Forward(attentionOutput)
	if err != nil {
		return nil, err
	}
	layerOutput, err := l.Output.Forward(intermediateOutput, hiddenStates)
	if err != nil {
		return nil, err
	}
	return layerOutput, nil
}

func (l *BertLayer) Parameters() []*Tensor {
	params := l.Attention.Parameters()
	params = append(params, l.Intermediate.Parameters()...)
	params = append(params, l.Output.Parameters()...)
	return params
}


// BertEncoder stacks multiple BertLayers.
type BertEncoder struct {
	Layers []*BertLayer
}

func NewBertEncoder(config BertConfig, initializerStdDev float64) *BertEncoder {
	layers := make([]*BertLayer, config.NumHiddenLayers)
	for i := 0; i < config.NumHiddenLayers; i++ {
		layers[i] = NewBertLayer(config, initializerStdDev)
	}
	return &BertEncoder{Layers: layers}
}

func (e *BertEncoder) Forward(hiddenStates *Tensor) (*Tensor, error) {
	var err error
	for _, layer := range e.Layers {
		hiddenStates, err = layer.Forward(hiddenStates)
		if err != nil {
			return nil, err
		}
	}
	return hiddenStates, nil
}

func (e *BertEncoder) Parameters() []*Tensor {
	var params []*Tensor
	for _, layer := range e.Layers {
		params = append(params, layer.Parameters()...)
	}
	return params
}

// LayerNorm normalizes the activations of a layer.
type LayerNorm struct {
	Gamma   *Tensor
	Beta    *Tensor
	Epsilon float64
}

func NewLayerNorm(hiddenSize int, epsilon float64) *LayerNorm {
	gamma := NewTensor(nil, []int{hiddenSize}, true)
	beta := NewTensor(nil, []int{hiddenSize}, true)
	// Initialize gamma to 1s and beta to 0s
	for i := range gamma.Data {
		gamma.Data[i] = 1
	}
	return &LayerNorm{
		Gamma:   gamma,
		Beta:    beta,
		Epsilon: epsilon,
	}
}

func (ln *LayerNorm) Forward(x *Tensor) *Tensor {
	// LayerNorm forward pass. Normalization is performed across the last dimension.
	lastDim := x.Shape[len(x.Shape)-1]
	numVectors := tensorSize(x.Shape) / lastDim
	resultData := make([]float64, len(x.Data))

	// Store intermediate values for backward pass
	normalizedVectors := make([][]float64, numVectors)
	meanVectors := make([]float64, numVectors)
	invStdDevVectors := make([]float64, numVectors)

	for i := 0; i < numVectors; i++ {
		offset := i * lastDim
		vectorSlice := x.Data[offset : offset+lastDim]

		// Calculate mean
		sum := 0.0
		for _, val := range vectorSlice {
			sum += val
		}
		mean := sum / float64(lastDim)
		meanVectors[i] = mean

		// Calculate variance
		variance := 0.0
		for _, val := range vectorSlice {
			variance += math.Pow(val-mean, 2)
		}
		variance /= float64(lastDim)

		invStdDev := 1.0 / math.Sqrt(variance + ln.Epsilon)
		invStdDevVectors[i] = invStdDev

		normalizedVectors[i] = make([]float64, lastDim)
		// Normalize, scale, and shift
		for j, val := range vectorSlice {
			normalized := (val - mean) * invStdDev
			normalizedVectors[i][j] = normalized
			resultData[offset+j] = normalized*ln.Gamma.Data[j] + ln.Beta.Data[j]
		}
	}

	// We assume x.Grad != nil is the check for whether to compute gradients.
	// A RequiresGrad bool would be cleaner, but we follow the existing pattern.
	result := NewTensor(resultData, x.Shape, x.Grad != nil)
	if x.Grad != nil {
		result.creator = &op{
			inputs: []*Tensor{x, ln.Gamma, ln.Beta},
			backward: func() {
				// Backward pass for LayerNorm.
				// Based on compiler errors, we assume Tensor.Grad is of type []float64.
				if x.Grad == nil {
					x.Grad = make([]float64, len(x.Data))
				}
				if ln.Gamma.Grad == nil {
					ln.Gamma.Grad = make([]float64, len(ln.Gamma.Data))
				}
				if ln.Beta.Grad == nil {
					ln.Beta.Grad = make([]float64, len(ln.Beta.Data))
				}
				if result.Grad == nil {
					// The gradient of the output tensor must be initialized by the subsequent
					// operation's backward pass. If it's nil here, we can't proceed.
					// For safety, we can initialize it, but it should have been populated.
					result.Grad = make([]float64, len(result.Data))
				}

				gammaGrad := make([]float64, lastDim)
				betaGrad := make([]float64, lastDim)

				for i := 0; i < numVectors; i++ {
					offset := i * lastDim
					gradYSlice := result.Grad[offset : offset+lastDim]
					normalizedSlice := normalizedVectors[i]

					for j := 0; j < lastDim; j++ {
						betaGrad[j] += gradYSlice[j]
						gammaGrad[j] += gradYSlice[j] * normalizedSlice[j]
					}
				}

				for j := 0; j < lastDim; j++ {
					ln.Beta.Grad[j] += betaGrad[j]
					ln.Gamma.Grad[j] += gammaGrad[j]
				}

				for i := 0; i < numVectors; i++ {
					offset := i * lastDim
					gradYSlice := result.Grad[offset : offset+lastDim]
					vectorSlice := x.Data[offset : offset+lastDim]
					mean := meanVectors[i]
					invStdDev := invStdDevVectors[i]

					gradXHat := make([]float64, lastDim)
					for j := 0; j < lastDim; j++ {
						gradXHat[j] = gradYSlice[j] * ln.Gamma.Data[j]
					}

					gradInvStdDev := 0.0
					for j := 0; j < lastDim; j++ {
						gradInvStdDev += gradXHat[j] * (vectorSlice[j] - mean)
					}

					gradVariance := gradInvStdDev * -0.5 * math.Pow(invStdDev, 3)
					gradXCenteredFromVar := make([]float64, lastDim)
					for j := 0; j < lastDim; j++ {
						gradXCenteredFromVar[j] = gradVariance * (2.0 / float64(lastDim)) * (vectorSlice[j] - mean)
					}

					gradXCenteredFromXHat := make([]float64, lastDim)
					for j := 0; j < lastDim; j++ {
						gradXCenteredFromXHat[j] = gradXHat[j] * invStdDev
					}

					gradXCentered := make([]float64, lastDim)
					for j := 0; j < lastDim; j++ {
						gradXCentered[j] = gradXCenteredFromXHat[j] + gradXCenteredFromVar[j]
					}

					gradMean := 0.0
					for j := 0; j < lastDim; j++ {
						gradMean -= gradXCentered[j]
					}

					gradXFromMean := gradMean / float64(lastDim)

					for j := 0; j < lastDim; j++ {
						x.Grad[offset+j] += gradXCentered[j] + gradXFromMean
					}
				}
			},
		}
	}
	return result
}

func (ln *LayerNorm) Parameters() []*Tensor {
	return []*Tensor{ln.Gamma, ln.Beta}
}

// Helper functions for tensor manipulation that are missing from nlu.go
// These would need to be implemented properly.
//Helper functions for tensor manipulation that are missing from nlu.go
 // These would need to be implemented properly.
 func (t *Tensor) transposeForScores(numHeads, headSize int) *Tensor {
	// Placeholder
	// Reshape to (batch, seq_len, num_heads, head_size)
	newShape := []int{t.Shape[0], t.Shape[1], numHeads, headSize}
	reshaped, err := t.Reshape(newShape)
	if err != nil {
		panic(err)
 	}
	// Transpose to (batch, num_heads, seq_len, head_size)
	transposed, err := reshaped.Transpose(1, 2)
	if err != nil {
		panic(err)
	}
	return transposed
}
 
 func (t *Tensor) transpose(dim1, dim2 int) *Tensor {
	transposed, err := t.Transpose(dim1, dim2)
	if err != nil {
		panic(err)
 	}
	return transposed
}
 
 func (t *Tensor) DivScalar(val float64) *Tensor {
	// This should be implemented in tensor.go
	result := make([]float64, len(t.Data))
	for i, v := range t.Data {
		result[i] = v / val
 	}
	return &Tensor{Data: result, Shape: t.Shape}
}
 
 func (t *Tensor) slice(start, end int) *Tensor {
	// This is a simplified slice implementation.
	// It assumes slicing along the first dimension.
	if len(t.Shape) == 0 {
		return t // Cannot slice a scalar
 	}
	if start < 0 {
		start = 0
	}
	if end > t.Shape[0] {
		end = t.Shape[0]
	}
	if start >= end {
		return &Tensor{Data: []float64{}, Shape: []int{0}}
	}

	rowsToSlice := end - start
	stride := 1
	for i := 1; i < len(t.Shape); i++ {
		stride *= t.Shape[i]
	}

	newData := t.Data[start*stride : end*stride]
	newShape := make([]int, len(t.Shape))
	copy(newShape, t.Shape)
	newShape[0] = rowsToSlice

	return &Tensor{Data: newData, Shape: newShape}
}

// Pooler is the final layer in the BERT model, which is typically a simple feed-forward network.
type Pooler struct {
	Dense     *Linear
	Activation func(x *Tensor) *Tensor
}

func NewPooler(config BertConfig, initializerStdDev float64) *Pooler {
	return &Pooler{
		Dense:     NewLinear(config.HiddenSize, config.HiddenSize, initializerStdDev),
		Activation: func(x *Tensor) *Tensor { return x.Tanh() }, // Using Tanh as the activation
	}
}

func (p *Pooler) Forward(hiddenStates *Tensor) (*Tensor, error) {
	// For pooler, we typically use the hidden state of the [CLS] token, which is the first token.
	clsHiddenState := hiddenStates.slice(0, 1) // Shape (batch_size, 1, hidden_size)

	// Pass through the dense layer and activation
	poolerOutput, err := p.Dense.Forward(clsHiddenState)
	if err != nil {
		return nil, err
	}
	fmt.Printf("Input shape to pooler: %v\n", clsHiddenState.Shape)
	fmt.Printf("Pooler weight shape: %v\n", p.Dense.Weight.Shape)
	poolerOutput = p.Activation(poolerOutput)

	return poolerOutput, nil
}

func (p *Pooler) Parameters() []*Tensor {
	return p.Dense.Parameters()
}
