package model

import (
	"encoding/gob"
	"fmt"
	"log"
	"os"

	"github.com/zendrulat/nlptagger/neural/nn"
	"github.com/zendrulat/nlptagger/neural/nnu/bert"
	"github.com/zendrulat/nlptagger/neural/tensor"
)

func init() {
	gob.Register(bert.BertConfig{})
}

// DecoderStepState holds the state of a single decoder time step.
type DecoderStepState struct {
	LSTMCell      *nn.LSTMCell
	EmbeddedInput *tensor.Tensor
	DecoderHidden *tensor.Tensor
	DecoderCell   *tensor.Tensor
}

// MoEClassificationModel is a wrapper around the MoELayer to make it a trainable model.
type MoEClassificationModel struct {
	Embedding             *nn.Embedding
	BertModel             *bert.BertModel
	ParentClassifier      *nn.Linear
	ChildClassifier       *nn.Linear
	SentenceEmbedding     *nn.Embedding
	SentenceDecoderTemplate *nn.LSTMCell
	SentenceClassifier    *nn.Linear
	BertConfig            bert.BertConfig
	pooledOutput          *tensor.Tensor // Add this field
	decoderStates         []*DecoderStepState
}

func NewMoEClassificationModel(vocabSize, embeddingDim, numParentClasses, numChildClasses, sentenceVocabSize, numExperts, k, maxSeqLength int) (*MoEClassificationModel, error) {
	embedding := nn.NewEmbedding(vocabSize, embeddingDim)

	bertConfig := bert.BertConfig{
		VocabSize:             vocabSize,
		HiddenSize:            embeddingDim,
		MaxPositionEmbeddings: maxSeqLength,
		NumHiddenLayers:       4,
		NumAttentionHeads:     4,
		IntermediateSize:      embeddingDim * 4,
		TypeVocabSize:         2,
		InitializerRange:      0.02,
	}
	bertModel := bert.NewBertModel(bertConfig, nil)

	parentClassifier, err := nn.NewLinear(embeddingDim, numParentClasses)
	if err != nil {
		return nil, err
	}
	childClassifier, err := nn.NewLinear(embeddingDim, numChildClasses)
	if err != nil {
		return nil, err
	}

	sentenceEmbedding := nn.NewEmbedding(sentenceVocabSize, embeddingDim)
	sentenceDecoderTemplate, err := nn.NewLSTMCell(embeddingDim, embeddingDim)
	if err != nil {
		return nil, err
	}
	sentenceClassifier, err := nn.NewLinear(embeddingDim, sentenceVocabSize)
	if err != nil {
		return nil, err
	}

	return &MoEClassificationModel{
		Embedding:             embedding,
		BertModel:             bertModel,
		ParentClassifier:      parentClassifier,
		ChildClassifier:       childClassifier,
		SentenceEmbedding:     sentenceEmbedding,
		SentenceDecoderTemplate: sentenceDecoderTemplate,
		SentenceClassifier:    sentenceClassifier,
		BertConfig:            bertConfig,
	}, nil
}

func (m *MoEClassificationModel) SetMode(training bool) {
}

func (m *MoEClassificationModel) Forward(inputs ...*tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor, error) {
	inputIDs := inputs[0]
	targetSentenceIDs := inputs[1] // New: target sentence for teacher forcing

	batchSize, seqLength := inputIDs.Shape[0], inputIDs.Shape[1]

	tokenTypeIDs := tensor.NewTensor([]int{batchSize, seqLength}, make([]float64, batchSize*seqLength), false)
	posTagIDs := tensor.NewTensor([]int{batchSize, seqLength}, make([]float64, batchSize*seqLength), false)
	nerTagIDs := tensor.NewTensor([]int{batchSize, seqLength}, make([]float64, batchSize*seqLength), false)

	bertOutput, err := m.BertModel.Forward(inputIDs, tokenTypeIDs, posTagIDs, nerTagIDs)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("bert model forward failed: %w", err)
	}

	clsTokenOutput, err := bertOutput.Slice(1, 0, 1)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to slice CLS token output: %w", err)
	}

	clsTokenOutputReshaped, err := clsTokenOutput.Reshape([]int{batchSize, m.BertConfig.HiddenSize})
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to reshape CLS token output: %w", err)
	}

	parentLogits, err := m.ParentClassifier.Forward(clsTokenOutputReshaped)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("parent classifier forward failed: %w", err)
	}

	childLogits, err := m.ChildClassifier.Forward(clsTokenOutputReshaped)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("child classifier forward failed: %w", err)
	}

	// Decoder part
	targetSeqLen := targetSentenceIDs.Shape[1]
	sentenceVocabSize := m.SentenceClassifier.Weights.Shape[1]
	decoderHidden := clsTokenOutputReshaped
	decoderCell := tensor.NewTensor(decoderHidden.Shape, make([]float64, len(decoderHidden.Data)), true) // Initial cell state

	allLstmOutputs := make([]*tensor.Tensor, targetSeqLen)
	m.decoderStates = make([]*DecoderStepState, targetSeqLen)

	for t := 0; t < targetSeqLen; t++ {
		decoderInputData := make([]float64, batchSize)
		for b := 0; b < batchSize; b++ {
			decoderInputData[b] = targetSentenceIDs.Data[b*targetSeqLen+t]
		}
		decoderInput := tensor.NewTensor([]int{batchSize, 1}, decoderInputData, true)

		embedded, err := m.SentenceEmbedding.Forward(decoderInput)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("decoder embedding failed: %w", err)
		}

		embeddedReshaped, err := embedded.Reshape([]int{batchSize, m.BertConfig.HiddenSize})
		if err != nil {
			return nil, nil, nil, fmt.Errorf("decoder embedding reshape failed: %w", err)
		}

		cell := &nn.LSTMCell{
			InputSize:  m.SentenceDecoderTemplate.InputSize,
			HiddenSize: m.SentenceDecoderTemplate.HiddenSize,
			Wf:         m.SentenceDecoderTemplate.Wf,
			Wi:         m.SentenceDecoderTemplate.Wi,
			Wc:         m.SentenceDecoderTemplate.Wc,
			Wo:         m.SentenceDecoderTemplate.Wo,
			Bf:         m.SentenceDecoderTemplate.Bf,
			Bi:         m.SentenceDecoderTemplate.Bi,
			Bc:         m.SentenceDecoderTemplate.Bc,
			Bo:         m.SentenceDecoderTemplate.Bo,
		}

		lstmOutput, c, err := cell.Forward(embeddedReshaped, decoderHidden, decoderCell)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("decoder lstm failed: %+v", err)
		}

		m.decoderStates[t] = &DecoderStepState{
			LSTMCell:      cell,
			EmbeddedInput: embeddedReshaped,
			DecoderHidden: decoderHidden,
			DecoderCell:   decoderCell,
		}

		decoderHidden = lstmOutput
		decoderCell = c
		allLstmOutputs[t] = lstmOutput
	}

	// Concatenate all lstmOutputs
	concatenatedLstmOutputsData := make([]float64, batchSize*targetSeqLen*m.BertConfig.HiddenSize)
	for t := 0; t < targetSeqLen; t++ {
		copy(concatenatedLstmOutputsData[t*batchSize*m.BertConfig.HiddenSize:(t+1)*batchSize*m.BertConfig.HiddenSize], allLstmOutputs[t].Data)
	}
	concatenatedLstmOutputs := tensor.NewTensor([]int{batchSize * targetSeqLen, m.BertConfig.HiddenSize}, concatenatedLstmOutputsData, true)

	// Pass concatenated outputs to SentenceClassifier
	predictions, err := m.SentenceClassifier.Forward(concatenatedLstmOutputs)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("decoder output linear layer failed: %w", err)
	}

	// Reshape predictions back to [batchSize, targetSeqLen, sentenceVocabSize]
	decoderOutputs, err := predictions.Reshape([]int{batchSize, targetSeqLen, sentenceVocabSize})
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to reshape decoder outputs: %w", err)
	}

	return parentLogits, childLogits, decoderOutputs, nil
}

func (m *MoEClassificationModel) Backward(parentGrad, childGrad, sentenceGrad *tensor.Tensor) error {
	// Backward for sentence generation
	reshapedSentenceGrad, err := sentenceGrad.Reshape([]int{sentenceGrad.Shape[0] * sentenceGrad.Shape[1], sentenceGrad.Shape[2]})
	if err != nil {
		return fmt.Errorf("failed to reshape sentenceGrad for SentenceClassifier backward: %w", err)
	}
	err = m.SentenceClassifier.Backward(reshapedSentenceGrad)
	if err != nil {
		return fmt.Errorf("sentence classifier backward failed: %w", err)
	}

	gradAllLstmOutputs := m.SentenceClassifier.Input().Grad
	batchSize := sentenceGrad.Shape[0]
	targetSeqLen := sentenceGrad.Shape[1]
	hiddenSize := m.BertConfig.HiddenSize

	gradDecoderHidden := tensor.NewTensor([]int{batchSize, hiddenSize}, make([]float64, batchSize*hiddenSize), true)
	gradDecoderCell := tensor.NewTensor([]int{batchSize, hiddenSize}, make([]float64, batchSize*hiddenSize), true)

	for t := targetSeqLen - 1; t >= 0; t-- {
		stepState := m.decoderStates[t]
		cell := stepState.LSTMCell

		offset := t * batchSize * hiddenSize
		gradLstmOutput := tensor.NewTensor(
			[]int{batchSize, hiddenSize},
			gradAllLstmOutputs.Data[offset:offset+batchSize*hiddenSize],
			true,
		)
		gradLstmOutput.Add(gradDecoderHidden)

		err := cell.Backward(gradLstmOutput, gradDecoderCell)
		if err != nil {
			return fmt.Errorf("sentence decoder backward failed at step %d: %w", t, err)
		}

		gradDecoderHidden = cell.PrevHidden.Grad
		gradDecoderCell = cell.PrevCell.Grad

		err = m.SentenceEmbedding.Backward(cell.InputTensor.Grad)
		if err != nil {
			return fmt.Errorf("sentence embedding backward failed at step %d: %w", t, err)
		}
	}

	// Backward for classifiers
	err = m.ParentClassifier.Backward(parentGrad)
	if err != nil {
		return fmt.Errorf("parent classifier backward failed: %w", err)
	}

	err = m.ChildClassifier.Backward(childGrad)
	if err != nil {
		return fmt.Errorf("child classifier backward failed: %w", err)
	}

	clsTokenGrad := m.ParentClassifier.Input().Grad
	clsTokenGrad.Add(gradDecoderHidden) // Add gradient from decoder's initial hidden state

	bertGrad := tensor.NewTensor([]int{clsTokenGrad.Shape[0], m.BertConfig.MaxPositionEmbeddings, m.BertConfig.HiddenSize}, make([]float64, clsTokenGrad.Shape[0]*m.BertConfig.MaxPositionEmbeddings*m.BertConfig.HiddenSize), false)

	batchSize = clsTokenGrad.Shape[0]
	hiddenSize = clsTokenGrad.Shape[1]

	for i := 0; i < batchSize; i++ {
		copy(bertGrad.Data[i*m.BertConfig.MaxPositionEmbeddings*hiddenSize:i*m.BertConfig.MaxPositionEmbeddings*hiddenSize+hiddenSize], clsTokenGrad.Data[i*hiddenSize:(i+1)*hiddenSize])
	}

	err = m.BertModel.Backward(bertGrad)
	if err != nil {
		return fmt.Errorf("bert model backward failed: %w", err)
	}

	return nil
}

func (m *MoEClassificationModel) Parameters() []*tensor.Tensor {
	var params []*tensor.Tensor
	if m.BertModel != nil {
		params = append(params, m.BertModel.Parameters()...)
	}
	params = append(params, m.ParentClassifier.Parameters()...)
	params = append(params, m.ChildClassifier.Parameters()...)
	params = append(params, m.SentenceEmbedding.Parameters()...)
	params = append(params, m.SentenceDecoderTemplate.Parameters()...)
	params = append(params, m.SentenceClassifier.Parameters()...)
	return params
}


// Description prints a summary of the MoEClassificationModel's architecture.
func (m *MoEClassificationModel) Description() {
}

// SaveMoEClassificationModelToGOB saves the MoEClassificationModel to a file in Gob format.
func SaveMoEClassificationModelToGOB(model *MoEClassificationModel, filePath string) error {
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to create file for saving MoE model: %w", err)
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	if err := encoder.Encode(model); err != nil {
		return fmt.Errorf("failed to encode MoE model to Gob: %w", err)
	}

	return nil
}

// LoadMoEClassificationModelFromGOB loads a MoEClassificationModel from a file in Gob format.
package model

import (
	"encoding/gob"
	"fmt"
	"log"
	"os"

	"github.com/zendrulat/nlptagger/neural/nn"
	"github.com/zendrulat/nlptagger/neural/nnu/bert"
	"github.com/zendrulat/nlptagger/neural/tensor"
)

func init() {
	gob.Register(bert.BertConfig{})
}

// DecoderStepState holds the state of a single decoder time step.
type DecoderStepState struct {
	LSTMCell      *nn.LSTMCell
	EmbeddedInput *tensor.Tensor
	DecoderHidden *tensor.Tensor
	DecoderCell   *tensor.Tensor
}

// MoEClassificationModel is a wrapper around the MoELayer to make it a trainable model.
type MoEClassificationModel struct {
	Embedding             *nn.Embedding
	BertModel             *bert.BertModel
	ParentClassifier      *nn.Linear
	ChildClassifier       *nn.Linear
	SentenceEmbedding     *nn.Embedding
	SentenceDecoderTemplate *nn.LSTMCell
	SentenceClassifier    *nn.Linear
	BertConfig            bert.BertConfig
	pooledOutput          *tensor.Tensor // Add this field
	decoderStates         []*DecoderStepState
}

func NewMoEClassificationModel(vocabSize, embeddingDim, numParentClasses, numChildClasses, sentenceVocabSize, numExperts, k, maxSeqLength int) (*MoEClassificationModel, error) {
	embedding := nn.NewEmbedding(vocabSize, embeddingDim)

	bertConfig := bert.BertConfig{
		VocabSize:             vocabSize,
		HiddenSize:            embeddingDim,
		MaxPositionEmbeddings: maxSeqLength,
		NumHiddenLayers:       4,
		NumAttentionHeads:     4,
		IntermediateSize:      embeddingDim * 4,
		TypeVocabSize:         2,
		InitializerRange:      0.02,
	}
	bertModel := bert.NewBertModel(bertConfig, nil)

	parentClassifier, err := nn.NewLinear(embeddingDim, numParentClasses)
	if err != nil {
		return nil, err
	}
	childClassifier, err := nn.NewLinear(embeddingDim, numChildClasses)
	if err != nil {
		return nil, err
	}

	sentenceEmbedding := nn.NewEmbedding(sentenceVocabSize, embeddingDim)
	sentenceDecoderTemplate, err := nn.NewLSTMCell(embeddingDim, embeddingDim)
	if err != nil {
		return nil, err
	}
	sentenceClassifier, err := nn.NewLinear(embeddingDim, sentenceVocabSize)
	if err != nil {
		return nil, err
	}

	return &MoEClassificationModel{
		Embedding:             embedding,
		BertModel:             bertModel,
		ParentClassifier:      parentClassifier,
		ChildClassifier:       childClassifier,
		SentenceEmbedding:     sentenceEmbedding,
		SentenceDecoderTemplate: sentenceDecoderTemplate,
		SentenceClassifier:    sentenceClassifier,
		BertConfig:            bertConfig,
	}, nil
}

func (m *MoEClassificationModel) SetMode(training bool) {
}

func (m *MoEClassificationModel) Forward(inputs ...*tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor, error) {
	inputIDs := inputs[0]
	targetSentenceIDs := inputs[1] // New: target sentence for teacher forcing

	batchSize, seqLength := inputIDs.Shape[0], inputIDs.Shape[1]

	tokenTypeIDs := tensor.NewTensor([]int{batchSize, seqLength}, make([]float64, batchSize*seqLength), false)
	posTagIDs := tensor.NewTensor([]int{batchSize, seqLength}, make([]float64, batchSize*seqLength), false)
	nerTagIDs := tensor.NewTensor([]int{batchSize, seqLength}, make([]float64, batchSize*seqLength), false)

	bertOutput, err := m.BertModel.Forward(inputIDs, tokenTypeIDs, posTagIDs, nerTagIDs)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("bert model forward failed: %w", err)
	}

	clsTokenOutput, err := bertOutput.Slice(1, 0, 1)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to slice CLS token output: %w", err)
	}

	clsTokenOutputReshaped, err := clsTokenOutput.Reshape([]int{batchSize, m.BertConfig.HiddenSize})
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to reshape CLS token output: %w", err)
	}

	parentLogits, err := m.ParentClassifier.Forward(clsTokenOutputReshaped)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("parent classifier forward failed: %w", err)
	}

	childLogits, err := m.ChildClassifier.Forward(clsTokenOutputReshaped)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("child classifier forward failed: %w", err)
	}

	// Decoder part
	targetSeqLen := targetSentenceIDs.Shape[1]
	sentenceVocabSize := m.SentenceClassifier.Weights.Shape[1]
	decoderHidden := clsTokenOutputReshaped
	decoderCell := tensor.NewTensor(decoderHidden.Shape, make([]float64, len(decoderHidden.Data)), true) // Initial cell state

	allLstmOutputs := make([]*tensor.Tensor, targetSeqLen)
	m.decoderStates = make([]*DecoderStepState, targetSeqLen)

	for t := 0; t < targetSeqLen; t++ {
		decoderInputData := make([]float64, batchSize)
		for b := 0; b < batchSize; b++ {
			decoderInputData[b] = targetSentenceIDs.Data[b*targetSeqLen+t]
		}
		decoderInput := tensor.NewTensor([]int{batchSize, 1}, decoderInputData, true)

		embedded, err := m.SentenceEmbedding.Forward(decoderInput)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("decoder embedding failed: %w", err)
		}

		embeddedReshaped, err := embedded.Reshape([]int{batchSize, m.BertConfig.HiddenSize})
		if err != nil {
			return nil, nil, nil, fmt.Errorf("decoder embedding reshape failed: %w", err)
	}

		cell := &nn.LSTMCell{
			InputSize:  m.SentenceDecoderTemplate.InputSize,
			HiddenSize: m.SentenceDecoderTemplate.HiddenSize,
			Wf:         m.SentenceDecoderTemplate.Wf,
			Wi:         m.SentenceDecoderTemplate.Wi,
			Wc:         m.SentenceDecoderTemplate.Wc,
			Wo:         m.SentenceDecoderTemplate.Wo,
			Bf:         m.SentenceDecoderTemplate.Bf,
			Bi:         m.SentenceDecoderTemplate.Bi,
			Bc:         m.SentenceDecoderTemplate.Bc,
			Bo:         m.SentenceDecoderTemplate.Bo,
		}

		lstmOutput, c, err := cell.Forward(embeddedReshaped, decoderHidden, decoderCell)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("decoder lstm failed: %+v", err)
		}

		m.decoderStates[t] = &DecoderStepState{
			LSTMCell:      cell,
			EmbeddedInput: embeddedReshaped,
			DecoderHidden: decoderHidden,
			DecoderCell:   decoderCell,
		}

		decoderHidden = lstmOutput
		decoderCell = c
		allLstmOutputs[t] = lstmOutput
	}

	// Concatenate all lstmOutputs
	concatenatedLstmOutputsData := make([]float64, batchSize*targetSeqLen*m.BertConfig.HiddenSize)
	for t := 0; t < targetSeqLen; t++ {
		copy(concatenatedLstmOutputsData[t*batchSize*m.BertConfig.HiddenSize:(t+1)*batchSize*m.BertConfig.HiddenSize], allLstmOutputs[t].Data)
	}
	concatenatedLstmOutputs := tensor.NewTensor([]int{batchSize * targetSeqLen, m.BertConfig.HiddenSize}, concatenatedLstmOutputsData, true)

	// Pass concatenated outputs to SentenceClassifier
	predictions, err := m.SentenceClassifier.Forward(concatenatedLstmOutputs)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("decoder output linear layer failed: %w", err)
	}

	// Reshape predictions back to [batchSize, targetSeqLen, sentenceVocabSize]
	decoderOutputs, err := predictions.Reshape([]int{batchSize, targetSeqLen, sentenceVocabSize})
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to reshape decoder outputs: %w", err)
	}

	return parentLogits, childLogits, decoderOutputs, nil
}

func (m *MoEClassificationModel) Backward(parentGrad, childGrad, sentenceGrad *tensor.Tensor) error {
	// Backward for sentence generation
	reshapedSentenceGrad, err := sentenceGrad.Reshape([]int{sentenceGrad.Shape[0] * sentenceGrad.Shape[1], sentenceGrad.Shape[2]})
	if err != nil {
		return fmt.Errorf("failed to reshape sentenceGrad for SentenceClassifier backward: %w", err)
	}
	err = m.SentenceClassifier.Backward(reshapedSentenceGrad)
	if err != nil {
		return fmt.Errorf("sentence classifier backward failed: %w", err)
	}

	gradAllLstmOutputs := m.SentenceClassifier.Input().Grad
	batchSize := sentenceGrad.Shape[0]
	targetSeqLen := sentenceGrad.Shape[1]
	hiddenSize := m.BertConfig.HiddenSize

	gradDecoderHidden := tensor.NewTensor([]int{batchSize, hiddenSize}, make([]float64, batchSize*hiddenSize), true)
	gradDecoderCell := tensor.NewTensor([]int{batchSize, hiddenSize}, make([]float64, batchSize*hiddenSize), true)

	for t := targetSeqLen - 1; t >= 0; t-- {
		stepState := m.decoderStates[t]
		cell := stepState.LSTMCell

		offset := t * batchSize * hiddenSize
		gradLstmOutput := tensor.NewTensor(
			[]int{batchSize, hiddenSize},
			gradAllLstmOutputs.Data[offset:offset+batchSize*hiddenSize],
			true,
		)
		gradLstmOutput.Add(gradDecoderHidden)

		err := cell.Backward(gradLstmOutput, gradDecoderCell)
		if err != nil {
			return fmt.Errorf("sentence decoder backward failed at step %d: %w", t, err)
		}

		gradDecoderHidden = cell.PrevHidden.Grad
		gradDecoderCell = cell.PrevCell.Grad

		err = m.SentenceEmbedding.Backward(cell.InputTensor.Grad)
		if err != nil {
			return fmt.Errorf("sentence embedding backward failed at step %d: %w", t, err)
		}
	}

	// Backward for classifiers
	err = m.ParentClassifier.Backward(parentGrad)
	if err != nil {
		return fmt.Errorf("parent classifier backward failed: %w", err)
	}

	err = m.ChildClassifier.Backward(childGrad)
	if err != nil {
		return fmt.Errorf("child classifier backward failed: %w", err)
	}

	clsTokenGrad := m.ParentClassifier.Input().Grad
	clsTokenGrad.Add(gradDecoderHidden) // Add gradient from decoder's initial hidden state

	bertGrad := tensor.NewTensor([]int{clsTokenGrad.Shape[0], m.BertConfig.MaxPositionEmbeddings, m.BertConfig.HiddenSize}, make([]float64, clsTokenGrad.Shape[0]*m.BertConfig.MaxPositionEmbeddings*m.BertConfig.HiddenSize), false)

	batchSize = clsTokenGrad.Shape[0]
	hiddenSize = clsTokenGrad.Shape[1]

	for i := 0; i < batchSize; i++ {
		copy(bertGrad.Data[i*m.BertConfig.MaxPositionEmbeddings*hiddenSize:i*m.BertConfig.MaxPositionEmbeddings*hiddenSize+hiddenSize], clsTokenGrad.Data[i*hiddenSize:(i+1)*hiddenSize])
	}

	err = m.BertModel.Backward(bertGrad)
	if err != nil {
		return fmt.Errorf("bert model backward failed: %w", err)
	}

	return nil
}

func (m *MoEClassificationModel) Parameters() []*tensor.Tensor {
	var params []*tensor.Tensor
	if m.BertModel != nil {
		params = append(params, m.BertModel.Parameters()...)
	}
	params = append(params, m.ParentClassifier.Parameters()...)
	params = append(params, m.ChildClassifier.Parameters()...)
	params = append(params, m.SentenceEmbedding.Parameters()...)
	params = append(params, m.SentenceDecoderTemplate.Parameters()...)
	params = append(params, m.SentenceClassifier.Parameters()...)
	return params
}


// Description prints a summary of the MoEClassificationModel's architecture.
func (m *MoEClassificationModel) Description() {
}

// SaveMoEClassificationModelToGOB saves the MoEClassificationModel to a file in Gob format.
func SaveMoEClassificationModelToGOB(model *MoEClassificationModel, filePath string) error {
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to create file for saving MoE model: %w", err)
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	if err := encoder.Encode(model); err != nil {
		return fmt.Errorf("failed to encode MoE model to Gob: %w", err)
	}

	return nil
}

// LoadMoEClassificationModelFromGOB loads a MoEClassificationModel from a file in Gob format.
func LoadMoEClassificationModelFromGOB(filePath string, vocabSize, numParentClasses, numChildClasses, maxSeqLength int) (*MoEClassificationModel, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("error opening MoE model gob file: %w", err)
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	var loadedModel MoEClassificationModel
	err = decoder.Decode(&loadedModel)
	if err != nil {
		return nil, fmt.Errorf("error decoding MoE model from gob: %w. It's possible the model file was created with a different training script (e.g. train_moe instead of train_intent_classifier)", err)
	}

	// Check if the loaded model is of the correct type.
	// If BertConfig.HiddenSize is 0, it's likely that the wrong model type was loaded.
	if loadedModel.BertConfig.HiddenSize == 0 {
		return nil, fmt.Errorf("loaded model BertConfig has a HiddenSize of 0, which is invalid. It's likely that the model file at %s was created with a different training script (e.g. train_moe) than the one expected by this inference tool (train_intent_classifier)", filePath)
	}

	// --- Start of explicit BertModel re-creation and weight copying ---
	// Update BertConfig's VocabSize with the provided vocabSize
	loadedModel.BertConfig.VocabSize = vocabSize
	loadedModel.BertConfig.MaxPositionEmbeddings = maxSeqLength // Add this line
	// Create a new BertModel using the loaded BertConfig
	newBertModel := bert.NewBertModel(loadedModel.BertConfig, nil)

	// Copy WordEmbeddings weights
	if loadedModel.BertModel != nil && loadedModel.BertModel.BertEmbeddings != nil && loadedModel.BertModel.BertEmbeddings.WordEmbeddings != nil && loadedModel.BertModel.BertEmbeddings.WordEmbeddings.Weight != nil {
		// Determine the minimum length to copy to avoid out-of-bounds access
		minLen := len(loadedModel.BertModel.BertEmbeddings.WordEmbeddings.Weight.Data)
		if len(newBertModel.BertEmbeddings.WordEmbeddings.Weight.Data) < minLen {
			minLen = len(newBertModel.BertEmbeddings.WordEmbeddings.Weight.Data)
		}
		copy(newBertModel.BertEmbeddings.WordEmbeddings.Weight.Data[:minLen], loadedModel.BertModel.BertEmbeddings.WordEmbeddings.Weight.Data[:minLen])
		if len(newBertModel.BertEmbeddings.WordEmbeddings.Weight.Data) > minLen {
			log.Printf("WARNING: New WordEmbeddings size (%d) is larger than loaded size (%d). New embeddings will be randomly initialized for new tokens.", len(newBertModel.BertEmbeddings.WordEmbeddings.Weight.Data), minLen)
		}
	} else {
		log.Printf("WARNING: WordEmbeddings not found in loaded model, using newly initialized weights.")
	}

	// Copy PositionEmbeddings weights
	if loadedModel.BertModel != nil && loadedModel.BertModel.BertEmbeddings != nil && loadedModel.BertModel.BertEmbeddings.PositionEmbeddings != nil && loadedModel.BertModel.BertEmbeddings.PositionEmbeddings.Weight != nil {
		minLen := len(loadedModel.BertModel.BertEmbeddings.PositionEmbeddings.Weight.Data)
		if len(newBertModel.BertEmbeddings.PositionEmbeddings.Weight.Data) < minLen {
			minLen = len(newBertModel.BertEmbeddings.PositionEmbeddings.Weight.Data)
		}
		copy(newBertModel.BertEmbeddings.PositionEmbeddings.Weight.Data[:minLen], loadedModel.BertModel.BertEmbeddings.PositionEmbeddings.Weight.Data[:minLen])
	} else {
		log.Printf("WARNING: PositionEmbeddings not found in loaded model, using newly initialized weights.")
	}

	// Copy TokenTypeEmbeddings weights
	if loadedModel.BertModel != nil && loadedModel.BertModel.BertEmbeddings != nil && loadedModel.BertModel.BertEmbeddings.TokenTypeEmbeddings != nil && loadedModel.BertModel.BertEmbeddings.TokenTypeEmbeddings.Weight != nil {
		minLen := len(loadedModel.BertModel.BertEmbeddings.TokenTypeEmbeddings.Weight.Data)
		if len(newBertModel.BertEmbeddings.TokenTypeEmbeddings.Weight.Data) < minLen {
			minLen = len(newBertModel.BertEmbeddings.TokenTypeEmbeddings.Weight.Data)
		}
		copy(newBertModel.BertEmbeddings.TokenTypeEmbeddings.Weight.Data[:minLen], loadedModel.BertModel.BertEmbeddings.TokenTypeEmbeddings.Weight.Data[:minLen])
		if len(newBertModel.BertEmbeddings.TokenTypeEmbeddings.Weight.Data) > minLen {
			log.Printf("WARNING: New TokenTypeEmbeddings size (%d) is larger than loaded size (%d). New embeddings will be randomly initialized for new token types.", len(newBertModel.BertEmbeddings.TokenTypeEmbeddings.Weight.Data), minLen)
		}
	} else {
		log.Printf("WARNING: TokenTypeEmbeddings not found in loaded model, using newly initialized weights.")
	}

	// Assign the newly created and weight-copied BertModel
	loadedModel.BertModel = newBertModel
	// --- End of explicit BertModel re-creation and weight copying ---

	// Re-initialize the Activation function in the Pooler after loading,
	// as gob cannot serialize/deserialize functions.
	if loadedModel.BertModel != nil && loadedModel.BertModel.Pooler != nil {
		loadedModel.BertModel.Pooler.Activation = func(x *tensor.Tensor) *tensor.Tensor {
			t, _ := x.Tanh()
			return t
		}
	}

	// Re-initialize ParentClassifier and ChildClassifier after loading,
	// as gob might not correctly deserialize their internal *tensor.Tensor fields.
	if loadedModel.ParentClassifier == nil {
		parentClassifier, err := nn.NewLinear(loadedModel.BertConfig.HiddenSize, numParentClasses)
		if err != nil {
			return nil, fmt.Errorf("failed to re-initialize parent classifier: %w", err)
		}
		loadedModel.ParentClassifier = parentClassifier
	} else if loadedModel.ParentClassifier != nil {
		// Store deserialized weights and biases
		parentWeightsData := loadedModel.ParentClassifier.Weights.Data
		parentBiasesData := loadedModel.ParentClassifier.Biases.Data
		parentInputDim := loadedModel.ParentClassifier.Weights.Shape[0]

		// Create new Linear layer
		newParentClassifier, err := nn.NewLinear(parentInputDim, numParentClasses) // Use numParentClasses
		if err != nil {
			return nil, fmt.Errorf("failed to re-initialize parent classifier: %%w", err)
		}

		// Copy deserialized data to new Linear layer
		var minLen int
		minLen = len(parentWeightsData)
		if len(newParentClassifier.Weights.Data) < minLen {
			minLen = len(newParentClassifier.Weights.Data)
		}
		copy(newParentClassifier.Weights.Data[:minLen], parentWeightsData[:minLen])

		minLen = len(parentBiasesData)
		if len(newParentClassifier.Biases.Data) < minLen {
			minLen = len(newParentClassifier.Biases.Data)
		}
		copy(newParentClassifier.Biases.Data[:minLen], parentBiasesData[:minLen])

		loadedModel.ParentClassifier = newParentClassifier
	}

	if loadedModel.ChildClassifier == nil {
		childClassifier, err := nn.NewLinear(loadedModel.BertConfig.HiddenSize, numChildClasses)
		if err != nil {
			return nil, fmt.Errorf("failed to re-initialize child classifier: %%w", err)
		}
		loadedModel.ChildClassifier = childClassifier
	} else if loadedModel.ChildClassifier != nil {
		// Store deserialized weights and biases
		childWeightsData := loadedModel.ChildClassifier.Weights.Data
		childBiasesData := loadedModel.ChildClassifier.Biases.Data
		childInputDim := loadedModel.ChildClassifier.Weights.Shape[0]

		// Create new Linear layer
		newChildClassifier, err := nn.NewLinear(childInputDim, numChildClasses) // Use numChildClasses
		if err != nil {
			return nil, fmt.Errorf("failed to re-initialize child classifier: %%w", err)
		}

		// Copy deserialized data to new Linear layer
		var minLen int
		minLen = len(childWeightsData)
		if len(newChildClassifier.Weights.Data) < minLen {
			minLen = len(newChildClassifier.Weights.Data)
		}
		copy(newChildClassifier.Weights.Data[:minLen], childWeightsData[:minLen])

		minLen = len(childBiasesData)
		if len(newChildClassifier.Biases.Data) < minLen {
			minLen = len(newChildClassifier.Biases.Data)
		}
		copy(newChildClassifier.Biases.Data[:minLen], childBiasesData[:minLen])

		loadedModel.ChildClassifier = newChildClassifier
	}

	return &loadedModel, nil
}