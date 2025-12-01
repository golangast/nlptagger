package model

import (
	"encoding/gob"
	"fmt"
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