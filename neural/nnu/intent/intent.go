package intent

import (
	"encoding/gob"
	"os"

	. "github.com/zendrulat/nlptagger/neural/nn"
	. "github.com/zendrulat/nlptagger/neural/tensor"
)

// NewSimpleIntentClassifier creates a new SimpleIntentClassifier model.
func NewSimpleIntentClassifier(vocabSize, embeddingDim, hiddenDim, parentIntentVocabSize, childIntentVocabSize int) (*SimpleIntentClassifier, error) {
	embedding := NewEmbedding(vocabSize, embeddingDim)
	hiddenLayer, err := NewLinear(embeddingDim, hiddenDim)
	if err != nil {
		return nil, err
	}
	hiddenLayer2, err := NewLinear(hiddenDim, hiddenDim)
	if err != nil {
		return nil, err
	}
	outputLayer, err := NewLinear(hiddenDim, parentIntentVocabSize+childIntentVocabSize)
	if err != nil {
		return nil, err
	}

	return &SimpleIntentClassifier{
		Embedding:             embedding,
		HiddenLayer:           hiddenLayer,
		HiddenLayer2:          hiddenLayer2,
		OutputLayer:           outputLayer,
		ParentIntentVocabSize: parentIntentVocabSize,
		ChildIntentVocabSize:  childIntentVocabSize,
	}, nil
}

// SimpleIntentClassifier defines a simple model for intent classification.
type SimpleIntentClassifier struct {
	Embedding             *Embedding
	HiddenLayer           *Linear
	OutputLayer           *Linear
	HiddenLayer2          *Linear
	ParentIntentVocabSize int
	ChildIntentVocabSize  int
}

// Forward performs the forward pass of the model.
func (m *SimpleIntentClassifier) Forward(input *Tensor) (*Tensor, *Tensor, error) {
	embedded, err := m.Embedding.Forward(input)
	if err != nil {
		return nil, nil, err
	}

	hidden, err := m.HiddenLayer.Forward(embedded)
	if err != nil {
		return nil, nil, err
	}

	hidden2, err := m.HiddenLayer2.Forward(hidden)
	if err != nil {
		return nil, nil, err
	}

	output, err := m.OutputLayer.Forward(hidden2)
	if err != nil {
		return nil, nil, err
	}

	// Split the output into parent and child logits
	splitTensors, err := Split(output, 1, []int{m.ParentIntentVocabSize, m.ChildIntentVocabSize})
	if err != nil {
		return nil, nil, err
	}
	parentLogits := splitTensors[0]
	childLogits := splitTensors[1]

	return parentLogits, childLogits, nil
}

// Parameters returns all the learnable parameters of the model.
func (m *SimpleIntentClassifier) Parameters() []*Tensor {
	return append(append(append(m.Embedding.Parameters(), m.HiddenLayer.Parameters()...), m.HiddenLayer2.Parameters()...), m.OutputLayer.Parameters()...)
}

// Save saves the model to a file.
func (m *SimpleIntentClassifier) Save(filePath string) error {
	file, err := os.Create(filePath)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	err = encoder.Encode(m)
	if err != nil {
		return err
	}

	return nil
}

// LoadSimpleIntentClassifier loads a SimpleIntentClassifier from a file.
func LoadSimpleIntentClassifier(filePath string) (*SimpleIntentClassifier, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	var model SimpleIntentClassifier
	err = decoder.Decode(&model)
	if err != nil {
		return nil, err
	}

	return &model, nil
}

// Backward performs the backward pass of the model.
func (m *SimpleIntentClassifier) Backward(parentGrad, childGrad *Tensor) error {
	// Combine the gradients
	combinedGrad, err := Concat([]*Tensor{parentGrad, childGrad}, 1)
	if err != nil {
		return err
	}

	// Backpropagate through the layers
	err = m.OutputLayer.Backward(combinedGrad)
	if err != nil {
		return err
	}

	// The gradient for HiddenLayer2 is now in m.OutputLayer.Inputs()[0].Grad
	err = m.HiddenLayer2.Backward(m.OutputLayer.Inputs()[0].Grad)
	if err != nil {
		return err
	}

	// The gradient for HiddenLayer is now in m.HiddenLayer2.Inputs()[0].Grad
	err = m.HiddenLayer.Backward(m.HiddenLayer2.Inputs()[0].Grad)
	if err != nil {
		return err
	}

	// The gradient for Embedding is now in m.HiddenLayer.Inputs()[0].Grad
	err = m.Embedding.Backward(m.HiddenLayer.Inputs()[0].Grad)
	if err != nil {
		return err
	}

	return nil
}