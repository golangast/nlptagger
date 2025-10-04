package intent

import (
	"encoding/gob"
	"os"

	. "nlptagger/neural/nn"
	. "nlptagger/neural/tensor"
)


// SimpleIntentClassifier defines a simple model for intent classification.
type SimpleIntentClassifier struct {
	Embedding    *Embedding
	HiddenLayer  *Linear
	OutputLayer  *Linear
	HiddenLayer2 *Linear
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
	outputGrad := m.OutputLayer.Backward(combinedGrad)
	hidden2Grad := m.HiddenLayer2.Backward(outputGrad)
	hiddenGrad := m.HiddenLayer.Backward(hidden2Grad)
	m.Embedding.Backward(hiddenGrad)

	return nil
}