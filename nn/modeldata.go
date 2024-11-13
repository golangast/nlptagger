package modeldata

import (
	"github.com/golangast/nlptagger/nn/simplenn"
)

// NN loads or trains a neural network model for POS tagging.

func ModelData() (*simplenn.SimpleNN, error) {

	nn := simplenn.SimpleNN{}
	_, _, _, _, _, trainingData := simplenn.CreateVocab()
	// Train and save the model

	return nn.TrainAndSaveModel(trainingData)
}
