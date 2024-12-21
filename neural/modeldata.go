package modeldata

import (
	"github.com/golangast/nlptagger/neural/nnu"
	"github.com/golangast/nlptagger/neural/nnu/train"
	"github.com/golangast/nlptagger/neural/nnu/vocab"
)

// NN loads or trains a neural network model for POS tagging.

func ModelData(modeldirectory string) (*nnu.SimpleNN, error) {

	_, _, _, _, _, trainingData := vocab.CreateVocab()

	return train.TrainAndSaveModel(trainingData, modeldirectory)
}
