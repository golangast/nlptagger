package gobs

import (
	"encoding/gob"
	"os"

	"github.com/golangast/nlptagger/neural/nnu"
)

func SaveModelToGOB(model *nnu.SimpleNN, filePath string) error {
	file, err := os.Create(filePath)

	if err != nil {
		return err
	}

	encoder := gob.NewEncoder(file)
	err = encoder.Encode(model)
	if err != nil {
		return err
	}
	file.Close()

	return nil
}

func LoadModelFromGOB(filePath string) (*nnu.SimpleNN, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}

	decoder := gob.NewDecoder(file)
	var model nnu.SimpleNN
	err = decoder.Decode(&model)
	if err != nil {
		return nil, err
	}
	file.Close()
	return &model, nil
}
