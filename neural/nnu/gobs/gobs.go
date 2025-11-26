// Package gobs handles saving and loading neural network models using the gob encoding.
// gob is used for serialization of Go data structures.

package gobs

import (
	"encoding/gob"
	"fmt"
	"os"

	"github.com/zendrulat/nlptagger/neural/nnu"
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
	nn := new(nnu.SimpleNN)

	err = decoder.Decode(&nn)
	if err != nil {
		return nil, err
	}
	file.Close()
	return nn, nil
}

func DeleteGobFile(filePath string) error {
	err := os.Remove(filePath)
	if err != nil {
		// Handle the error (e.g., log it, return it)
		return fmt.Errorf("failed to delete gob file %s: %w", filePath, err)
	}
	return nil // Indicate success
}
