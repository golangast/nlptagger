package main

import (
	"log"

	"nlptagger/neural/nnu/word2vec"
)

func trainword2vec() {
	trainingDataPath := "trainingdata/bartdata/para.txt"
	modelSavePath := "gob_models/word2vec_model.gob"
	vectorSize := 100
	epochs := 100
	window := 2
	negativeSamples := 5
	minWordFrequency := 0
	useCBOW := false

	log.Println("Starting Word2Vec model training...")
	_, err := word2vec.TrainWord2VecModel(trainingDataPath, modelSavePath, vectorSize, epochs, window, negativeSamples, minWordFrequency, useCBOW)
	if err != nil {
		log.Fatalf("Word2Vec model training failed: %v", err)
	}
	log.Println("Word2Vec model training completed and saved.")
}
