package main

import (
	"flag"
	"log"

	"github.com/zendrulat/nlptagger/neural/nnu/word2vec"
)

var (
	dimModel = flag.Int("dim", 64, "Dimension of the model")
)

func main() {
	flag.Parse()

	// Train Word2Vec model first
	word2vecTrainingDataPath := "./trainingdata/WikiQA-train.txt"
	word2vecModelSavePath := "gob_models/word2vec_model.gob"
	word2vecVectorSize := *dimModel
	word2vecEpochs := 1
	word2vecWindow := 5
	word2vecNegativeSamples := 5
	word2vecMinWordFrequency := 1
	word2vecUseCBOW := true

	_, err := word2vec.LoadModel(word2vecModelSavePath)
	if err != nil {
		log.Printf("Word2Vec model not found, training a new one...")
		_, err = word2vec.TrainWord2VecModel(
			word2vecTrainingDataPath,
			word2vecModelSavePath,
			word2vecVectorSize,
			word2vecEpochs,
			word2vecWindow,
			word2vecNegativeSamples,
			word2vecMinWordFrequency,
			word2vecUseCBOW,
		)
		if err != nil {
			log.Fatalf("Error training Word2Vec model: %v", err)
		}
	} else {
		log.Println("Loaded existing Word2Vec model.")
	}
}
