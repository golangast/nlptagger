package main

import (
	"flag"
	"fmt"
	"log"

	"github.com/zendrulat/nlptagger/neural/nnu/seq2seq"
	"github.com/zendrulat/nlptagger/neural/nnu/vocab"
	"github.com/zendrulat/nlptagger/neural/tokenizer"
)

func main() {
	modelPath := flag.String("model_path", "gob_models/seq2seq_description_model.gob", "Path to the saved Seq2Seq model")
	inputVocabPath := flag.String("input_vocab_path", "gob_models/seq2seq_input_vocab.gob", "Path to the input vocabulary")
	outputVocabPath := flag.String("output_vocab_path", "gob_models/seq2seq_output_vocab.gob", "Path to the output vocabulary")
	query := flag.String("query", "create a new file", "Query to get a description for")
	flag.Parse()

	// Load vocabularies
	inputVocab, err := vocab.LoadVocabulary(*inputVocabPath)
	if err != nil {
		log.Fatalf("Failed to load input vocabulary: %v", err)
	}
	outputVocab, err := vocab.LoadVocabulary(*outputVocabPath)
	if err != nil {
		log.Fatalf("Failed to load output vocabulary: %v", err)
	}

	// Initialize tokenizer
	queryTokenizer, err := tokenizer.NewTokenizer(inputVocab)
	if err != nil {
		log.Fatalf("Failed to create query tokenizer: %v", err)
	}

	// Load the model
	model, err := seq2seq.Load(*modelPath, queryTokenizer)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	model.OutputVocab = outputVocab

	// Predict
	description, err := model.Predict(*query, 50)
	if err != nil {
		log.Fatalf("Failed to predict: %v", err)
	}

	fmt.Printf("Query: %s\n", *query)
	fmt.Printf("Predicted Description: %s\n", description)
}
