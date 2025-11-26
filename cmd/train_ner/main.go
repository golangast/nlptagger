package main

import (
	"encoding/gob"
	"encoding/json"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"strings"
	"time"

	"nlptagger/neural/nn/ner"
	"nlptagger/neural/nnu"
	"nlptagger/neural/nnu/word2vec"
	"nlptagger/tagger/tag"
)

// SemanticData represents the structure of semantic_output_data.json
type SemanticData struct {
	Query          string `json:"query"`
	SemanticOutput struct {
		TargetResource struct {
			Name     string `json:"name"`
			Children []struct {
				Name string `json:"name"`
			} `json:"children"`
		} `json:"target_resource"`
	} `json:"semantic_output"`
}

func main() {
	rand.Seed(time.Now().UnixNano())

	// 1. Load Data
	data, err := loadSemanticData("trainingdata/semantic_output_data.json")
	if err != nil {
		log.Fatalf("Failed to load data: %v", err)
	}

	// 2. Prepare Training Data (Synthetic NER tagging)
	var trainingData []tag.Tag
	vocab := make(map[string]int)
	vocab["<UNK>"] = 0
	vocabIndex := 1

	for _, item := range data {
		tokens := strings.Fields(item.Query)
		tags := make([]string, len(tokens))
		for i := range tags {
			tags[i] = "O"
		}

		// Simple matching for names (this is a heuristic for synthetic data)
		names := []string{item.SemanticOutput.TargetResource.Name}
		for _, child := range item.SemanticOutput.TargetResource.Children {
			names = append(names, child.Name)
		}

		for _, name := range names {
			if name == "" || name == "." {
				continue
			}
			nameTokens := strings.Fields(name)
			if len(nameTokens) == 0 {
				continue
			}

			// Find the name in the query tokens
			for i := 0; i <= len(tokens)-len(nameTokens); i++ {
				match := true
				for j := 0; j < len(nameTokens); j++ {
					if strings.Trim(tokens[i+j], "',.") != nameTokens[j] { // Simple trim for punctuation
						match = false
						break
					}
				}
				if match {
					tags[i] = "B-NAME"
					for j := 1; j < len(nameTokens); j++ {
						tags[i+j] = "I-NAME"
					}
				}
			}
		}

		// Build vocab and training examples
		for i, token := range tokens {
			if _, ok := vocab[token]; !ok {
				vocab[token] = vocabIndex
				vocabIndex++
			}
			trainingData = append(trainingData, tag.Tag{
				Token:  token,
				NerTag: []string{tags[i]},
			})
		}
	}

	log.Printf("Vocab size: %d", len(vocab))
	log.Printf("Training examples (tokens): %d", len(trainingData))

	// Count tags
	tagCounts := make(map[string]int)
	for _, t := range trainingData {
		for _, tag := range t.NerTag {
			tagCounts[tag]++
		}
	}
	log.Printf("Tag counts: %v", tagCounts)

	// 3. Initialize Model
	// Load Word2Vec model
	word2vecModel, err := word2vec.LoadModel("gob_models/word2vec_model.gob")
	if err != nil {
		log.Fatalf("Failed to load Word2Vec model: %v", err)
	}

	inputSize := word2vecModel.VectorSize
	hiddenSize := 50
	outputSize := 3 // O, B-NAME, I-NAME

	// We need to map tags to indices for the NN
	nerTagVocab := ner.CreateTagVocabNer(trainingData)
	outputSize = len(nerTagVocab)
	log.Printf("Output size (tags): %d", outputSize)

	nn := &nnu.SimpleNN{
		InputSize:    inputSize,
		HiddenSize:   hiddenSize,
		OutputSize:   outputSize,
		LearningRate: 0.01,  // Lower learning rate for embeddings
		TokenVocab:   vocab, // Keep vocab for reference, though we use w2v for input
		NerTagVocab:  nerTagVocab,
	}
	// Manually initialize weights
	nn.WeightsIH = nnu.NewMatrix(hiddenSize, inputSize)
	nn.WeightsHO = nnu.NewMatrix(outputSize, hiddenSize)
	nn.HiddenBiases = make([]float64, hiddenSize)
	nn.OutputBiases = make([]float64, outputSize)

	// 4. Train Loop
	epochs := 5 // Increased epochs
	for epoch := 0; epoch < epochs; epoch++ {
		totalError := 0.0
		for _, example := range trainingData {
			// Word2Vec input
			var input []float64
			id, ok := word2vecModel.Vocabulary[example.Token]
			if ok {
				input = word2vecModel.WordVectors[id]
			}

			if input == nil {
				// Handle unknown words (zero vector)
				input = make([]float64, inputSize)
			}

			// Forward
			outputs := ner.ForwardPassNer(nn, input)
			nn.Outputs = outputs // Store for backprop
			nn.Inputs = input    // Store for backprop

			// Calculate Error
			targetTag := example.NerTag[0]

			// Set target output
			targetOutput := make([]float64, outputSize)
			targetIndex := nerTagVocab[targetTag]
			targetOutput[targetIndex] = 1.0

			nn.Targets = targetOutput

			// Calculate total error for logging (MSE)
			for i := 0; i < outputSize; i++ {
				diff := targetOutput[i] - outputs[i]
				totalError += diff * diff
			}

			// Backpropagate with higher learning rate
			nn.Backpropagate(0, 0.1)
		}
		if epoch%20 == 0 {
			log.Printf("Epoch %d, Total Error: %f", epoch, totalError)
		}
	}

	// 5. Save Model
	savePath := "gob_models/ner_model.gob"
	file, err := os.Create(savePath)
	if err != nil {
		log.Fatalf("Failed to create file: %v", err)
	}
	defer file.Close()
	encoder := gob.NewEncoder(file)
	err = encoder.Encode(nn)
	if err != nil {
		log.Fatalf("Failed to encode model: %v", err)
	}
	log.Printf("NER model saved to %s", savePath)
	log.Printf("Final Tag counts: %v", tagCounts)
}

func loadSemanticData(path string) ([]SemanticData, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	bytes, err := ioutil.ReadAll(file)
	if err != nil {
		return nil, err
	}
	var data []SemanticData
	err = json.Unmarshal(bytes, &data)
	return data, err
}
