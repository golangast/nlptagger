package bert

import (
	"encoding/json"
	"fmt"
	"os"
	"time"

	"log"

	"github.com/golangast/nlptagger/neural/nnu/bartsimple"
	"github.com/golangast/nlptagger/neural/nnu/word2vec"
	"github.com/golangast/nlptagger/tagger"
	"github.com/golangast/nlptagger/tagger/nertagger"
	"github.com/golangast/nlptagger/tagger/postagger"
)

// TrainingExample represents the structure of the bert.json file.
// We assume it contains pairs of sentences and their corresponding command labels.

// LoadTrainingData loads the BERT training data from a JSON file.
func LoadTrainingData(path string) ([]TrainingExample, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("could not open training data file: %w", err)
	}
	defer file.Close()

	var data []TrainingExample
	if err := json.NewDecoder(file).Decode(&data); err != nil {
		return nil, fmt.Errorf("could not decode training data: %w", err)
	}
	return data, nil
}

// Train function to orchestrate the training of the BERT model.
func Train(config BertConfig, data []TrainingExample, epochs int, learningRate float64, vocabulary *bartsimple.Vocabulary, word2vecEmbeddings map[string][]float64) (*BertModel, error) {
	// 1. Initialize the Model
	model := NewBertModel(config, data, word2vecEmbeddings)

	// 2. Initialize the Optimizer
	optimizer := NewAdam(model.Parameters(), learningRate)

	fmt.Print("Starting BERT model training [")
		start := time.Now()
	// 3. Training Loop
	tokenizer, err := bartsimple.NewTokenizer(vocabulary, vocabulary.BeginningOfSentenceID, vocabulary.EndOfSentenceID, vocabulary.PaddingTokenID, vocabulary.UnknownTokenID)
	if err != nil {
		return nil, err
	}


	for epoch := 0; epoch < epochs; epoch++ {
			fmt.Print("++")

		totalLoss := 0.0
		for _, item := range data {
			optimizer.ZeroGrad()

			tokenIDs, err := tokenizer.Encode(item.Text)
			if err != nil {
				return nil, err
			}
			inputTensor := NewTensor(nil, []int{1, len(tokenIDs)}, false)
			for i, id := range tokenIDs {
				inputTensor.Data[i] = float64(id)
			}
			tokenTypeIDs := NewTensor(make([]float64, len(tokenIDs)), []int{1, len(tokenIDs)}, false)

			tags := tagger.Tagging(item.Text)
			posTagIDs := make([]float64, len(tags.PosTag))
			posTagToIDMap := postagger.PosTagToIDMap()
			for i, tag := range tags.PosTag {
				posTagIDs[i] = float64(posTagToIDMap[tag])
			}
			posTagTensor := NewTensor(posTagIDs, []int{1, len(posTagIDs)}, false)

			nerTagIDs := make([]float64, len(tags.NerTag))
			nerTagToIDMap := nertagger.NerTagToIDMap()
			for i, tag := range tags.NerTag {
				nerTagIDs[i] = float64(nerTagToIDMap[tag])
			}
			nerTagTensor := NewTensor(nerTagIDs, []int{1, len(nerTagIDs)}, false)

			logits,err := model.Forward(inputTensor, tokenTypeIDs, posTagTensor, nerTagTensor)
            if err != nil {
                return nil, err
            }
            // Debugging: Print logits immediately after forward pass

			// Create a label tensor
			labelTensor := NewTensor([]float64{float64(item.Label)}, []int{1}, false)

			// Calculate loss
			loss := CrossEntropyLoss(logits, labelTensor)
			totalLoss += loss.Data[0]

			// Backward pass and optimization
			loss.Backward()
			optimizer.Step()
			// Debugging: Print loss after each item

		}

		// avgLoss := totalLoss / float64(len(data))
		// fmt.Printf("Epoch %d, Loss: %.6f\n", epoch+1, avgLoss)
	}
	fmt.Print("]: ")

	elapsed := time.Since(start)
	fmt.Print(elapsed)

	return model, nil
}

// SetupModel loads a model from modelPath or creates a new one if loading fails.
func SetupModel(modelPath string, vocabulary *bartsimple.Vocabulary, dim, heads, maxLen int, pretrainedEmbeddings map[string][]float64) (*bartsimple.SimplifiedBARTModel, error) {
	// If loading fails, create a new one

	// Load word2vec model only when creating a new BART model
	word2vecModel, err := word2vec.LoadModel("gob_models/word2vec_model.gob")
	if err != nil {
		log.Printf("Warning: Could not load word2vec model: %v. BART embeddings will be initialized randomly.", err)
		pretrainedEmbeddings = nil // Or an empty map
	} else {
		log.Println("Word2vec model loaded successfully.")
		pretrainedEmbeddings = word2vec.ConvertToMap(word2vecModel.WordVectors, word2vecModel.Vocabulary)
	}

	tokenizer, err := bartsimple.NewTokenizer(
		vocabulary,
		vocabulary.BeginningOfSentenceID,
		vocabulary.EndOfSentenceID,
		vocabulary.PaddingTokenID,
		vocabulary.UnknownTokenID,
	)
	if err != nil {
		log.Fatalf("Failed to create tokenizer: %v", err)
	}

	fmt.Printf("Creating new simplified BART model with vocab size: %d\n", len(vocabulary.WordToToken))
	newModel, createErr := bartsimple.NewSimplifiedBARTModel(tokenizer, vocabulary, dim, heads, maxLen, pretrainedEmbeddings)
	if createErr != nil {
		return nil, fmt.Errorf("failed to create a new simplified BART model: %w", createErr)
	}

	// Save the newly created model so it can be used next time.
	fmt.Printf("Saving newly created model to %s...\n", modelPath)
	if err := bartsimple.SaveSimplifiedBARTModelToGOB(newModel, modelPath); err != nil {
		// Log as a warning because the model is still usable in memory for this run
		fmt.Printf("Warning: Error saving newly created BART model: %v\n", err)
	} else {
		fmt.Println("New model saved successfully.")
	}

	return newModel, nil
}

func RunTraining(model *bartsimple.SimplifiedBARTModel, bartDataPath, modelPath string, epochs int, learningRate float64, batchSize int) {
	fmt.Println("--- Running in Training Mode ---")

	// Train Word2Vec model first
	word2vecTrainingDataPath := "trainingdata/tagdata/nlp_training_data.json"
	word2vecModelSavePath := "gob_models/word2vec_model.gob"
	word2vecVectorSize := 100
	word2vecEpochs := 50
	word2vecWindow := 5
	word2vecNegativeSamples := 5
	word2vecMinWordFrequency := 1
	word2vecUseCBOW := true

	_, err := word2vec.TrainWord2VecModel(
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

	// 1. Load BART-specific training data
	bartTrainingData, err := bartsimple.LoadBARTTrainingData(bartDataPath)
	if err != nil {
		log.Fatalf("Error loading BART training data from %s: %v", bartDataPath, err)
	}
	fmt.Printf("Loaded %d training sentences for BART model.\n", len(bartTrainingData.Sentences))
	// 2. Train the model
	err = bartsimple.TrainBARTModel(model, bartTrainingData, epochs, learningRate, batchSize)
	if err != nil {
		log.Fatalf("BART model training failed: %v", err)
	}

	// 3. Save the trained model
	fmt.Printf("Training complete. Saving trained model to %s...\n", modelPath)
	if err := bartsimple.SaveSimplifiedBARTModelToGOB(model, modelPath); err != nil {
		log.Fatalf("Error saving trained BART model: %v", err)
	}
	fmt.Println("Model saved successfully.")
}