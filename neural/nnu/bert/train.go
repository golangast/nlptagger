package bert

import (
	"encoding/json"
	"fmt"
	"os"
	"time"

	"log"

	"github.com/golangast/nlptagger/neural/nnu/bartsimple"
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
func SetupModel(modelPath string, vocabulary *bartsimple.Vocabulary, dim, heads, maxLen int) (*bartsimple.SimplifiedBARTModel, error) {
	// Attempt to load the model
	model, err := bartsimple.LoadSimplifiedBARTModelFromGOB(modelPath)
	if err == nil && model != nil {
		// Check if the loaded model's vocabulary size matches the current vocabulary.
		// This is a critical check to prevent panics from using an old model with a new, larger vocabulary.
		if model.VocabSize == len(vocabulary.WordToToken) {
			// Ensure the loaded model uses the up-to-date vocabulary and tokenizer
			model.Vocabulary = vocabulary
			if model.TokenEmbedding != nil {
				model.TokenEmbedding.VocabSize = model.VocabSize
			}
			tokenizer, tknErr := bartsimple.NewTokenizer(vocabulary, vocabulary.BeginningOfSentenceID, vocabulary.EndOfSentenceID, vocabulary.PaddingTokenID, vocabulary.UnknownTokenID)
			if tknErr != nil {
				return nil, fmt.Errorf("failed to create tokenizer for loaded model: %w", tknErr)
			}
			model.Tokenizer = tokenizer
			return model, nil
		}
		// If vocabulary sizes do not match, the model is incompatible.
		fmt.Printf("Loaded model has a vocabulary size of %d, but the current vocabulary has size %d. Rebuilding model.\n", model.VocabSize, len(vocabulary.WordToToken))
		// Fall through to create a new model.
	}

	// If loading fails, create a new one
	if err != nil {
		fmt.Printf("Error loading simplified BART model: %v. Creating a new one.\n", err)
	} else if model == nil {
		fmt.Println("Model file loaded without error, but model is nil. Creating a new one.")
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
	newModel, createErr := bartsimple.NewSimplifiedBARTModel(tokenizer, vocabulary, dim, heads, maxLen)
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