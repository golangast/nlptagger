package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/golangast/nlptagger/neural/nnu/bartsimple" // Assuming this is the correct import path
	"github.com/golangast/nlptagger/neural/nnu/vocab"
)
var (
	trainMode    = flag.Bool("train", false, "Enable training mode")
	epochs       = flag.Int("epochs", 10, "Number of training epochs")
	learningRate = flag.Float64("lr", 0.001, "Learning rate for training")
	bartDataPath = flag.String("data", "trainingdata/bart_training_data.json", "Path to BART training data for the model")
	dimModel     = flag.Int("dim", 128, "Dimension of the model")
	numHeads     = flag.Int("heads", 8, "Number of attention heads")
	maxSeqLength = flag.Int("maxlen", 512, "Maximum sequence length")
)

func main() {
	flag.Parse()

	modelPath := "gob_models/simplified_bart_model.gob"
	jsonpath := "trainingdata/tagdata/nlp_training_data.json"
	vocabPath := "gob_models/vocabulary.gob" // Path to your vocabulary file
	var vocabulary *bartsimple.Vocabulary
	var err error
	trainingData, err := vocab.LoadTrainingDataJSON(jsonpath)
	if err != nil {
		fmt.Println("error loading training data: %w", err)
	}
	// Create vocabularies
	tokenVocab := vocab.CreateTokenVocab(trainingData.Sentences)
	// Use CreateVocab to either load the existing vocabulary or build a new one
	vocabulary = bartsimple.NewVocabulary()
	for word := range tokenVocab {
		vocabulary.AddToken(word, len(vocabulary.TokenToWord)) // This might reassign IDs if not careful
	}
	vocabulary.UnknownTokenID = tokenVocab["UNK"]
	// Example of using the vocabulary (after loading/creating)
	// Save the vocabulary (optional)
	if err := vocabulary.Save(vocabPath); err != nil {
		fmt.Printf("Error saving vocabulary: %v\n", err)
	}

	// Load or build the vocabulary
	vocabulary, err = bartsimple.LoadVocabulary(vocabPath)
	if err != nil {
		fmt.Printf("Attempting to load vocabulary from %s\n", vocabPath)
		fmt.Printf("Error loading vocabulary: %v. Building a new one from training data.\n", err)
		// If loading fails, build a new vocabulary from training data
		vocabulary = bartsimple.NewVocabulary()
		// Re-load training data if needed, or use the 'trainingData' loaded earlier
		trainingData, loadErr := vocab.LoadTrainingDataJSON(jsonpath)
		if loadErr != nil {
			log.Fatalf("Error loading training data to build new vocabulary: %v", loadErr)
		}
		uniqueWords := make(map[string]bool)
		for _, sentence := range trainingData.Sentences {
			words := strings.Fields(sentence.Sentence)
			for _, word := range words {
				cleanedWord := strings.ToLower(strings.Trim(word, ".,!?;:\"'()[]{}-_")) // Basic cleaning
				if cleanedWord != "" {
					uniqueWords[cleanedWord] = true
				}
			}
		}
		tokenVocab := vocab.CreateTokenVocab(trainingData.Sentences)
		for word := range tokenVocab {
			vocabulary.AddToken(word, len(vocabulary.TokenToWord))
		}
		// Make sure special tokens are added if they are not in your training data
		vocabulary.AddToken("[UNK]", len(vocabulary.TokenToWord))
		vocabulary.AddToken("[BOS]", len(vocabulary.TokenToWord))
		vocabulary.AddToken("[EOS]", len(vocabulary.TokenToWord))
		vocabulary.AddToken("[PAD]", len(vocabulary.TokenToWord))
		// Add unique words from training data in a consistent order (e.g., alphabetical)
		//sortedWords := []string{}
		// for word := range uniqueWords {
		// 	sortedWords = append(sortedWords, word)
		// }
		vocabulary.UnknownTokenID = vocabulary.WordToToken["[UNK]"]
		vocabulary.BeginningOfSentenceID = vocabulary.WordToToken["[BOS]"]
		vocabulary.EndOfSentenceID = vocabulary.WordToToken["[EOS]"]
		vocabulary.PaddingTokenID = vocabulary.WordToToken["[PAD]"]

		// Save the new vocabulary
		if err := vocabulary.Save(vocabPath); err != nil {
			fmt.Printf("Error saving newly built vocabulary: %v\n", err)
		}
	}
	if vocabulary.UnknownTokenID == -1 {
		if unkID, ok := vocabulary.WordToToken["[UNK]"]; ok {
			vocabulary.UnknownTokenID = unkID
		} else {
			// This indicates an issue with the saved vocabulary file
			fmt.Println("Loaded vocabulary does not contain '[UNK]' token and UnknownTokenID is -1.")
		}
	}
	if vocabulary.BeginningOfSentenceID == -1 {
		if bosID, ok := vocabulary.WordToToken["[BOS]"]; ok {
			vocabulary.BeginningOfSentenceID = bosID
		} else {
			fmt.Println("Warning: '[BOS]' token not found in vocabulary after loading/creation.")
		}
	}

	var model *bartsimple.SimplifiedBARTModel
	var modelLoadErr error
	fmt.Printf("Vocabulary size before attempting to load model: %d\n", len(vocabulary.WordToToken))
	model, modelLoadErr = bartsimple.LoadSimplifiedBARTModelFromGOB(modelPath) // Assuming LoadSimplifiedBARTModelFromGOB exists and handles loading
	if modelLoadErr != nil {
		fmt.Printf("Attempting to load simplified BART model from %s\n", modelPath)
		fmt.Printf("Error loading simplified BART model: %v. Creating a new one.\n", modelLoadErr)
		log.Println("Model loading failed, skipping saving of a new model for now to avoid panic.")
		var createErr error
		tokenizer, err := bartsimple.NewTokenizer(vocabulary, 0, 0, 0, 0)
		if err != nil {
			fmt.Println("Error creating tokenizer: ", err)
			return // Handle the error appropriately, perhaps exit
		}
		fmt.Printf("Vocabulary size being passed to NewSimplifiedBARTModel: %d\n", len(vocabulary.WordToToken))
		model, createErr = bartsimple.NewSimplifiedBARTModel(tokenizer, vocabulary, *dimModel, *numHeads, *maxSeqLength)
		if createErr != nil {
			log.Fatalf("Failed to create a new simplified BART model: %v", createErr)
		}

		// If loading fails, create a new model, passing the tokenizer and vocabulary
		

	} else {
		// Check if the model object is nil EVEN IF modelLoadErr is nil
		if model == nil {
			log.Fatalf("Model loading succeeded according to error, but the model object is nil!")
		}
		model.Vocabulary = vocabulary

	}
	// If loaded successfully, you might want to update model fields that are not serialized
	// Assign the correctly loaded/created vocabulary to the model
	model.Vocabulary = vocabulary

	// Create a tokenizer using the loaded/built vocabulary
	bosID := vocabulary.BeginningOfSentenceID
	padID := vocabulary.PaddingTokenID
	eosID := vocabulary.EndOfSentenceID
	// Use the potentially updated UnknownTokenID
	// Set unkID to the size of the vocabulary to ensure it's out of the valid range [0, Size-1)
	unkID := vocabulary.Size

	tokenizer, err := bartsimple.NewTokenizer(vocabulary, bosID, eosID, padID, unkID)
	if err != nil {
		fmt.Println("Error creating tokenizer: ", err)
		return // Handle the error appropriately, perhaps exit
	}
	if model != nil {
		model.Vocabulary = vocabulary // This vocabulary has UnknownTokenID = 103
		model.Tokenizer = tokenizer   // This tokenizer was created with the vocabulary where UnknownTokenID = 103
	} else {
		fmt.Println("Model is nil after loading or creation.")
	}
	// Assign the newly created tokenizer to the model
	model.Tokenizer = tokenizer

	if *trainMode {
		fmt.Println("--- Running in Training Mode ---")

		// 1. Load BART-specific training data
		bartTrainingData, err := bartsimple.LoadBARTTrainingData(*bartDataPath)
		if err != nil {
			log.Fatalf("Error loading BART training data from %s: %v", *bartDataPath, err)
		}
		fmt.Printf("Loaded %d training sentences for BART model.\n", len(bartTrainingData.Sentences))

		// 2. Train the model
		err = bartsimple.TrainBARTModel(model, bartTrainingData, *epochs, *learningRate)
		if err != nil {
			log.Fatalf("BART model training failed: %v", err)
		}

		// 3. Save the trained model
		fmt.Printf("Training complete. Saving trained model to %s...\n", modelPath)
		if err := bartsimple.SaveSimplifiedBARTModelToGOB(model, modelPath); err != nil {
			log.Fatalf("Error saving trained BART model: %v", err)
		}
		fmt.Println("Model saved successfully.")
	} else {
		fmt.Println("--- Running in Inference Mode ---")
		//go run . -train -epochs 50 -lr 0.001 -data trainingdata/bart_training_data.json
		//go run . -train -dim 256 -heads 8 -maxlen 256
		answer := InputScanDirections("\n what do you want?")
		// Process the command using BartProcessCommand
		summary, err := model.BartProcessCommand(answer) // If BartProcessCommand needs the vocabulary, pass it here
		if err != nil {
			log.Fatalf("Error processing command with BART model: %v", err)
		}
		fmt.Printf("Generated Summary: %s\n", summary)
	}
}

func InputScanDirections(directions string) string {
	fmt.Println(directions)

	scannerdesc := bufio.NewScanner(os.Stdin)
	tr := scannerdesc.Scan()
	if tr {
		dir := scannerdesc.Text()
		stripdir := strings.TrimSpace(dir)
		return stripdir
	} else {
		return ""
	}

}
