package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/golangast/nlptagger/neural/nnu/bartsimple" // Assuming this is the correct import path
	"github.com/golangast/nlptagger/neural/nnu/vocab"
)

func main() {
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
		// You might need to handle special tokens differently here
		// if their IDs are fixed or need specific handling.
		// For simplicity, assuming AddToken handles it:

		vocabulary.AddToken(word) // This might reassign IDs if not careful
	}
	vocabulary.UnknownTokenID = tokenVocab["UNK"]
	// Example of using the vocabulary (after loading/creating)
	fmt.Printf("Vocabulary size: %d\n", vocabulary.Size)

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
		tokenVocab := vocab.CreateTokenVocab(trainingData.Sentences)
		for word := range tokenVocab {
			vocabulary.AddToken(word)
		}
		// Make sure special tokens are added if they are not in your training data
		vocabulary.AddToken("[UNK]")
		vocabulary.AddToken("[BOS]")
		vocabulary.AddToken("[EOS]")
		vocabulary.AddToken("[PAD]")

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
			log.Fatalf("Loaded vocabulary does not contain '[UNK]' token and UnknownTokenID is -1.")
		}
	}
	if vocabulary.BeginningOfSentenceID == -1 {
		if bosID, ok := vocabulary.WordToToken["[BOS]"]; ok {
			vocabulary.BeginningOfSentenceID = bosID
		} else {
			log.Println("Warning: '[BOS]' token not found in vocabulary after loading/creation.")
		}
	}

	var model *bartsimple.SimplifiedBARTModel
	var modelLoadErr error
	// Attempt to load the simplified BART model
	// Pass the loaded/created tokenizer and vocabulary to NewSimplifiedBARTModel
	model, modelLoadErr = bartsimple.LoadSimplifiedBARTModelFromGOB(modelPath) // Assuming LoadSimplifiedBARTModelFromGOB exists and handles loading
	if modelLoadErr != nil {
		fmt.Printf("Attempting to load simplified BART model from %s\n", modelPath)
		fmt.Printf("Error loading simplified BART model: %v. Creating a new one.\n", modelLoadErr)
		var dummyDimModel = 128     // Replace with actual value
		var dummyNumHeads = 8       // Replace with actual value
		var dummyMaxSeqLength = 512 // Replace with actual value
		log.Println("Model loading failed, skipping saving of a new model for now to avoid panic.")
		var createErr error
		tokenizer, err := bartsimple.NewTokenizer(vocabulary, 0, 0, 0, 0)
		if err != nil {
			fmt.Println("Error creating tokenizer: ", err)
			return // Handle the error appropriately, perhaps exit
		}
		model, createErr = bartsimple.NewSimplifiedBARTModel(tokenizer, vocabulary, dummyDimModel, dummyNumHeads, dummyMaxSeqLength)
		if createErr != nil {
			log.Fatalf("Failed to create a new simplified BART model: %v", createErr)
		}

		// If loading fails, create a new model, passing the tokenizer and vocabulary
		if err := bartsimple.SaveSimplifiedBARTModelToGOB(model, modelPath); err != nil {
			// Assuming model was potentially created here after failure
			log.Printf("Error saving new simplified BART model: %v", err)
		} else {
			model.Vocabulary = vocabulary

			// Assuming model was potentially created here after failure
			log.Printf("New simplified BART model saved to %s", modelPath)
		}

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
		log.Fatalf("Model is nil after loading or creation.")
	}
	// Assign the newly created tokenizer to the model
	model.Tokenizer = tokenizer

	// Crucial check: Ensure model is not nil before proceeding

	//answer := "generate a webserver"
	answer := InputScanDirections("what do you want?")
	// Process the command using BartProcessCommand
	start := time.Now()
	summary, err := model.BartProcessCommand(answer) // If BartProcessCommand needs the vocabulary, pass it here
	if err != nil {
		log.Fatalf("Error processing command with BART model: %v", err)
	}
	duration := time.Since(start)
	fmt.Printf("BartProcessCommand completed in %s\n", duration)

	fmt.Printf("Generated Summary: %s\n", summary)

	// The final log message might still be generic if BartProcessCommand doesn't change it
	// log.Println("Simplified BART model forward pass completed. Output logits generated.")
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
