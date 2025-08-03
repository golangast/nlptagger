package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"strings"

	"golang.org/x/net/html"

	"github.com/golangast/nlptagger/neural/nnu/bartsimple"
	"github.com/golangast/nlptagger/neural/nnu/bert"
	"github.com/golangast/nlptagger/neural/nnu/vocab"
)

var (
	trainBart    = flag.Bool("train-bart", false, "Enable BART model training")
	trainBert    = flag.Bool("train-bert", false, "Enable BERT model training")
	epochs       = flag.Int("epochs", 10, "Number of training epochs")
	learningRate = flag.Float64("lr", 0.001, "Learning rate for training")
	bartDataPath = flag.String("bart-data", "trainingdata/bartdata/bartdata.json", "Path to BART training data")
	bertDataPath = flag.String("bert-data", "trainingdata/bertdata/bert.json", "Path to BERT training data")
	dimModel     = flag.Int("dim", 64, "Dimension of the model")
	numHeads     = flag.Int("heads", 4, "Number of attention heads")
	maxSeqLength = flag.Int("maxlen", 64, "Maximum sequence length")
	batchSize    = flag.Int("batchsize", 4, "Batch size for training")
)

func main() {
	flag.Parse()

	// Define paths
	const bartModelPath = "gob_models/simplified_bart_model.gob"
	const bertModelPath = "gob_models/bert_classifier_model.gob"
	const trainingDataPath = "trainingdata/tagdata/nlp_training_data.json"
	const vocabPath = "gob_models/vocabulary.gob"

	vocabulary, err := setupVocabulary(vocabPath, trainingDataPath)
	if err != nil {
		log.Fatalf("Failed to set up vocabulary: %v", err)
	}

	bartModel, err := setupModel(bartModelPath, vocabulary, *dimModel, *numHeads, *maxSeqLength)
	if err != nil {
		log.Fatalf("Failed to set up BART model: %v", err)
	}

	if *trainBart {
		runTraining(bartModel, *bartDataPath, bartModelPath)
		return // Exit after training
	}

	// BERT model setup and training
	bertConfig := bert.BertConfig{
		VocabSize:             len(vocabulary.WordToToken),
		HiddenSize:            *dimModel,
		NumHiddenLayers:       2, // Example value
		NumAttentionHeads:     *numHeads,
		IntermediateSize:      *dimModel * 4, // Example value
		MaxPositionEmbeddings: *maxSeqLength,
		TypeVocabSize:         2, // Example value
		LayerNormEps:          1e-12,
		HiddenDropoutProb:     0.1,
	}

	var bertModel *bert.BertModel
	if *trainBert {
		bertTrainingData, err := bert.LoadTrainingData(*bertDataPath)
		if err != nil {
			log.Fatalf("Error loading BERT training data: %v", err)
		}
		bertModel, err = bert.Train(bertConfig, bertTrainingData, *epochs, *learningRate)
		if err != nil {
			log.Fatalf("BERT model training failed: %v", err)
		}
		// Add saving logic for the trained BERT model if needed
	} else {
		// Add loading logic for a pre-trained BERT model here
		// For now, we'll just initialize a new model.
		bertModel = bert.NewBertModel(bertConfig)
		fmt.Println("Initialized new BERT model for inference.")
	}

	runInference(bartModel, bertModel, bertConfig)
}

// setupVocabulary loads a vocabulary from vocabPath or builds a new one if loading fails.
func setupVocabulary(vocabPath, trainingDataPath string) (*bartsimple.Vocabulary, error) {
	// Attempt to load the vocabulary first
	vocabulary, err := bartsimple.LoadVocabulary(vocabPath)
	if err == nil && vocabulary != nil && vocabulary.WordToToken != nil {
		fmt.Printf("Successfully loaded vocabulary from %s\n", vocabPath)
		// Validate that the loaded vocabulary contains the essential special tokens.
		// If not, we'll treat it as an invalid vocabulary and rebuild it.
		_, unkExists := vocabulary.WordToToken["[UNK]"]
		_, padExists := vocabulary.WordToToken["[PAD]"]
		_, bosExists := vocabulary.WordToToken["[BOS]"]
		_, eosExists := vocabulary.WordToToken["[EOS]"]

		if unkExists && padExists && bosExists && eosExists {
			// All essential tokens exist, set the IDs and return.
			vocabulary.UnknownTokenID = vocabulary.WordToToken["[UNK]"]
			vocabulary.PaddingTokenID = vocabulary.WordToToken["[PAD]"]
			vocabulary.BeginningOfSentenceID = vocabulary.WordToToken["[BOS]"]
			vocabulary.EndOfSentenceID = vocabulary.WordToToken["[EOS]"]
			return vocabulary, nil
		}
		fmt.Println("Loaded vocabulary is missing one or more special tokens. Rebuilding.")
	} else if err != nil {
		// If there was an error loading (e.g., file not found), print it and proceed to build.
		fmt.Printf("Error loading vocabulary: %v. Building a new one from training data.\n", err)
	}

	// If loading fails, build a new one

	trainingData, loadErr := vocab.LoadTrainingDataJSON(trainingDataPath)
	if loadErr != nil {
		return nil, fmt.Errorf("error loading training data to build new vocabulary: %w", loadErr)
	}

	fmt.Println("Building new vocabulary...")
	allWords := make(map[string]bool)

	// Gather words from all sources.
	addWordsFromJSONTrainingData(trainingData, allWords)
	addWordsFromRAGDocs("trainingdata/ragdata/ragdocs.txt", allWords)
	addWordsFromRAGJSON("trainingdata/ragdata/rag_data.json", allWords)
	addWordsFromHTML("docs/index.html", allWords)

	// 5. Expand vocabulary with words from WikiQA data
	addWordsFromWikiQA("trainingdata/WikiQA-train.txt", allWords)


	newVocab := bartsimple.NewVocabulary()

	// Add special tokens first to ensure they have consistent IDs
	newVocab.AddToken("[PAD]", 0)
	newVocab.AddToken("[UNK]", 1)
	newVocab.AddToken("[BOS]", 2)
	newVocab.AddToken("[EOS]", 3)

	for word := range allWords {
		// Avoid re-adding special tokens
		if _, exists := newVocab.WordToToken[word]; !exists {
			newVocab.AddToken(word, len(newVocab.TokenToWord))
		}
	}

	newVocab.PaddingTokenID = newVocab.WordToToken["[PAD]"]
	newVocab.UnknownTokenID = newVocab.WordToToken["[UNK]"]
	newVocab.BeginningOfSentenceID = newVocab.WordToToken["[BOS]"]
	newVocab.EndOfSentenceID = newVocab.WordToToken["[EOS]"]

	// Save the new vocabulary
	if err := newVocab.Save(vocabPath); err != nil {
		// Log the error but continue, as we have a functional vocabulary in memory
		fmt.Printf("Warning: Error saving newly built vocabulary: %v\n", err)
	} else {
		fmt.Printf("Saved new vocabulary to %s\n", vocabPath)
	}

	return newVocab, nil
}

// addWordsFromJSONTrainingData extracts words from the primary annotated training data.
func addWordsFromJSONTrainingData(trainingData *vocab.TrainingDataJSON, wordSet map[string]bool) {
	fmt.Println("Expanding vocabulary with words from JSON training data...")
	tokenVocab := vocab.CreateTokenVocab(trainingData.Sentences)
	for word := range tokenVocab {
		wordSet[word] = true
	}
}

// addWordsFromRAGDocs extracts words from the plain text RAG documents
func addWordsFromRAGDocs(path string, wordSet map[string]bool) {
	fmt.Printf("Expanding vocabulary with words from %s\n", path)
	file, err := os.Open(path)
	if err != nil {
		fmt.Printf("Warning: could not open RAG docs to expand vocabulary: %v\n", err)
		return
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		words := strings.Fields(scanner.Text())
		for _, word := range words {
			cleanedWord := strings.ToLower(strings.Trim(word, ".,!?;:\"'()[]{}-_"))
			if cleanedWord != "" {
				wordSet[cleanedWord] = true
			}
		}
	}
}

// addWordsFromWikiQA extracts words from the WikiQA training data.
func addWordsFromWikiQA(path string, wordSet map[string]bool) {
	fmt.Printf("Expanding vocabulary with words from %s\n", path)
	file, err := os.Open(path)
	if err != nil {
		fmt.Printf("Warning: could not open WikiQA data to expand vocabulary: %v\n", err)
		return
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.SplitN(line, "\t", 4) // Split into max 4 parts
		if len(parts) > 1 {                   // Ensure there's at least a question and answer
			question := parts[0]             // The question
			words := strings.Fields(question) // Then process the question
			for _, word := range words {
				cleanedWord := strings.ToLower(strings.Trim(word, ".,!?;:\"'()[]{}-_"))
				if cleanedWord != "" {
					wordSet[cleanedWord] = true
				}
			}
		}
	}
}

// addWordsFromRAGJSON extracts words from the structured RAG JSON data.
func addWordsFromRAGJSON(path string, wordSet map[string]bool) {
	fmt.Printf("Expanding vocabulary with words from %s\n", path)
	file, err := os.Open(path)
	if err != nil {
		fmt.Printf("Warning: could not open RAG JSON data to expand vocabulary: %v\n", err)
		return
	}
	defer file.Close()

	var ragData []struct {
		Content string `json:"Content"`
	}
	if err := json.NewDecoder(file).Decode(&ragData); err != nil {
		fmt.Printf("Warning: could not decode RAG JSON data from %s: %v\n", path, err)
		return
	}

	for _, entry := range ragData {
		words := strings.Fields(entry.Content)
		for _, word := range words {
			cleanedWord := strings.ToLower(strings.Trim(word, ".,!?;:\"'()[]{}-_"))
			if cleanedWord != "" {
				wordSet[cleanedWord] = true
			}
		}
	}
}

// addWordsFromHTML extracts text content from an HTML file to expand the vocabulary.
func addWordsFromHTML(path string, wordSet map[string]bool) {
	fmt.Printf("Expanding vocabulary with words from %s\n", path)
	file, err := os.Open(path)
	if err != nil {
		fmt.Printf("Warning: could not open docs file to expand vocabulary: %v\n", err)
		return
	}
	defer file.Close()

	tokenizer := html.NewTokenizer(file)
htmlLoop:
	for {
		tt := tokenizer.Next()
		switch tt {
		case html.ErrorToken:
			if tokenizer.Err() != io.EOF {
				fmt.Printf("Warning: error parsing HTML from %s: %v\n", path, tokenizer.Err())
			}
			break htmlLoop // End of the document
		case html.TextToken:
			words := strings.Fields(string(tokenizer.Text()))
			for _, word := range words {
				cleanedWord := strings.ToLower(strings.Trim(word, ".,!?;:\"'()[]{}-_"))
				if cleanedWord != "" {
					wordSet[cleanedWord] = true
				}
			}
		}
	}
}

// setupModel loads a model from modelPath or creates a new one if loading fails.
func setupModel(modelPath string, vocabulary *bartsimple.Vocabulary, dim, heads, maxLen int) (*bartsimple.SimplifiedBARTModel, error) {
	// Attempt to load the model
	model, err := bartsimple.LoadSimplifiedBARTModelFromGOB(modelPath)
	if err == nil && model != nil {
		// Check if the loaded model's vocabulary size matches the current vocabulary.
		// This is a critical check to prevent panics from using an old model with a new, larger vocabulary.
		if model.VocabSize == len(vocabulary.WordToToken) {
			fmt.Printf("Successfully loaded simplified BART model from %s\n", modelPath)
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

	tokenizer, tknErr := bartsimple.NewTokenizer(vocabulary, vocabulary.BeginningOfSentenceID, vocabulary.EndOfSentenceID, vocabulary.PaddingTokenID, vocabulary.UnknownTokenID)
	if tknErr != nil {
		return nil, fmt.Errorf("failed to create tokenizer for new model: %w", tknErr)
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

func runTraining(model *bartsimple.SimplifiedBARTModel, bartDataPath, modelPath string) {
	fmt.Println("--- Running in Training Mode ---")

	// 1. Load BART-specific training data
	bartTrainingData, err := bartsimple.LoadBARTTrainingData(bartDataPath)
	if err != nil {
		log.Fatalf("Error loading BART training data from %s: %v", bartDataPath, err)
	}
	fmt.Printf("Loaded %d training sentences for BART model.\n", len(bartTrainingData.Sentences))
	// 2. Train the model
	err = bartsimple.TrainBARTModel(model, bartTrainingData, *epochs, *learningRate, *batchSize)
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

func runInference(bartModel *bartsimple.SimplifiedBARTModel, bertModel *bert.BertModel, bertConfig bert.BertConfig) {
	// Define a map of intents to functions
	intentMap := map[string]func(args []string){
		"CREATE_WEBSERVER": createWebserver,
		// Add other intents here
	}

	fmt.Println("--- Running in Inference Mode ---")
	for {
		command := InputScanDirections("\nEnter a command (or 'quit' to exit):")
		if strings.ToLower(command) == "quit" {
			fmt.Println("Exiting.")
			break
		}
		if command == "" {
			continue
		}

		// 1. Use the BERT model to predict the intent
		intent, err := bertModel.BertProcessCommand(command, bertConfig, bartModel.Tokenizer)
		if err != nil {
			log.Printf("Error processing command with BERT model: %v", err)
			continue
		}

		fmt.Printf("Predicted Intent: %s\n", intent)

		// 2. Execute the action based on the intent
		if action, ok := intentMap[intent]; ok {
			// For now, we are not extracting entities. We will pass the raw command.
			args := strings.Fields(command)
			action(args)
		} else {
			fmt.Printf("Unknown intent: %s\n", intent)
		}
	}
}
type BARTTrainingPair struct {
Input string
Output string

}
func updateTrainingData(input, output, path string) {
	// Load existing data
	file, err := os.ReadFile(path)
	if err != nil {
		log.Printf("Could not read training data file: %v", err)
		return
	}

	var data struct {
		Sentences []BARTTrainingPair `json:"sentences"`
	}
	if err := json.Unmarshal(file, &data); err != nil {
		log.Printf("Could not unmarshal training data: %v", err)
		return
	}

	// Add new data
	data.Sentences = append(data.Sentences, BARTTrainingPair{Input: input, Output: output})

	// Write updated data back to the file
	updatedData, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		log.Printf("Could not marshal updated training data: %v", err)
		return
	}

	if err := os.WriteFile(path, updatedData, 0644); err != nil {
		log.Printf("Could not write updated training data: %v", err)
		return
	}

	fmt.Println("Training data updated. Please retrain the model with --train-bart to apply the changes.")
}

// createWebserver is a placeholder function to demonstrate command execution.
func createWebserver(args []string) {
	if len(args) < 4 {
		fmt.Println("Usage: create webserver <name> with handler <handler_name>")
		return
	}
	serverName := args[0]
	handlerName := args[3]
	fmt.Printf("Creating webserver named '%s' with handler '%s'\n", serverName, handlerName)
	// In a real application, you would generate the webserver code here.
}


// InputScanDirections prompts the user for input and returns the cleaned string.
func InputScanDirections(directions string) string {
	fmt.Println(directions)

	scannerdesc := bufio.NewScanner(os.Stdin)
	if scannerdesc.Scan() {
		dir := scannerdesc.Text()
		return strings.TrimSpace(dir)
	}
	if err := scannerdesc.Err(); err != nil {
		log.Printf("Error reading input: %v", err)
	}
	return ""
}