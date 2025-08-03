# nlptagger
This project is heavily under construction and will change a lot because I am learning as I am making and accuracy isn't #1 right now.


When you need a program to understand context of commands.


![GitHub repo file count](https://img.shields.io/github/directory-file-count/golangast/nlptagger) 
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/golangast/nlptagger)
![GitHub repo size](https://img.shields.io/github/repo-size/golangast/nlptagger)
![GitHub](https://img.shields.io/github/license/golangast/nlptagger)
![GitHub commit activity](https://img.shields.io/github/commit-activity/w/golangast/nlptagger)
![Go 100%](https://img.shields.io/badge/Go-100%25-blue)
![status beta](https://img.shields.io/badge/Status-Beta-red)
<img src="https://img.shields.io/github/license/golangast/nlptagger.svg"><img src="https://img.shields.io/github/stars/golangast/nlptagger.svg">[![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](http://endrulats.com)[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)[![GitHub go.mod Go version of a Go module](https://img.shields.io/github/go-mod/go-version/gomods/athens.svg)](https://github.com/golangast/nlptagger)[![GoDoc reference example](https://img.shields.io/badge/godoc-reference-blue.svg)](https://pkg.go.dev/github.com/golangast/nlptagger/nlptaggerer)[![GoReportCard example](https://goreportcard.com/badge/github.com/golangast/nlptagger)](https://goreportcard.com/report/github.com/golangast/nlptagger)[![saythanks](https://img.shields.io/badge/say-thanks-ff69b4.svg)](https://saythanks.io/to/golangast)



- [nlptagger](#nlptagger)
	- [General info](#general-info)
	- [Why build this?](#why-build-this)
	- [What does it do?](#what-does-it-do)
	- [Technologies](#technologies)
	- [Requirements](#requirements)
	- [How to run as is?](#how-to-run-as-is)
	- [Repository overview](#repository-overview)
	- [Overview of the code.](#overview-of-the-code)
	- [Just added](#just-added)
	- [Special thanks](#special-thanks)
	- [Why Go?](#why-go)


## General info
*This project is used for tagging cli commands.  I am using it to generate go code but I made this completely separate so others can enjoy it.
*I will keep working on it and hopefully improving the guessing of intent.

-Background
1. Tokenization: This is the very first step in most NLP pipelines. It involves breaking down text into individual units called tokens (words, punctuation marks, etc.). Tokenization is fundamental because it creates the building blocks for further analysis.

2. Part-of-Speech (POS) Tagging: POS tagging assigns grammatical categories (noun, verb, adjective, etc.) to each token. It's a crucial step for understanding sentence structure and is often used as input for more complex tasks like phrase tagging.

3. Named Entity Recognition (NER): NER identifies and classifies named entities (people, organizations, locations, dates, etc.) in text. This is more specific than POS tagging but still more generic than phrase tagging, as it focuses on individual entities rather than complete phrases.

4. Dependency Parsing: Dependency parsing analyzes the grammatical relationships between words in a sentence, creating a tree-like structure that shows how words depend on each other. It provides a deeper understanding of sentence structure than phrase tagging, which focuses on contiguous chunks.

5. Lemmatization and Stemming: These techniques reduce words to their base or root forms (e.g., "running" to "run"). They help to normalize text and improve the accuracy of other NLP tasks.

6. Word2Vec is a technique that represents words as numerical vectors capturing semantic relationships: words with similar meanings have closer vectors. This allows algorithms to understand and process text more effectively by leveraging word similarities.

7. Semantic roles describe the roles of words or phrases within a sentence, such as agent, action, or object. Identifying these roles helps to understand the meaning and relationships between different parts of a sentence.

8. Retrieval Augmented Generation (RAG) is a technique that enhances large language models (LLMs) by grounding them in external knowledge. It improves the accuracy, reliability, and context-awareness of LLMs by retrieving relevant information from a knowledge base and using it to inform their responses, enabling applications like more accurate question answering and the ability to utilize user-specific data while providing sources for the information.

*Phrase tagging often uses the output of these more generic techniques as input. For example:

POS tags are commonly used to define rules for identifying phrases (e.g., a noun phrase might be defined as a sequence of words starting with a determiner followed by one or more adjectives and a noun).
NER can be used to identify specific types of phrases (e.g., a phrase tagged as "PERSON" might indicate a person's name).

Bart sequence-to-sequence neural network that learns to translate natural language into a different, more structured format based on the examples you provide during training.


## Why build this?
* Go never changes
* It is nice to not have terminal drop downs


## What does it do?
* It tags words for commands.
*I made an overview video on this project but there have been a lot of changes.
[video](https://www.youtube.com/watch?v=KkcOn2rpD-c)



## Technologies
*Just Go.


## Requirements
* go 1.23 for gonew

## How to run as is?


```go
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

	"github.com/golangast/nlptagger/neural/nnu/bartsimple" // Assuming this is the correct import path
	"github.com/golangast/nlptagger/neural/nnu/vocab"
)

var (
	trainMode    = flag.Bool("train", false, "Enable training mode")
	epochs       = flag.Int("epochs", 10, "Number of training epochs")
	learningRate = flag.Float64("lr", 0.001, "Learning rate for training")
	bartDataPath = flag.String("data", "trainingdata/bartdata/bartdata.json", "Path to BART training data for the model")
	dimModel     = flag.Int("dim", 64, "Dimension of the model")
	numHeads     = flag.Int("heads", 4, "Number of attention heads")
	maxSeqLength = flag.Int("maxlen", 64, "Maximum sequence length")
	batchSize    = flag.Int("batchsize", 4, "Batch size for training")
)

func main() {
	flag.Parse()

	// Define paths, consider making these flags as well for more flexibility
	const modelPath = "gob_models/simplified_bart_model.gob"
	const trainingDataPath = "trainingdata/tagdata/nlp_training_data.json"
	const vocabPath = "gob_models/vocabulary.gob"

	vocabulary, err := setupVocabulary(vocabPath, trainingDataPath)
	if err != nil {
		log.Fatalf("Failed to set up vocabulary: %v", err)
	}

	model, err := setupModel(modelPath, vocabulary, *dimModel, *numHeads, *maxSeqLength)
	if err != nil {
		log.Fatalf("Failed to set up model: %v", err)
	}

	if *trainMode {
		runTraining(model, *bartDataPath, modelPath)
	} else {
		runInference(model)
	}
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

func runInference(model *bartsimple.SimplifiedBARTModel) {
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

		// Process the command using BartProcessCommand
		summary, err := model.BartProcessCommand(command)
		if err != nil {
			log.Printf("Error processing command with BART model: %v", err)
			continue // Continue to next loop iteration
		}
		fmt.Printf("Generated Summary: %s\n", summary)
	}
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

```

*- clone it
```bash
git clone https://github.com/golangast/nlptagger
```
* - or
* - install gonew to pull down project quickly
```bash
go install golang.org/x/tools/cmd/gonew@latest
```
* - run gonew
```bash
gonew github.com/golangast/nlptagger example.com/nlptagger
```

* - cd into nlptagger
=======
```bash
cd nlptagger
```

* - run the project
```bash
go run . -model true  -epochs 100 -learningrate 0.1 -hiddensize 100 -vectorsize 100 -window 10 -maxgrad 20 -similaritythreshold .6
```

## Repository overview
```bash
├── trainingdata #training data
│   └── contextdata #used for training the context model
│   └── ragdata #used for training the rag model
│   └── roledata #used for training the semantic role model
│   └── tagdata #used for training the tagger model
├── gob_models #model files
├── neural #neural network
│   ├── nn #neural networks for tagging
│       ├── dr #implements a neural network for dependency relation tagging.
│       ├── g #provides a basic implementation for an Approximate Nearest Neighbor search.
│       ├── ner #implements a basic neural network for Named Entity Recognition.
│       ├── phrase #provides a simple neural network for phrase tagging.
│       ├── pos #provides functions for Part-of-Speech tagging using a neural network.
│       ├── semanticrole #semantic role labeling using a BiLSTM and word embeddings.
│           ├── bilstm_model #provides a Bidirectional LSTM for semantic role labeling.
│           ├── train_bilstm.go #training a BiLSTM model for Semantic Role Labeling.
│   ├── nnu #neural network utils
│       ├── calc #provides functions for calculating neural network performance metrics.
│       ├── contextvector #contextvector computes context vectors for text.
│       ├── gobs #utility for creating gob files
│       ├── intent #interprets intent of the command and uses contextvector
│       ├── predict #predicting various tags for input text using a neural network.
│       ├── rag #functions for Retrieval Augmented Generation (RAG).
│       ├── train #loading data, preparing inputs and evaluating model accuracy.
│       ├── vacab #functions for creating and managing vocabularies
│       ├── word2vec #Word2Vec model for creating word embeddings.
│   └── sematicrole 
├── tagger #tagger folder
│   ├── dependencyrelation #dependency relation
│   ├── nertagger	#ner tagging
│   ├── phrasetagger #phraase tagging
│   ├── postagger #pos tagging
│   ├── stem #stemming tokens before tagging
│   ├── tag #tag data structure
│   └── tagger.go

```

## Overview of the code.
*Tries to guess intent of the program.
```
## Things to remember
* it is only for cli commands

 ```



## Just added
Bart model


## Special thanks
* [Go Team because they are gods](https://github.com/golang/go/graphs/contributors)


## Why Go?
* The language is done since 1.0.https://youtu.be/rFejpH_tAHM there are little features that get added after 10 years but whatever you learn now will forever be useful.
* It also has a compatibility promise https://go.dev/doc/go1compat
* It was also built by great people. https://hackernoon.com/why-go-ef8850dc5f3c
* 14th used language https://insights.stackoverflow.com/survey/2021
* Highest starred language https://github.com/golang/go
* It is also number 1 language to go to and not from https://www.jetbrains.com/lp/devecosystem-2021/#Do-you-plan-to-adopt--migrate-to-other-languages-in-the-next--months-If-so-to-which-ones
* Go is growing in all measures https://madnight.github.io/githut/#/stars/2023/3
* Jobs are almost doubling every year. https://stacktrends.dev/technologies/programming-languages/golang/
* Companies that use go. https://go.dev/wiki/GoUsers
* Why I picked Go https://youtu.be/fD005g07cU4
