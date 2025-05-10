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
  - [Repository overview](#repository-overview)
  - [Overview of the code.](#Overview-of-the-code.)
  - [Things to remember](#things-to-remember)
  - [Reference Commands](#reference-commands)
  - [Special thanks](#special-thanks)
  - [Why Go?](#why-go)
  - [Just added](#just-added)


## General info
*This project is used for tagging cli commands.  It is not a LLM or trying to be.  I am using it to generate go code but I made this completely separate so others can enjoy it.
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
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/golangast/nlptagger/neural/nn/g"
	"github.com/golangast/nlptagger/neural/nn/semanticrole"
	"github.com/golangast/nlptagger/neural/nnu"
	"github.com/golangast/nlptagger/neural/nnu/intent"
	"github.com/golangast/nlptagger/neural/nnu/rag"
	"github.com/golangast/nlptagger/neural/nnu/train"
	"github.com/golangast/nlptagger/neural/nnu/word2vec"
)

var model = "true"
var hiddensize = 100
var vectorsize = 100
var window = 10
var epochs = 1
var learningrate = 0.01
var maxgrad = 20.0
var similaritythreshold = .6
var logfile = "train.log"

func init() {
	flag.StringVar(&model, "model", "true", "whether or not to use model or manual")
	flag.IntVar(&hiddensize, "hiddensize", 100, "hiddensize determines the number of neurons in the hidden layer")
	flag.IntVar(&vectorsize, "vectorsize", 100, "VectorSize can allow for a more nuanced representation of words")
	flag.IntVar(&window, "window", 10, "Context window size")
	flag.IntVar(&epochs, "epochs", 1, "Number of training epochs")
	flag.Float64Var(&learningrate, "learningrate", 0.01, "Learning rate")
	flag.Float64Var(&maxgrad, "maxgrad", 20, "updates to the model's weights are kept within a reasonable range")
	flag.Float64Var(&similaritythreshold, "similaritythreshold", .6, "Its purpose is to refine the similarity calculations, ensuring a tighter definition of similarity and controlling the results")
	flag.StringVar(&logfile, "logFile", "train.log", "Path to the log file")
	flag.Parse()
	f, err := os.OpenFile(logfile, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
	if err != nil {
		log.Fatalf("error opening file: %v", err)
	}
	defer f.Close()
	log.SetOutput(f)
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	log.Printf("Starting training with model=%s, epochs=%d, learningRate=%f, vectorSize=%d, hiddenSize=%d, maxGrad=%f, window=%d", model, epochs, learningrate, vectorsize, hiddensize, maxgrad, window) // Log hyperparameters
}

/*
check if you are running it manually or not.

	manuallly..
	go run . -model true  -epochs 100 -learningrate 0.1 -hiddensize 100 -vectorsize 100 -window 10 -maxgrad 20 -similaritythreshold .6
	automatically...
	 go run .
*/
func main() {
	trainWord2VecModel()
}

func trainWord2VecModel() {

	var sw2v *word2vec.SimpleWord2Vec
	var err error

	if model == "true" {
		var err error
		sw2v, err = word2vec.LoadModel("trained_model.gob")
		if err != nil {
			fmt.Println("Error loading the model in loadmodel:", err)
		}
	}

	sw2v = &word2vec.SimpleWord2Vec{
		Vocabulary:          make(map[string]int),
		WordVectors:         make(map[int][]float64),
		VectorSize:          vectorsize, // each word in the vocabulary is represented by a vector of VectorSize numbers. A larger VectorSize can allow for a more nuanced representation of words, but it also increases the computational cost of training and storage.
		ContextEmbeddings:   make(map[string][]float64),
		Window:              window, // Example context window size
		Epochs:              epochs,
		ContextLabels:       make(map[string]string),
		UNKToken:            "<UNK>",
		HiddenSize:          hiddensize, // This means hiddensize determines the number of neurons in the hidden layer. A larger hidden size usually allows the network to learn more complex patterns, but also increases the computational resources required.
		LearningRate:        learningrate,
		MaxGrad:             maxgrad,             //Exploding gradients occur when the gradients during training become excessively large, causing instability and hindering the learning process. By limiting the norm of the gradients to maxGrad, the updates to the model's weights are kept within a reasonable range, promoting more stable and effective training.
		SimilarityThreshold: similaritythreshold, //Its purpose is to refine the similarity calculations, ensuring a tighter definition of similarity and controlling the results
	}
	sw2v.Ann, err = g.NewANN(sw2v.VectorSize, "euclidean")
	if err != nil {
		fmt.Println("Error creating ANN:", err) // Handle the error properly
		return                                  // Exit if ANN creation fails
	}

	nn := nnu.NewSimpleNN("datas/tagdata/training_data.json")
	// Train the model
	c, err := train.JsonModelTrain(sw2v, nn)
	if err != nil {
		fmt.Println("Error in JsonModelTrain:", err)
	}

	// Save the trained model
	err = sw2v.SaveModel("./gob_models/trained_model.gob")
	if err != nil {
		fmt.Println("Error saving the model:", err)
	}

	i := intent.IntentClassifier{}
	//com := InputScanDirections("what would you like to do?")

	intents, err := i.ProcessCommand("generate a webserver named jim and handler named jill", sw2v.Ann.Index, c)
	if err != nil {
		fmt.Println("Error with ProcessCommand", err)
	}

	// Embed the command (user input) using the Word2Vec model.
	commandVector, err := embedCommand("generate a webserver named jim and handler named jill", sw2v)
	if err != nil {
		fmt.Println("Error embedding command:", err)
		return
	}
	//./gob_models/word2vec_model.gob
	myModel, err := semanticrole.NewSemanticRoleModel("./gob_models/trained_model.gob", "./gob_models/bilstm_model.gob", "./gob_models/role_map.gob")
	if err != nil {
		fmt.Println("Error creating SemanticRoleModel:", err)
	} else {
		fmt.Println("Semantic Role Model:", myModel)
	}
	// Load RAG documents.

	ragDocs, err := rag.ReadPlainTextDocuments("datas/ragdata/ragdocs.txt", sw2v) // Use the new function
	if err != nil {
		fmt.Println("Error reading document:", err)
		return
	}

	// fmt.Println("ragdocs before read:", ragDocs)
	if ragDocs.Documents == nil {
		fmt.Println("ragDocs is nil after reading the file")
		return
	}

	ragDocs.CalculateIDF() // Corrected the call to CalculateIDF

	relevantDocs := ragDocs.Search(commandVector, "generate a webserver named jim and handler named jill", similaritythreshold)

	// Incorporate relevant documents into the response.
	fmt.Println("~~~ this is the intent: ", intents+"\n")
	fmt.Println(" ")
	// add the docs information here
	if len(relevantDocs) > 0 {
		for i, doc := range relevantDocs {
			fmt.Println("Relevant Doc:", i, "--", doc.Content)
		}
		fmt.Println("Number of relevant documents found:", len(relevantDocs))
	} else {
		fmt.Println("No relevant documents found.")
	}

	relevantDocs = ragDocs.Search(commandVector, "generate a webserver named jim and handler named jill", similaritythreshold)

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

// embedCommand embeds the command using the Word2Vec model.
func embedCommand(command string, sw2v *word2vec.SimpleWord2Vec) ([]float64, error) {
	words := strings.Split(command, " ")
	var embeddings [][]float64
	for _, word := range words {
		if vector, ok := sw2v.WordVectors[sw2v.Vocabulary[word]]; ok {
			embeddings = append(embeddings, vector)
		} else {
			embeddings = append(embeddings, sw2v.WordVectors[sw2v.Vocabulary[sw2v.UNKToken]]) // Use UNK token embedding if word not found
		}

	}

	if len(embeddings) == 0 {
		return nil, fmt.Errorf("no embeddings found for command")
	}

	// Average the embeddings to get a command vector
	commandVector := make([]float64, len(embeddings[0]))
	for _, embedding := range embeddings {
		for i, val := range embedding {
			commandVector[i] += val
		}
	}
	for i := range commandVector {
		commandVector[i] /= float64(len(embeddings))
	}

	return commandVector, nil
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
* it is not a LLM or trying to be
* it is only for cli commands

 ```



## Just added
RAG


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
