# nlptagger

[![Go Report Card](https://goreportcard.com/badge/github.com/golangast/nlptagger)](https://goreportcard.com/report/github.com/golangast/nlptagger)
[![GoDoc](https://img.shields.io/badge/godoc-reference-blue.svg)](https://pkg.go.dev/github.com/golangast/nlptagger)
[![Go Version](https://img.shields.io/github/go-mod/go-version/golangast/nlptagger)](https://github.com/golangast/nlptagger)
![GitHub top language](https://img.shields.io/github/languages/top/golangast/nlptagger)
[![GitHub license](https://img.shields.io/github/license/golangast/nlptagger)](https://github.com/golangast/nlptagger/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/golangast/nlptagger)](https://github.com/golangast/nlptagger/stargazers)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/m/golangast/nlptagger)](https://github.com/golangast/nlptagger/graphs/commit-activity)
![GitHub repo size](https://img.shields.io/github/repo-size/golangast/nlptagger)
![Status](https://img.shields.io/badge/Status-Beta-red)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/golangast/nlptagger/graphs/commit-activity)
[![saythanks](https://img.shields.io/badge/say-thanks-ff69b4.svg)](https://saythanks.io/to/golangast)

A versatile, high-performance Natural Language Processing (NLP) toolkit written entirely in **Go** (Golang). The project provides a command-line utility for training and utilizing foundational NLP models, including **Word2Vec** embeddings, a sophisticated **Mixture of Experts (MoE)** model, and a practical **Intent Classifier**.

> **Note:** This project is currently in a beta stage and is under active development. The API and functionality are subject to change. Accuracy is not the primary focus at this stage, as the main goal is to explore and implement these NLP models in Go.

## Table of Contents

- [🌐 Project Website](https://golangast.github.io/nlptagger/)
- [✨ Key Features](#-key-features)
- [🚀 Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Building from Source](#building-from-source)
- [🛠️ Usage](#️-usage)
  - [Training Models](#1-training-models)
  - [Running MoE Inference](#2-running-moe-inference)
- [⚙️ Project Structure](#️-project-structure)
- [📊 Data & Configuration](#-data--configuration)
- [🗺️ Roadmap](#️-roadmap)
- [Future Direction: Semantic Parsing and Reasoning](#future-direction-semantic-parsing-and-reasoning)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)
- [🙏 Special Thanks](#-special-thanks)
- [Why Go?](#why-go)

## ✨ Key Features

The application is structured as a dispatcher that runs specialized modules for various NLP tasks:

*   **Word2Vec Training**: Generate high-quality distributed word representations (embeddings) from a text corpus.
*   **Mixture of Experts (MoE) Architecture**: Train a powerful MoE model, designed for improved performance, scalability, and handling of complex sequential or structural data.
*   **Intent Classification**: Develop a model for accurately categorizing user queries into predefined semantic intents.
*   **Efficient Execution**: Built in Go, leveraging its performance and concurrency features for fast training and inference.

## 🚀 Getting Started

### Prerequisites

You need a working **Go environment** (version 1.25 or higher is recommended) installed on your system.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/golangast/nlptagger.git
    cd nlptagger
    ```

### Building from Source

You can build the executable from the root of the project directory:

```bash
go build .
```

This will create an `nlptagger` executable in the current directory.

## 🛠️ Usage

The main executable (`nlptagger` or `main.go`) controls all operations using specific command-line flags. All commands should be run from the root directory of the project.

### 1. Training Models

Use the respective flags to initiate the training process. Each flag executes a separate module located in the `cmd/` directory.

| Model                 | Flag                      | Command                               |
| :-------------------- | :------------------------ | :------------------------------------ |
| **Word2Vec**          | `--train-word2vec`        | `go run main.go --train-word2vec`     |
| **Mixture of Experts (MoE)** | `--train-moe`             | `go run main.go --train-moe`          |
| **Intent Classifier** | `--train-intent-classifier` | `go run main.go --train-intent-classifier` |

### 2. Running MoE Inference

To run predictions using a previously trained MoE model, use the `--moe_inference` flag and pass the input query string.

| Action          | Flag              | Command Example                                                              |
| :---------------- | :---------------- | :--------------------------------------------------------------------------- |
| **MoE Inference** | `--moe_inference` | `go run main.go --moe_inference "schedule a meeting with John for tomorrow at 2pm"` |

### 3. Running Command Generation

To run the command generation logic based on intent predictions and NER/POS tags, use the `example/main.go` with a query:

```bash
go run ./example/main.go -query "create a file named jack in folder named jill"
```

Expected Output:

```
--- Top 3 Parent Intent Predictions ---
  - Word: webserver_creation   (Confidence: 29.17%)
  - Word: version_control      (Confidence: 20.43%)
  - Word: deployment           (Confidence: 10.19%)
------------------------------------
--- Top 3 Child Intent Predictions ---
  - Word: create               (Confidence: 17.68%)
  - Word: create_data_structure (Confidence: 14.66%)
  - Word: create_server_and_handler (Confidence: 11.70%)
-----------------------------------

Description: The model's top prediction is an action related to webserver_creation, specifically to create.
Intent Sentence: Not found in training data.

--- POS Tagging Results ---
Tokens: [create a file named jack in folder named jill]
POS Tags: [CODE_VB DET IN NN NN IN NN VBN NN]

--- NER Tagging Results ---
Tokens: [create a file named jack in folder named jill]
NER Tags: [COMMAND DETERMINER OBJECT_TYPE NAME_PREFIX NAME PREPOSITION OBJECT_TYPE NAME_PREFIX NAME]

--- Generating Command ---
Generated Command: mkdir -p jill && touch jill/jack
```

## 🧩 Integrating `nlptagger` into Your Projects

This project is more than just command-line tools. It's a collection of Go packages. You can use these packages in your own Go projects.

Example usage is in the /example folder.

```go
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math" // Keep math for softmax
	"math/rand"
	"os"
	"os/exec"
	"sort"

	"nlptagger/neural/moe"
	mainvocab "nlptagger/neural/nnu/vocab"
	"nlptagger/neural/tensor"
	"nlptagger/neural/tokenizer"
	"nlptagger/tagger/nertagger"
	"nlptagger/tagger/postagger"
	"nlptagger/tagger/tag"
)

// IntentTrainingExample represents a single training example for intent classification.
type IntentTrainingExample struct {
	Query        string `json:"query"`
	ParentIntent string `json:"parent_intent"`
	ChildIntent  string `json:"child_intent"`
	Description  string `json:"description"`
	Sentence     string `json:"sentence"`
}

// IntentTrainingData represents the structure of the intent training data JSON.
type IntentTrainingData []IntentTrainingExample

// LoadIntentTrainingData loads the intent training data from a JSON file.
func LoadIntentTrainingData(filePath string) (*IntentTrainingData, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open training data file %s: %w", filePath, err)
	}
	defer file.Close()

	bytes, err := io.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("failed to read training data file %s: %w", filePath, err)
	}

	var data IntentTrainingData
	err = json.Unmarshal(bytes, &data)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal training data JSON from %s: %w", filePath, err)
	}

	return &data, nil
}

type Prediction struct {
	TokenID    int
	Word       string
	Confidence float64
}

func getTopNPredictions(probabilities []float64, vocab []string, n int) []Prediction {
	predictions := make([]Prediction, 0, len(probabilities))
	for i, p := range probabilities {
		if i < 2 { // Skip <pad> and UNK
			continue
		}
		if i < len(vocab) {
			word := vocab[i]
			predictions = append(predictions, Prediction{
				TokenID:    i,
				Word:       word,
				Confidence: p * 100.0,
			})
		}
	}

	// Sort predictions by confidence
	sort.Slice(predictions, func(i, j int) bool {
		return predictions[i].Confidence > predictions[j].Confidence
	})

	if len(predictions) < n {
		return predictions
	}
	return predictions[:n]
}

var (
	query        = flag.String("query", "", "Query for MoE inference")
	maxSeqLength = flag.Int("maxlen", 32, "Maximum sequence length")
)

func main() {
	rand.Seed(1) // Seed the random number generator for deterministic behavior
	flag.Parse()

	if *query == "" {
		log.Fatal("Please provide a query using the -query flag.")
	}

	// Define paths
	const vocabPath = "gob_models/query_vocabulary.gob"
	const moeModelPath = "gob_models/moe_classification_model.gob"
	const parentIntentVocabPath = "gob_models/parent_intent_vocabulary.gob"
	const childIntentVocabPath = "gob_models/child_intent_vocabulary.gob"
	const intentTrainingDataPath = "trainingdata/intent_data.json"

	// Load vocabularies
	vocabulary, err := mainvocab.LoadVocabulary(vocabPath)
	if err != nil {
		log.Fatalf("Failed to set up input vocabulary: %v", err)
	}

	// Setup parent intent vocabulary
	parentIntentVocabulary, err := mainvocab.LoadVocabulary(parentIntentVocabPath)
	if err != nil {
		log.Fatalf("Failed to set up parent intent vocabulary: %v", err)
	}

	// Setup child intent vocabulary
	childIntentVocabulary, err := mainvocab.LoadVocabulary(childIntentVocabPath)
	if err != nil {
		log.Fatalf("Failed to set up child intent vocabulary: %v", err)
	}

	// Create tokenizer
	tok, err := tokenizer.NewTokenizer(vocabulary)
	if err != nil {
		log.Fatalf("Failed to create tokenizer: %w", err)
	}

	// Load the trained MoEClassificationModel model
	model, err := moe.LoadIntentMoEModelFromGOB(moeModelPath)
	if err != nil {
		log.Fatalf("Failed to load MoE model: %v", err)
	}

	// Load intent training data
	intentTrainingData, err := LoadIntentTrainingData(intentTrainingDataPath)
	if err != nil {
		log.Fatalf("Failed to load intent training data: %v", err)
	}

	log.Printf("--- DEBUG: Parent Intent Vocabulary (TokenToWord): %v ---", parentIntentVocabulary.TokenToWord)
	log.Printf("--- DEBUG: Child Intent Vocabulary (TokenToWord): %v ---", childIntentVocabulary.TokenToWord)

	log.Printf("Running MoE inference for query: \"%s\"", *query)

	// Encode the query
	tokenIDs, err := tok.Encode(*query)
	if err != nil {
		log.Fatalf("Failed to encode query: %v", err)
	}

	// Pad or truncate the sequence to a fixed length
	if len(tokenIDs) > *maxSeqLength {
		tokenIDs = tokenIDs[:*maxSeqLength] // Truncate from the end
	} else {
		for len(tokenIDs) < *maxSeqLength {
			tokenIDs = append(tokenIDs, vocabulary.PaddingTokenID) // Appends padding
		}
	}
	inputData := make([]float64, len(tokenIDs))
	for i, id := range tokenIDs {
		inputData[i] = float64(id)
	}
	inputTensor := tensor.NewTensor([]int{1, len(inputData)}, inputData, false) // RequiresGrad=false for inference

	// Create a dummy target tensor for inference, as the Forward method expects two inputs.
	// The actual content of this tensor won't be used for parent/child intent classification.
	dummyTargetTokenIDs := make([]float64, *maxSeqLength)
	for i := 0; i < *maxSeqLength; i++ {
		dummyTargetTokenIDs[i] = float64(vocabulary.PaddingTokenID)
	}
	dummyTargetTensor := tensor.NewTensor([]int{1, *maxSeqLength}, dummyTargetTokenIDs, false)

	// Forward pass
	parentLogits, childLogits, _, _, err := model.Forward(inputTensor, dummyTargetTensor)
	if err != nil {
		log.Fatalf("MoE model forward pass failed: %v", err)
	}

	// Interpret parent intent output
	parentProbabilities := softmax(parentLogits.Data)
	topParentPredictions := getTopNPredictions(parentProbabilities, parentIntentVocabulary.TokenToWord, 3)

	fmt.Println("--- Top 3 Parent Intent Predictions ---")
	for _, p := range topParentPredictions {
		importance := ""
		if p.Confidence > 50.0 {
			importance = " (Important)"
		}
		fmt.Printf("  - Word: %-20s (Confidence: %.2f%%)%s\n", p.Word, p.Confidence, importance)
	}
	fmt.Println("------------------------------------")

	// Interpret child intent output
	childProbabilities := softmax(childLogits.Data)
	topChildPredictions := getTopNPredictions(childProbabilities, childIntentVocabulary.TokenToWord, 3)

	fmt.Println("--- Top 3 Child Intent Predictions ---")
	for _, p := range topChildPredictions {
		importance := ""
		if p.Confidence > 50.0 {
			importance = " (Important)"
		}
		fmt.Printf("  - Word: %-20s (Confidence: %.2f%%)%s\n", p.Word, p.Confidence, importance)
	}
	fmt.Println("-----------------------------------")

	if len(topParentPredictions) > 0 && len(topChildPredictions) > 0 {
		predictedParentWord := topParentPredictions[0].Word
		predictedChildWord := topChildPredictions[0].Word
		fmt.Printf("\nDescription: The model's top prediction is an action related to %s, specifically to %s.\n", predictedParentWord, predictedChildWord)

		// Find and print the intent sentence
		foundSentence := ""
		for _, example := range *intentTrainingData {
			if example.ParentIntent == predictedParentWord && example.ChildIntent == predictedChildWord {
				foundSentence = example.Sentence
				break
			}
		}

		if foundSentence != "" {
			fmt.Printf("Intent Sentence: %s\n", foundSentence)
		} else {
			fmt.Println("Intent Sentence: Not found in training data.")
		}
	}

	// Perform POS tagging
	posResult := postagger.Postagger(*query)
	fmt.Println("\n--- POS Tagging Results ---")
	fmt.Printf("Tokens: %v\n", posResult.Tokens)
	fmt.Printf("POS Tags: %v\n", posResult.PosTag)

	// Perform NER tagging
	nerResult := nertagger.Nertagger(posResult)
	fmt.Println("\n--- NER Tagging Results ---")
	fmt.Printf("Tokens: %v\n", nerResult.Tokens)
	fmt.Printf("NER Tags: %v\n", nerResult.NerTag)

	// Generate and execute command based on NER/POS tags and intent predictions
	fmt.Println("\n--- Generating Command ---")
	command := generateCommand("file_system", topChildPredictions[0].Word, nerResult)
	if command != "" {
		fmt.Printf("Generated Command: %s\n", command)
		// Execute the command
		cmd := exec.Command("bash", "-c", command)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		err := cmd.Run()
		if err != nil {
			log.Printf("Error executing command: %v", err)
		}
	} else {
		fmt.Println("Could not generate a command.")
	}
}

// softmax applies the softmax function to a slice of float64.
func softmax(logits []float64) []float64 {
	if len(logits) == 0 {
		return []float64{}
	}
	maxLogit := logits[0]
	for _, logit := range logits {
		if logit > maxLogit {
			maxLogit = logit
		}
	}
	expSum := 0.0
	for _, logit := range logits {
		expSum += math.Exp(logit - maxLogit)
	}

	probabilities := make([]float64, len(logits))
	for i, logit := range logits {
		probabilities[i] = math.Exp(logit-maxLogit) / expSum
	}
	return probabilities
}

func generateCommand(parentIntent, childIntent string, nerResult tag.Tag) string {
	switch parentIntent {
	case "file_system":
		switch childIntent {
		case "create":
			var fileName, folderName string
			for i, tag := range nerResult.NerTag {
				if tag == "OBJECT_TYPE" && nerResult.Tokens[i] == "file" {
					if i+2 < len(nerResult.Tokens) && nerResult.NerTag[i+1] == "NAME_PREFIX" && nerResult.NerTag[i+2] == "NAME" {
						fileName = nerResult.Tokens[i+2]
					}
				} else if tag == "OBJECT_TYPE" && nerResult.Tokens[i] == "folder" {
					if i+2 < len(nerResult.Tokens) && nerResult.NerTag[i+1] == "NAME_PREFIX" && nerResult.NerTag[i+2] == "NAME" {
						folderName = nerResult.Tokens[i+2]
					}
				}
			}
			if fileName != "" && folderName != "" {
				return fmt.Sprintf("mkdir -p %s && touch %s/%s", folderName, folderName, fileName)
			} else if fileName != "" {
				return fmt.Sprintf("touch %s", fileName)
			}
		}
		// Add other file_system child intents here (e.g., "delete", "read")
	}
	// Add other parent intents here

	return ""
}
```

The output is usable and structured.
![Alt text](docs/img/out.png?raw=true "Title")


The `neural/` and `tagger/` directories contain the reusable components. Import them as needed.

## ⚙️ Project Structure


The project is a collection of tools. Its structure reflects this.

```
nlptagger/
├── main.go         # Dispatches to common tools.
├── go.mod          # Go module definition.
├── cmd/            # Each subdirectory is a command-line tool.
│   ├── train_word2vec/ # Example: Word2Vec training.
│   └── moe_inference/  # Example: MoE inference.
├── neural/         # Core neural network code.
├── tagger/         # NLP tagging components.
├── trainingdata/   # Sample data for training.
└── gob_models/     # Saved models.
```

## 📊 Data & Configuration

*   **Data Structure**: Training modules look for data files in the `trainingdata/` directory. For example, `intent_data.json` is used for intent classification training.
*   **Configuration**: Model hyperparameters (learning rate, epochs, vector size, etc.) are currently hardcoded within their respective training modules in the `cmd/` directory. This is an area for future improvement.
*   **Model Output**: Trained models are saved as `.gob` files to the `gob_models/` directory by default.

## 🗺️ Roadmap

This project is under active development. Here are some of the planned features and improvements:

- [ ] Implement comprehensive unit and integration tests.
- [ ] Add more NLP tasks (e.g., Named Entity Recognition, Part-of-Speech tagging).
- [ ] Externalize model configurations from code into files (e.g., YAML, JSON).
- [ ] Improve model accuracy and performance.
- [ ] Enhance documentation with more examples and API references.
- [ ] Create a more user-friendly command-line interface.

## Future Direction: Semantic Parsing and Reasoning

The next level of abstraction for the Natural Language Processing (NLP) portion of the nlptagger project would be to move from Intent Recognition to Semantic Parsing and Reasoning.The current NLP process extracts a fixed set of elements (Parent Intent, Child Intent, Object Types, Names) and maps them directly to a command template.The next abstraction would involve creating an internal, structured data model of the user's request that captures meaning and relationships, independent of the final command format.

### 1. Abstraction: Semantic Parsing and Ontology Mapping 🧠
Instead of merely tagging words, the NLP layer would generate an Abstract Semantic Graph (ASG) or Structured Object that represents the complete meaning, including implicit details, constraints, and dependencies.

**Current NLP Output (Intent Recognition):**

| Identified Elements | Values in Query |
| --- | --- |
| Parent Intent | webserver_creation |
| Child Intent | create |
| Object Types | folder, webserver |
| Names | jack, jill |

**Next NLP Abstraction (Semantic Output):**

The NLP model would output a structured Go object (or equivalent JSON/YAML) based on an internal Ontology (a formal definition of all possible resources and actions).

```json
{
  "operation": "Create",
  "target_resource": {
    "type": "Filesystem::Folder",
    "name": "jack",
    "properties": {
      "path": "./"
    },
    "children": [
      {
        "type": "Deployment::GoWebserver",
        "name": "jill",
        "properties": {
          "source": "boilerplate_v1",
          "port": 8080,
          "runtime": "go"
        }
      }
    ]
  },
  "context": {
    "user_role": "admin"
  }
}
```

### 2. Intelligent Capabilities Added by this Abstraction
This abstraction provides the foundation for truly intelligent command generation:

**A. Reasoning and Inference**
The new layer can handle implicit and contextual details (Reasoning).
*Example Query:* "Make 'jill' in 'jack' and expose the service publicly."
*Inference:* The system automatically infers that a "publicly exposed service" implies setting the webserver's port to be publicly accessible and potentially generating an extra LoadBalancer resource (if using a cloud execution backend).

**B. Dependency Resolution**
The NLP can identify causal and temporal relationships (Dependency).
*Example Query:* "Set up my Go server, but only after you create the database."
*Semantic Output:* The output graph would establish a `depends_on` relationship between the `Deployment::GoWebserver` and the `Data::PostgreSQL` resource, ensuring the command executor runs them in the correct sequence.

**C. Constraint and Policy Checking**
By mapping the request to a resource Ontology, the system can apply policy checks before generating or running any command.
*Example:* If the user attempts to create a folder with a restricted name, the semantic model could flag it:
*Semantic Output:* Resource name validation fails against corporate naming policy.
*Generated Response:* "Error: The name 'jack' is reserved for system use. Please choose a different name."

## 🤝 Contributing

We welcome contributions! Please feel free to open issues for bug reports or feature requests, or submit pull requests for any enhancements.

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m '''Add AmazingFeature'''`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

> **Note on Tests:** There is currently a lack of automated tests. Contributions in this area are highly encouraged and appreciated!

## 📜 License

This project is licensed under the **GNU General Public License v3.0**. See the [LICENSE](LICENSE) file for details.

## 🙏 Special Thanks

*   [The Go Team and contributors](https://github.com/golang/go/graphs/contributors) for creating and maintaining Go.

## Why Go?

Go is a great choice for this project for several reasons:

*   **Stability:** The language has a strong compatibility promise. What you learn now will be useful for a long time. ([Go 1 Compatibility Promise](https://go.dev/doc/go1compat))
*   **Simplicity and Readability:** Go's simple syntax makes it easy to read and maintain code.
*   **Performance:** Go is a compiled language with excellent performance, which is crucial for NLP tasks.
*   **Concurrency:** Go's built-in concurrency features make it easy to write concurrent code for data processing and model training.
*   **Strong Community and Ecosystem:** Go has a growing community and a rich ecosystem of libraries and tools. ([Go User Community](https://go.dev/wiki/GoUsers))