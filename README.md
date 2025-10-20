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

### 3. Running Workflow Generation and Execution

The `example/main.go` program demonstrates how to parse a natural language query, generate a workflow, and execute it. This showcases the core capabilities of the `nlptagger` for understanding and acting upon user commands.

To run the example, use the following command with a query:

```bash
go run ./example/main.go -query "create folder jack with a go webserver jill"
```

You can also run it interactively:

```bash
go run ./example/main.go
```
Then, enter queries at the prompt.

Expected Output (for the query "create folder jack with a go webserver jill"):

```
Processing query: "create folder jack with a go webserver jill"

--- Generated Workflow (after inference and validation) ---
Node ID: Filesystem::Folder-jack-0, Operation: CREATE, Resource Type: Filesystem::Folder, Resource Name: jack, Properties: map[permissions:493], Command: , Dependencies: []
Node ID: Filesystem::File-jill-0, Operation: CREATE, Resource Type: Filesystem::File, Resource Name: jill, Properties: map[permissions:493], Command: , Dependencies: [Filesystem::Folder-jack-0]
Node ID: file-createfile-0, Operation: WRITE_FILE, Resource Type: , Resource Name: , Properties: map[], Command: , Dependencies: [Filesystem::File-jill-0]
```

## 🧩 Integrating `nlptagger` into Your Projects

This project is more than just command-line tools. It's a collection of Go packages. You can use these packages in your own Go projects.

Example usage is in the /example folder.

```go
package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"nlptagger/neural/parser"
	"nlptagger/neural/workflow"
)

var (
	query = flag.String("query", "", "Natural language query for the parser")
)

func main() {
	flag.Parse()

	// Create parser and executor instances
	p := parser.NewParser()
	executor := workflow.NewExecutor()

	// Process initial query from flag, if provided
	if *query != "" {
		processAndExecuteQuery(*query, p, executor)
	}

	// Start interactive loop
	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("\nEnter a query (e.g., \"create folder jack with a go webserver jill\"): ")
		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)

		if input == "exit" || input == "quit" {
			break
		}

		if input != "" {
			processAndExecuteQuery(input, p, executor)
		}
	}
}

func processAndExecuteQuery(q string, p *parser.Parser, executor *workflow.Executor) {
	log.Printf("Processing query: \"%s\"", q)

	// Parse the query into a workflow
	// The parser now handles semantic validation and inference internally.
	wf, err := p.Parse(q)
	if err != nil {
		log.Printf("Error parsing query: %v", err)
		return
	}

	fmt.Println("\n--- Generated Workflow (after inference and validation) ---")
	for _, node := range wf.Nodes {
		fmt.Printf("Node ID: %s, Operation: %s, Resource Type: %s, Resource Name: %s, Properties: %v, Command: %s, Dependencies: %v\n",
			node.ID, node.Operation, node.Resource.Type, node.Resource.Name, node.Resource.Properties, node.Command, node.Dependencies)
	}

	// Execute the generated workflow
	if err := executor.ExecuteWorkflow(wf); err != nil {
		log.Printf("Error executing workflow: %v", err)
		return
	}
}

```


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

## Future Direction: List of commands that it can generate.

### 1. Abstraction: Semantic Parsing and Ontology Mapping 🧠
Instead of merely tagging words, the NLP layer would generate an Abstract Semantic Graph (ASG) or Structured Object that represents the complete meaning, including implicit details, constraints, and dependencies.

**Current NLP Output (Intent Recognition):**

| Identified Elements | Values in Query |
| --- | --- |
| Parent Intent | webserver_creation |
| Child Intent | create |
| Object Types | folder, webserver |
| Names | jack, jill |


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