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

# nlptagger

A versatile, high-performance Natural Language Processing (NLP) toolkit written entirely in **Go** (Golang). The project provides a command-line utility for training and utilizing foundational NLP models, including **Word2Vec** embeddings, a sophisticated **Mixture of Experts (MoE)** model, and a practical **Intent Classifier**.

-----

## ✨ Key Features

The application is structured as a dispatcher that runs specialized modules for various NLP tasks:

  * **Word2Vec Training**: Generate high-quality distributed word representations (embeddings) from a text corpus.
  * **Mixture of Experts (MoE) Architecture**: Train a powerful MoE model, designed for improved performance, scalability, and handling of complex sequential or structural data.
  * **Intent Classification**: Develop a model for accurately categorizing user queries into predefined semantic intents.
  * **Efficient Execution**: Built in Go, leveraging its performance and concurrency features for fast training and inference.

-----

## 🚀 Getting Started

### Prerequisites

You need a working **Go environment** (version 1.18 or higher is recommended) installed on your system.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/golangast/nlptagger.git
    cd nlptagger
    ```

-----

## 🛠️ Usage

The main executable (`main.go`) controls all operations using specific command-line flags. All commands should be run from the root directory of the project.

### 1\. Training Models

Use the respective flags to initiate the training process. Each flag executes a separate module located in the `cmd/` directory.

| Model | Flag | Command |
| :--- | :--- | :--- |
| **Word2Vec** | `--train-word2vec` | `go run main.go --train-word2vec` |
| **Mixture of Experts (MoE)** | `--train-moe` | `go run main.go --train-moe` |
| **Intent Classifier** | `--train-intent-classifier` | `go run main.go --train-intent-classifier` |

### 2\. Running MoE Inference

To run predictions using a previously trained MoE model, use the `--moe_inference` flag and pass the input query string.

| Action | Flag | Command Example |
| :--- | :--- | :--- |
| **MoE Inference** | `--moe_inference` | `go run main.go --moe_inference "schedule a meeting with John for tomorrow at 2pm"` |

### 3\. Help / No Action

If no flags are provided, the application will prompt the user to specify an action:

```
$ go run main.go
2025/10/05 07:35:00 No action specified. Use -train-word2vec, -train-moe, -train-intent-classifier, or -moe_inference <query>.
```

-----

## ⚙️ Project Structure (Inferred)

The architecture is modular, with the main file dispatching tasks to specialized packages in the `cmd/` directory. This separation ensures clean code and independent development of model components.

```
nlptagger/
├── main.go                       # Main entry point and command dispatcher.
├── go.mod                        # Go module file.
├── cmd/                          # Directory for all specialized command-line modules
│   ├── train_word2vec/           # Handles all Word2Vec training logic.
│   ├── train_moe/                # Handles all MoE training and model saving logic.
│   ├── train_intent_classifier/  # Handles all Intent Classifier training logic.
│   └── moe_inference/            # Handles MoE model loading and prediction logic.
└── ...                           # Other internal packages (models, utils, data handling, etc.)
```

-----

## 📊 Data & Configuration (Inferred)

While not specified in `main.go`, training modules typically require input data and configuration.

  * **Data Structure**: Training modules are assumed to look for data files (e.g., plain text corpus, JSON/CSV files for labeled data) in pre-defined locations (e.g., a `data/` directory or specified via internal configuration flags).
  * **Configuration**: Model hyperparameters (learning rate, epochs, vector size, etc.) are likely managed either through configuration files (e.g., `config.json`, YAML) within the `cmd/` sub-packages or additional flags not exposed by `main.go`.
  * **Model Output**: Trained models are saved to disk, typically in a `models/` directory, for later use by the inference module.

-----

## 🤝 Contributing

We welcome contributions\! Please feel free to open issues for bug reports or feature requests, or submit pull requests for any enhancements.

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

-----


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


