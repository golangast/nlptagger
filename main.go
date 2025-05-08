package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/golangast/nlptagger/crf/crf_model"
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

type WordExample crf_model.WordExample
type TrainingExample crf_model.TrainingExample
type ViterbiOutput crf_model.ViterbiOutput

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

	intents, err := i.ProcessCommand("generate a webserver", sw2v.Ann.Index, c)
	if err != nil {
		fmt.Println("Error with ProcessCommand", err)
	}

	// Embed the command (user input) using the Word2Vec model.
	commandVector, err := embedCommand("generate a webserver", sw2v)
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
