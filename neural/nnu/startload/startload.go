package startload

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/golangast/nlptagger/neural/nn/g"
	"github.com/golangast/nlptagger/neural/nn/semanticrole"
	"github.com/golangast/nlptagger/neural/nnu"
	"github.com/golangast/nlptagger/neural/nnu/intent"
	"github.com/golangast/nlptagger/neural/nnu/train"
	"github.com/golangast/nlptagger/neural/nnu/word2vec"
)

func StartLoad(epochs, vectorsize, hiddensize, window int, learningrate, maxgrad, similaritythreshold float64, logfile, model string) {
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

	// Load the semantic role model
	semanticRoleModel, err := semanticrole.NewSemanticRoleModel("trained_model.gob", "bilstm_model.gob", "role_map.gob")
	if err != nil {
		fmt.Println("Error loading semantic role model:", err)
		return
	}

	// the model is saved because I am not sure if you
	//changed how the model is run
	err = sw2v.SaveModel("trained_model.gob")
	if err != nil {
		fmt.Println("Error saving the model:", err)
	}

	var i intent.IntentClassifier

	//ask command
	com := InputScanDirections("what would you like to do?")

	//process command
	tokens := strings.Split(com, " ")
	// 3. Load SRL training data

	predictedRoles, err := semanticRoleModel.PredictRoles(tokens)
	if err != nil {
		fmt.Println("Error predicting semantic roles:", err)
		return
	}
	fmt.Println("Predicted Roles:", predictedRoles)

	intents, err := i.ProcessCommand(com, sw2v.Ann.Index, c)
	if err != nil {
		fmt.Println("Error in ProcessCommand:", err)
	}

	//get the intent of the command
	fmt.Println("~~~ this is the intent: ", intents)

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
