package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"runtime/pprof"
	"strings"
	"testing"

	"github.com/golangast/nlptagger/neural/nn/g"
	"github.com/golangast/nlptagger/neural/nn/semanticrole"
	"github.com/golangast/nlptagger/neural/nnu"
	"github.com/golangast/nlptagger/neural/nnu/intent"
	"github.com/golangast/nlptagger/neural/nnu/train"
	"github.com/golangast/nlptagger/neural/nnu/word2vec"
)

var cpuprofile = flag.String("cpuprofile", "", "write cpu profile to file")

func BenchmarkMain(b *testing.B) {

	flag.Parse()
	if *cpuprofile != "" {
		f, err := os.Create(*cpuprofile)
		if err != nil {
			log.Fatal(err)
		}
		defer f.Close()
		if err := pprof.StartCPUProfile(f); err != nil {
			log.Fatal(err)
		}
		defer pprof.StopCPUProfile()
	}
	defer pprof.StopCPUProfile()

	// Memory profiling
	memFile, err := os.CreateTemp("", "mem.prof")
	if err != nil {
		b.Fatalf("could not create memory profile: %v", err)
	}
	defer os.Remove(memFile.Name()) // Clean up the temporary file
	if err != nil {
		b.Fatalf("could not create memory profile: %v", err)
	}
	defer memFile.Close()

	for i := 0; i < b.N; i++ {

		vectorsize := 50
		window := 5
		epochs := 1
		hiddensize := 10
		learningrate := 0.0001
		maxgrad := 0.01
		similaritythreshold := 0.9
		model := "false"

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
		//com := InputScanDirections("what would you like to do?")

		com := "generate a webserver named jim and handler named jill"

		//process command
		tokens := strings.Split(com, " ")
		// 3. Load SRL training data

		_, err = semanticRoleModel.PredictRoles(tokens)
		if err != nil {
			fmt.Println("Error predicting semantic roles:", err)
			return
		}

		intents, err := i.ProcessCommand(com, sw2v.Ann.Index, c)
		if err != nil {
			fmt.Println("Error in ProcessCommand:", err)
		}

		fmt.Println(intents)
	}
	if err := pprof.Lookup("heap").WriteTo(memFile, 0); err != nil {
		log.Fatalf("could not write memory profile: %v", err)
	}
}
