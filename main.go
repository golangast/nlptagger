package main

import (
	"flag"
	"log"
	"os"

	"github.com/golangast/nlptagger/crf/crf_model"
	"github.com/golangast/nlptagger/neural/nnu/startload"
)

var (
	epochs, vectorsize, hiddensize, window     int
	learningrate, maxgrad, similaritythreshold float64
	logfile, model                             string
)

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

	startload.StartLoad(epochs, vectorsize, hiddensize, window, learningrate, maxgrad, similaritythreshold, logfile, model)

}
