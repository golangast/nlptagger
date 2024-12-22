package train

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"

	"golang.org/x/exp/rand"

	"github.com/golangast/nlptagger/neural/nn/dr"
	"github.com/golangast/nlptagger/neural/nn/ner"
	"github.com/golangast/nlptagger/neural/nn/phrase"
	"github.com/golangast/nlptagger/neural/nn/pos"
	"github.com/golangast/nlptagger/neural/nnu"
	"github.com/golangast/nlptagger/neural/nnu/calc"
	"github.com/golangast/nlptagger/neural/nnu/gobs"
	"github.com/golangast/nlptagger/neural/nnu/predict"
	"github.com/golangast/nlptagger/tagger/tag"
)

var (
	epochs       = flag.Int("epochs", 1000, "Number of training epochs")
	learningRate = flag.Float64("learningRate", 0.01, "Learning rate")
	logFile      = flag.String("logFile", "train.log", "Path to the log file")
)

func init() {
	flag.Parse()
	// Configure logging
	f, err := os.OpenFile(*logFile, os.O_RDWR|os.O_CREATE|os.O_APPEND, 0666)
	if err != nil {
		log.Fatalf("error opening file: %v", err)
	}
	defer f.Close()
	log.SetOutput(f)
	log.SetFlags(log.LstdFlags | log.Lshortfile)
}

// Structure to represent training data in JSON
type TrainingDataJSON struct {
	Sentences []tag.Tag `json:"sentences"`
}

func Train(trainingData []tag.Tag, epochs int, learningRate float64, nn *nnu.SimpleNN) (float64, float64, float64, float64) {

	log.Printf("Starting training with epochs=%d, learningRate=%f", epochs, learningRate) // Log hyperparameters

	var posaccuracy, neraccuracy, phraseaccuracy, draccuracy float64

	for epoch := 0; epoch < epochs; epoch++ {
		for _, taggedSentence := range trainingData {
			// Prepare inputs and targets
			inputs, targets, maskedIndices := prepareMLMInput(nn.InputSize)

			// Augment the input data
			augmentedInputs := predict.AugmentData(inputs)

			// MLM Forward pass and loss calculation using augmented input
			predictions := predict.PredictMaskedWords(nn, augmentedInputs)
			mlmLoss := predict.CalculateMLMLoss(nn, predictions, targets, maskedIndices)

			// Original task loss calculation using augmented input
			originalOutputs := pos.ForwardPassPos(nn, augmentedInputs)
			originalLoss := predict.CalculateOriginalLoss(originalOutputs, targets)

			// Combine losses
			totalLoss := originalLoss + mlmLoss

			// Backpropagation and weight update using augmented input
			predict.Backpropagate(nn, totalLoss, originalOutputs, learningRate, augmentedInputs, targets)
			predict.UpdateWeights(nn, learningRate)

			// Calculate accuracy (using original inputs for accuracy evaluation)
			posaccuracy, neraccuracy, phraseaccuracy, draccuracy = calc.CalculateAccuracy(nn, []tag.Tag{taggedSentence}, CreateTokenVocab([]tag.Tag{taggedSentence}), pos.CreatePosTagVocab([]tag.Tag{taggedSentence}), ner.CreateTagVocabNer([]tag.Tag{taggedSentence}), phrase.CreatePhraseTagVocab([]tag.Tag{taggedSentence}), dr.CreateDRTagVocab([]tag.Tag{taggedSentence}))
		}
	}

	return posaccuracy, neraccuracy, phraseaccuracy, draccuracy
}

// trainAndSaveModel trains a new neural network model, or loads an existing one if available.
// It then saves the trained model to a file.
func TrainAndSaveModel(trainingData *TrainingDataJSON, modeldirectory string) (*nnu.SimpleNN, error) {
	// Delete existing model file if it exists.
	if _, err := os.Stat("trained_model.gob"); err == nil {
		// If the file exists, remove it.
		if err := os.Remove("trained_model.gob"); err != nil {
			// If there's an error during removal, return an error.
			return nil, fmt.Errorf("error deleting model file: %w", err)
		}
	}

	// Load or train the neural network model.
	nn, err := LoadModelOrTrainNew(trainingData, modeldirectory)
	if err != nil {
		// If there's an error during loading or training, return an error.
		return nil, fmt.Errorf("error loading or training model: %w", err)
	}

	// Further training (if needed).
	epochs := 1000       // Adjust the number of epochs as needed.
	learningRate := 0.01 // Adjust the learning rate as needed.
	// Train the model using the training data.
	Train(trainingData.Sentences, epochs, learningRate, nn)

	// Save the trained model to a file.
	if err := gobs.SaveModelToGOB(nn, "trained_model.gob"); err != nil {
		// If there's an error during saving, return an error.
		return nil, fmt.Errorf("error saving model: %w", err)
	}
	// Print a message indicating that the model has been saved.
	fmt.Println("Model saved to trained_model.gob")
	// Return the trained neural network model and nil error.
	return nn, nil
}

func LoadModelOrTrainNew(trainingData *TrainingDataJSON, modeldirectory string) (*nnu.SimpleNN, error) {
	tokenVocab, _, _, _, _, _ := CreateVocab()
	nn, err := gobs.LoadModelFromGOB("trained_model.gob")
	if err != nil {
		fmt.Println("Error loading model, creating a new one:", err)
		inputSize := len(tokenVocab)
		hiddenSize := 5 // Adjust as needed
		outputSize := len(tokenVocab)
		outputWeights := make([][]float64, outputSize)
		for i := range outputWeights {
			outputWeights[i] = make([]float64, hiddenSize)
		}
		// Create a new neural network model
		nn = nn.NewSimpleNN(inputSize, hiddenSize, outputSize, outputWeights)
		// Load training data
		trainingData, err := LoadTrainingDataFromJSON(modeldirectory)
		if err != nil {
			fmt.Println("Error loading training data:", err)
		}
		// Train the network
		epochs := 1000       // Adjust as needed
		learningRate := 0.01 // Adjust as needed
		posaccuracy, neraccuracy, phraseaccuracy, draccuracy := Train(trainingData.Sentences, epochs, learningRate, nn)
		fmt.Printf("Final POS Accuracy: %.2f%%\n", posaccuracy*100)
		fmt.Printf("Final NER Accuracy: %.2f%%\n", neraccuracy*100)
		fmt.Printf("Final Phrase Accuracy: %.2f%%\n", phraseaccuracy*100)
		fmt.Printf("Final Dependency relation Accuracy: %.2f%%\n", draccuracy*100)
		// Save the newly trained model
		err = gobs.SaveModelToGOB(nn, "trained_model.gob")
		if err != nil {
			return nil, fmt.Errorf("error saving model: %w", err)
		}
		fmt.Println("New model saved to trained_model.gob")
	} else {
		fmt.Println("Loaded model from trained_model.gob")
	}
	return nn, nil
}

// Function to load training data from a JSON file
func LoadTrainingDataFromJSON(filePath string) (*TrainingDataJSON, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}

	data, err := ioutil.ReadAll(file)
	if err != nil {
		return nil, err
	}

	var trainingData TrainingDataJSON
	err = json.Unmarshal(data, &trainingData)
	if err != nil {
		return nil, err
	}
	file.Close()

	return &trainingData, nil
}

func CreateVocab() (map[string]int, map[string]int, map[string]int, map[string]int, map[string]int, *TrainingDataJSON) {
	trainingData, err := LoadTrainingDataFromJSON("data/training_data.json")
	if err != nil {
		fmt.Println("error loading training data: %w", err)
	}
	// Create vocabularies
	tokenVocab := CreateTokenVocab(trainingData.Sentences)
	posTagVocab := pos.CreatePosTagVocab(trainingData.Sentences)
	nerTagVocab := ner.CreateTagVocabNer(trainingData.Sentences)
	phraseTagVocab := phrase.CreatePhraseTagVocab(trainingData.Sentences)
	drTagVocab := dr.CreateDRTagVocab(trainingData.Sentences)

	return tokenVocab, posTagVocab, nerTagVocab, phraseTagVocab, drTagVocab, trainingData
}

func CreateTokenVocab(trainingData []tag.Tag) map[string]int {
	tokenVocab := make(map[string]int)
	tokenVocab["UNK"] = 0 // Add "UNK" token initially
	index := 1
	for _, sentence := range trainingData { // Iterate through tag.Tag slice
		for _, token := range sentence.Tokens {
			if _, ok := tokenVocab[token]; !ok {
				tokenVocab[token] = index
				index++
			}
		}
	}

	// If index exceeded vocabulary size
	if index > len(tokenVocab)-1 { // Dynamically determine vocabulary size
		// Handle unknown tokens
		tokenVocab["UNK"] = len(tokenVocab) // Add "UNK" token
		index = len(tokenVocab)             // Update index to reflect new vocabulary size

	}

	return tokenVocab
}

// prepareMLMInput prepares the input for Masked Language Modeling.
// It masks a percentage of the input tokens and creates the corresponding targets.
func prepareMLMInput(inputSize int) ([]float64, []float64, map[int]bool) {
	inputs := make([]float64, inputSize)
	targets := make([]float64, inputSize)
	maskedIndices := make(map[int]bool)

	// Initialize inputs with some sample values (replace with your actual data)
	for i := 0; i < inputSize; i++ {
		inputs[i] = float64(i + 1)
	}

	numTokensToMask := int(0.15 * float64(inputSize)) // Mask 15% of the tokens

	// Generate random indices to mask and store them in maskedIndices map
	for i := 0; i < numTokensToMask; i++ {
		randomIndex := rand.Intn(inputSize)
		maskedIndices[randomIndex] = true
		targets[randomIndex] = inputs[randomIndex] // Target is the original value
		inputs[randomIndex] = 0                    // Mask the input with 0
	}
	return inputs, targets, maskedIndices
}
