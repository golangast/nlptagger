package simplenn

import (
	"encoding/gob"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"strings"

	"github.com/golangast/nlptagger/nn/nnu"
	"github.com/golangast/nlptagger/tagger/tag"
)

type SimpleNN struct {
	InputSize      int
	HiddenSize     int
	OutputSize     int
	WeightsIH      [][]float64
	WeightsHO      [][]float64
	TokenVocab     map[string]int
	PosTagVocab    map[string]int
	NerTagVocab    map[string]int
	PhraseTagVocab map[string]int
	DependencyTag  map[string]int
}

// Activation function (sigmoid)
func (nn SimpleNN) Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// Forward pass https://wiki.pathmind.com/neural-network
func (nn *SimpleNN) ForwardPass(inputs []float64) []float64 {
	// Create a slice to store the activations of the hidden layer.
	hidden := make([]float64, nn.HiddenSize)
	// Iterate over each neuron in the hidden layer.
	for i := range hidden {
		// Initialize the sum of weighted inputs for the current neuron.
		sum := 0.0
		// Iterate over each input neuron.
		for j := range inputs {
			// Calculate the weighted input and add it to the sum.
			sum += nn.WeightsIH[i][j] * inputs[j]
		}
		// Apply the sigmoid activation function to the sum and store the result in the hidden layer.
		hidden[i] = nn.Sigmoid(sum)
	}
	// Create a slice to store the activations of the output layer.
	output := make([]float64, nn.OutputSize)
	// Iterate over each neuron in the output layer.
	for i := range output {
		// Initialize the sum of weighted inputs for the current neuron.
		sum := 0.0
		// Iterate over each neuron in the hidden layer.
		for j := range hidden {
			// Calculate the weighted input and add it to the sum.
			sum += nn.WeightsHO[i][j] * hidden[j]
		}
		// Apply the sigmoid activation function to the sum and store the result in the output layer.
		output[i] = nn.Sigmoid(sum)
	}
	// Return the activations of the output layer.
	return output
}
func (nn *SimpleNN) CalculateError(targetTag string, outputs []float64, trainingData []tag.Tag) ([]float64, map[string]int, map[string]int, map[string]int, map[string]int) {
	// Create a slice to store the target output values.
	targetOutput := make([]float64, nn.OutputSize)
	// Create a vocabulary of POS tags from the training data.
	posTagVocab := CreatePosTagVocab(trainingData)
	// Get the index of the target POS tag in the vocabulary.
	targetTagIndex, ok := posTagVocab[targetTag]
	if ok {
		// Set the corresponding element in the target output to 1.
		targetOutput[targetTagIndex] = 1
	}

	nerTagVocab := CreateNerTagVocab(trainingData) // Use NER vocab
	targetNerTagIndex, ok := nerTagVocab[targetTag]
	// If the target POS tag is found in the vocabulary...
	if ok {
		// Set the corresponding element in the target output to 1.
		targetOutput[targetNerTagIndex] = 1
	}
	phraseTagVocab := CreatePhraseTagVocab(trainingData) // Use NER vocab
	targetPhraseTagIndex, ok := phraseTagVocab[targetTag]
	// If the target POS tag is found in the vocabulary...
	if ok {
		// Set the corresponding element in the target output to 1.
		targetOutput[targetPhraseTagIndex] = 1
	}
	drTagVocab := CreateDRTagVocab(trainingData) // Use NER vocab
	targetDRTagIndex, ok := drTagVocab[targetTag]
	// If the target POS tag is found in the vocabulary...
	if ok {
		// Set the corresponding element in the target output to 1.
		targetOutput[targetDRTagIndex] = 1
	}
	// Create a slice to store the errors for each output neuron.
	errors := make([]float64, nn.OutputSize)
	// Calculate the error for each output neuron.
	for i := range errors {
		errors[i] = targetOutput[i] - outputs[i]
	}
	return errors, posTagVocab, nerTagVocab, phraseTagVocab, drTagVocab
}

func (nn *SimpleNN) Backpropagate(errors []float64, outputs []float64, learningRate float64, inputs []float64) {
	// 1. Output layer weight updates
	for i := 0; i < nn.OutputSize; i++ {
		gradient := errors[i] * outputs[i] * (1 - outputs[i]) // Sigmoid derivative
		for j := 0; j < nn.HiddenSize; j++ {
			nn.WeightsHO[i][j] += learningRate * gradient * outputs[j] //Hidden Layer is input to Output Layer
		}
	}

	// 2. Hidden layer weight updates
	hiddenErrors := make([]float64, nn.HiddenSize)
	for i := 0; i < nn.HiddenSize; i++ {
		errorSum := 0.0
		for j := 0; j < nn.OutputSize; j++ {
			errorSum += errors[j] * nn.WeightsHO[j][i]
		}
		hiddenErrors[i] = errorSum * outputs[i] * (1 - outputs[i]) // Sigmoid derivative
	}
	for i := 0; i < nn.HiddenSize; i++ {
		for j := 0; j < nn.InputSize; j++ {
			nn.WeightsIH[i][j] += learningRate * hiddenErrors[i] * inputs[j] //Input Layer is input to Hidden Layer
		}
	}
}

func CreatePosTagVocab(trainingData []tag.Tag) map[string]int {
	posTagVocab := make(map[string]int)
	index := 0

	for _, taggedSentence := range trainingData {
		for _, posTag := range taggedSentence.PosTag {
			if _, ok := posTagVocab[posTag]; !ok { // Check if POS tag is already in the vocabulary
				posTagVocab[posTag] = index
				index++
			}
		}
	}

	return posTagVocab
}
func CreateNerTagVocab(trainingData []tag.Tag) map[string]int {
	nerTagVocab := make(map[string]int)
	index := 0

	for _, taggedSentence := range trainingData {
		for _, nerTag := range taggedSentence.NerTag {
			if _, ok := nerTagVocab[nerTag]; !ok { // Check if POS tag is already in the vocabulary
				nerTagVocab[nerTag] = index
				index++
			}
		}
	}

	return nerTagVocab
}
func CreatePhraseTagVocab(trainingData []tag.Tag) map[string]int {
	phraseTagVocab := make(map[string]int)
	index := 0

	for _, taggedSentence := range trainingData {
		for _, phraseTag := range taggedSentence.PhraseTag {
			if _, ok := phraseTagVocab[phraseTag]; !ok { // Check if POS tag is already in the vocabulary
				phraseTagVocab[phraseTag] = index
				index++
			}
		}
	}

	return phraseTagVocab
}
func CreateDRTagVocab(tags []tag.Tag) map[string]int {
	drTagVocab := make(map[string]int)
	index := 0

	for _, tag := range tags {
		//fmt.Printf("Sentence %d: Dependencies: %+v\n", i, tag.DepRelationsTag) // Print the Dependencies field

		for _, drt := range tag.Dependencies {
			//fmt.Println(j, drt.Head, drt.Dep) // Print relation details
			if _, ok := drTagVocab[drt.Dep]; !ok {
				drTagVocab[drt.Dep] = index
				index++
			}
		}
	}
	return drTagVocab
}

func (nn *SimpleNN) NewSimpleNN(inputSize, hiddenSize, outputSize int) *SimpleNN {
	nnn := &SimpleNN{
		InputSize:  inputSize,
		HiddenSize: hiddenSize,
		OutputSize: outputSize,
	}
	nnn.WeightsIH = make([][]float64, hiddenSize)
	for i := range nnn.WeightsIH {
		nnn.WeightsIH[i] = make([]float64, inputSize)
		for j := range nnn.WeightsIH[i] {
			nnn.WeightsIH[i][j] = rand.Float64()*2 - 1 // Initialize with random weights
		}
	}
	nnn.WeightsHO = make([][]float64, outputSize)
	for i := range nnn.WeightsHO {
		nnn.WeightsHO[i] = make([]float64, hiddenSize)
		for j := range nnn.WeightsHO[i] {
			nnn.WeightsHO[i][j] = rand.Float64()*2 - 1
		}
	}
	return nnn
}
func LoadModelFromGOB(filePath string) (*SimpleNN, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}

	decoder := gob.NewDecoder(file)
	var model SimpleNN
	err = decoder.Decode(&model)
	if err != nil {
		return nil, err
	}
	file.Close()
	return &model, nil
}

func (nn *SimpleNN) LoadModelOrTrainNew(trainingData *nnu.TrainingDataJSON) (*SimpleNN, error) {
	tokenVocab, posTagVocab, _, _, _, _ := CreateVocab()
	nn, err := LoadModelFromGOB("trained_model.gob")
	if err != nil {
		fmt.Println("Error loading model, creating a new one:", err)
		inputSize := len(tokenVocab)
		hiddenSize := 5 // Adjust as needed
		outputSize := len(posTagVocab)
		nn = nn.NewSimpleNN(inputSize, hiddenSize, outputSize)
		// Load training data
		trainingData, err := nnu.LoadTrainingDataFromJSON("data/training_data.json")
		if err != nil {
			fmt.Println("Error loading training data:", err)
		}
		// Train the network
		epochs := 100       // Adjust as needed
		learningRate := 0.1 // Adjust as needed
		posaccuracy, neraccuracy, phraseaccuracy, draccuracy := Train(trainingData.Sentences, epochs, learningRate, nn)
		fmt.Printf("Final POS Accuracy: %.2f%%\n", posaccuracy*100)
		fmt.Printf("Final NER Accuracy: %.2f%%\n", neraccuracy*100)
		fmt.Printf("Final Phrase Accuracy: %.2f%%\n", phraseaccuracy*100)
		fmt.Printf("Final Dependency relation Accuracy: %.2f%%\n", draccuracy*100)
		// Save the newly trained model
		err = SaveModelToGOB(nn, "trained_model.gob")
		if err != nil {
			return nil, fmt.Errorf("error saving model: %w", err)
		}
		fmt.Println("New model saved to trained_model.gob")
	} else {
		fmt.Println("Loaded model from trained_model.gob")
	}
	return nn, nil
}

func (nn *SimpleNN) PredictTags(sentence string) ([]string, []string, []string, []string) {
	tokenVocab, posTagVocab, nerTagVocab, phraseTagVocab, drTagVocab, _ := CreateVocab()
	// Tokenize the sentence into individual words.

	tokens := strings.Fields(sentence)
	// Create a slice to store the predicted POS tags.
	var predictedPosTags, predictedNerTags, predictedPhraseTags, predictedDRTags []string
	// Iterate over each token in the sentence.
	for _, token := range tokens {
		// Get the index of the token in the vocabulary.
		tokenIndex, ok := tokenVocab[token]
		// If the token is not in the vocabulary...
		if !ok {
			// Print a message indicating the token was not found.
			fmt.Printf("Token '%s' not found in vocabulary\n", token)
			// Assign "UNK" (unknown) as the POS tag for this token.
			predictedPosTags = append(predictedPosTags, "UNK")
			predictedNerTags = append(predictedNerTags, "O")         // Default NER tag for unknown tokens
			predictedPhraseTags = append(predictedPhraseTags, "UNK") // Default NER tag for unknown tokens
			predictedDRTags = append(predictedDRTags, "UNK")         // Default NER tag for unknown tokens
			// Move on to the next token.
			continue
		}

		// Create the input vector for the neural network.
		inputs := make([]float64, nn.InputSize)
		inputs[tokenIndex] = 1 // Set the element corresponding to the token index to 1.

		// Get the predicted output from the neural network.
		predictedOutput := nn.ForwardPass(inputs)
		// Get the index of the predicted POS tag.
		predictedTagIndex := nnu.MaxIndex(predictedOutput)
		// Get the actual POS tag string using the predicted index.
		predictedTag, ok := nnu.IndexToPosTag(posTagVocab, predictedTagIndex)
		if !ok {
			// Print an error message.
			fmt.Printf("Tag index %d not found in vocabulary\n", predictedTagIndex)
			// Append "UNK" to the predicted tags.
			predictedPosTags = append(predictedPosTags, "UNK")
			// Continue to the next token.
			continue
		}
		// Append the predicted tag to the list of predicted tags.
		predictedPosTags = append(predictedPosTags, predictedTag)

		// Get the index of the predicted NER tag.
		predictedNerTagIndex := nnu.MaxIndex(predictedOutput)
		// Get the actual NER tag string using the predicted index.
		predictedTagNer, ok := nnu.IndexToNerTag(nerTagVocab, predictedNerTagIndex)
		// If the predicted tag index is not found in the vocabulary...
		if !ok {
			// Print an error message.
			fmt.Printf("NER tag index %d not found in vocabulary\n", predictedNerTagIndex)
			// Append "O" (outside of any named entity) to the predicted NER tags.
			predictedNerTags = append(predictedNerTags, "O")
			// Continue to the next token.
			continue
		}
		// Append the predicted tag to the list of predicted NER tags.
		predictedNerTags = append(predictedNerTags, predictedTagNer)

		var predictedPhraseTagIndex = -1 // Initialize to an invalid index
		if len(predictedOutput) > 0 {    // Check if predictedOutput is not empty
			predictedPhraseTagIndex = nnu.MaxIndex(predictedOutput) // Get the maximum index from predictedOutput
		}

		if predictedPhraseTagIndex != -1 { //check for valid index
			predictedTagPhrase, ok := nnu.IndexToPhraseTag(phraseTagVocab, predictedPhraseTagIndex)
			if !ok {
				fmt.Printf("Phrase tag index %d not found in vocabulary\n", predictedPhraseTagIndex)
				predictedPhraseTags = append(predictedPhraseTags, "O")
				continue
			}
			predictedPhraseTags = append(predictedPhraseTags, predictedTagPhrase)

		} else { //handle invalid index
			predictedPhraseTags = append(predictedPhraseTags, "O") // Default to "O" if no valid index
			fmt.Printf("predictedOutput is empty or length 0\n")
		}
		// Dependency Relation Tag Prediction (Similar fix as Phrase Tag)
		predictedDRTagIndex := -1 // Initialize to an invalid index
		if len(predictedOutput) > 0 {
			predictedDRTagIndex = nnu.MaxIndex(predictedOutput)
		}

		if predictedDRTagIndex != -1 {
			predictedTagDR, ok := nnu.IndexToNerTag(drTagVocab, predictedDRTagIndex)

			if !ok {
				fmt.Printf("DR tag index %d not found in vocabulary\n", predictedDRTagIndex)
				predictedDRTags = append(predictedDRTags, "O")
				continue
			}
			predictedDRTags = append(predictedDRTags, predictedTagDR)
		} else {
			predictedDRTags = append(predictedDRTags, "O") // Default to "O" if no valid index
			fmt.Printf("predictedOutput is empty or length 0\n")
		}

	}
	// Return the list of predicted POS and NER tags.
	return predictedPosTags, predictedNerTags, predictedPhraseTags, predictedDRTags
}

// trainAndSaveModel trains a new neural network model, or loads an existing one if available.
// It then saves the trained model to a file.
func (nn *SimpleNN) TrainAndSaveModel(trainingData *nnu.TrainingDataJSON) (*SimpleNN, error) {
	// Delete existing model file if it exists.
	if _, err := os.Stat("trained_model.gob"); err == nil {
		// If the file exists, remove it.
		if err := os.Remove("trained_model.gob"); err != nil {
			// If there's an error during removal, return an error.
			return nil, fmt.Errorf("error deleting model file: %w", err)
		}
	}

	// Load or train the neural network model.
	nn, err := nn.LoadModelOrTrainNew(trainingData)
	if err != nil {
		// If there's an error during loading or training, return an error.
		return nil, fmt.Errorf("error loading or training model: %w", err)
	}

	// Further training (if needed).
	epochs := 100       // Adjust the number of epochs as needed.
	learningRate := 0.1 // Adjust the learning rate as needed.
	// Train the model using the training data.
	Train(trainingData.Sentences, epochs, learningRate, nn)

	// Save the trained model to a file.
	if err := SaveModelToGOB(nn, "trained_model.gob"); err != nil {
		// If there's an error during saving, return an error.
		return nil, fmt.Errorf("error saving model: %w", err)
	}
	// Print a message indicating that the model has been saved.
	fmt.Println("Model saved to trained_model.gob")
	// Return the trained neural network model and nil error.
	return nn, nil
}

func CreateVocab() (map[string]int, map[string]int, map[string]int, map[string]int, map[string]int, *nnu.TrainingDataJSON) {
	trainingData, err := nnu.LoadTrainingDataFromJSON("data/training_data.json")
	if err != nil {
		fmt.Println("error loading training data: %w", err)
	}
	// Create vocabularies
	tokenVocab := nnu.CreateTokenVocab(trainingData.Sentences)
	posTagVocab := CreatePosTagVocab(trainingData.Sentences)
	nerTagVocab := CreateNerTagVocab(trainingData.Sentences)
	phraseTagVocab := CreatePhraseTagVocab(trainingData.Sentences)
	drTagVocab := CreateDRTagVocab(trainingData.Sentences)

	return tokenVocab, posTagVocab, nerTagVocab, phraseTagVocab, drTagVocab, trainingData
}

// Training function
func Train(trainingData []tag.Tag, epochs int, learningRate float64, nn *SimpleNN) (float64, float64, float64, float64) {
	var posaccuracy float64
	var neraccuracy float64
	var phraseaccuracy float64
	var draccuracy float64
	for epoch := 0; epoch < epochs; epoch++ {
		for _, taggedSentence := range trainingData {
			for i := range taggedSentence.Tokens {
				token := taggedSentence.Tokens[i]
				targetTag := taggedSentence.PosTag[i]

				// Convert token to one-hot encoded input
				inputs := make([]float64, nn.InputSize)
				tokenVocab := nnu.CreateTokenVocab(trainingData)
				tokenIndex, ok := tokenVocab[token]
				if ok {
					if _, ok := tokenVocab[token]; !ok {
						fmt.Printf("Token '%s' not found in vocabulary!\n", token)
					}
					inputs[tokenIndex] = 1
				}

				// Forward pass
				outputs := nn.ForwardPass(inputs)

				errors, posTagVocab, nerTagVocab, phraseTagVocab, drTagVocab := nn.CalculateError(targetTag, outputs, trainingData)

				nn.Backpropagate(errors, outputs, learningRate, inputs)

				// Calculate accuracy for this epoch
				posaccuracy, neraccuracy, phraseaccuracy, draccuracy = calculateAccuracy(nn, trainingData, tokenVocab, posTagVocab, nerTagVocab, phraseTagVocab, drTagVocab)
				// fmt.Printf("Epoch %d: POS Accuracy = %.2f%%\n", epoch+1, posaccuracy*100)
				// fmt.Printf("Epoch %d: NER Accuracy = %.2f%%\n", epoch+1, neraccuracy*100)
				// fmt.Printf("Epoch %d: PHRASE Accuracy = %.2f%%\n", epoch+1, phraseaccuracy*100)
				//fmt.Printf("Epoch %d: Dependency Accuracy = %.2f%%\n", epoch+1, draccuracy*100)
			}

		}

	}
	return posaccuracy, neraccuracy, phraseaccuracy, draccuracy
}

func SaveModelToGOB(model *SimpleNN, filePath string) error {
	file, err := os.Create(filePath)

	if err != nil {
		return err
	}

	encoder := gob.NewEncoder(file)
	err = encoder.Encode(model)
	if err != nil {
		return err
	}
	file.Close()

	return nil
}

// Function to make a prediction
func predict(nn *SimpleNN, inputs []float64, posTagVocab map[string]int, nerTagVocab map[string]int, phraseTagVocab map[string]int, drTagVocab map[string]int) (string, string, string, string) {
	predictedOutput := nn.ForwardPass(inputs)
	predictedTagIndex := nnu.MaxIndex(predictedOutput)
	predictedPosTag, _ := nnu.IndexToPosTag(posTagVocab, predictedTagIndex)
	predictedNerTag, _ := nnu.IndexToNerTag(nerTagVocab, predictedTagIndex)
	predictedPhraseTag, _ := nnu.IndexToPhraseTag(phraseTagVocab, predictedTagIndex)
	predictedDRTag, _ := nnu.IndexToDRTag(drTagVocab, predictedTagIndex)
	return predictedPosTag, predictedNerTag, predictedPhraseTag, predictedDRTag
}

// Function to calculate accuracy
func calculateAccuracy(nn *SimpleNN, trainingData []tag.Tag, tokenVocab map[string]int, posTagVocab map[string]int, nerTagVocab map[string]int, phraseTagVocab map[string]int, drTagVocab map[string]int) (float64, float64, float64, float64) {
	poscorrectPredictions := 0
	nercorrectPredictions := 0
	phrasecorrectPredictions := 0
	drcorrectPredictions := 0
	postotalPredictions := 0
	nertotalPredictions := 0
	phrasetotalPredictions := 0
	drtotalPredictions := 0

	for _, taggedSentence := range trainingData {
		for i := range taggedSentence.Tokens {
			inputs := make([]float64, nn.InputSize)
			tokenIndex, ok := tokenVocab[taggedSentence.Tokens[i]]
			if ok {
				inputs[tokenIndex] = 1
			} else {
				inputs[tokenVocab["UNK"]] = 1 // Handle unknown tokens
			}

			predictedPosTag, predictedNerTag, predictedPhraseTag, predictedDRTag := predict(nn, inputs, posTagVocab, nerTagVocab, phraseTagVocab, drTagVocab)

			if predictedPosTag == taggedSentence.PosTag[i] {
				poscorrectPredictions++
			}
			if predictedNerTag == taggedSentence.NerTag[i] {
				nercorrectPredictions++
			}
			if predictedPhraseTag == taggedSentence.PhraseTag[i] {
				phrasecorrectPredictions++
			}
			if len(taggedSentence.Dependencies) > i { // Check if Dependency slice has an element at index i
				if predictedDRTag == taggedSentence.Dependencies[i].Dep {
					drcorrectPredictions++
				}
			} else {
				inputs[drTagVocab["UNK"]] = 1 // Handle unknown tokens
				//fmt.Printf("Warning: Dependency tag missing for token %d in sentence: %v\n", i, taggedSentence.Tokens)
				// Handle the missing tag appropriately.
				// Options:
				// 1. Skip the token for DR accuracy calculation (current behavior).
				// 2. Count it as an incorrect prediction.
				// 3. Use a default value for comparison.

			}
			postotalPredictions++
			nertotalPredictions++
			phrasetotalPredictions++
			drtotalPredictions++
		}
	}

	return float64(poscorrectPredictions) / float64(postotalPredictions), float64(nercorrectPredictions) / float64(nertotalPredictions), float64(phrasecorrectPredictions) / float64(phrasetotalPredictions), float64(drcorrectPredictions) / float64(drtotalPredictions)
}
