package crf_model

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"strings"

	"github.com/golangast/nlptagger/tagger/tag"
)

func Viterbi(model *CRFModel, sentenceFeatures []FeatureSet) ViterbiOutput {
	numWords := len(sentenceFeatures)
	numLabels := len(model.Labels)

	// Viterbi table: viterbi[i][j] is the score of the best sequence ending in label j at word i
	viterbi := make([][]float64, numWords)
	// Backpointers: backpointers[i][j] is the previous label that led to the best score at viterbi[i][j]
	backpointers := make([][]int, numWords)
	for i := 0; i < numWords; i++ {
		viterbi[i] = make([]float64, numLabels)
		backpointers[i] = make([]int, numLabels)
	}

	// Initialization: First word
	for j, label := range model.Labels {
		score := 0.0
		for featureName, featureValue := range sentenceFeatures[0].toMap() {
			if weight, ok := model.Weights[featureName+"="+featureValue]; ok {
				if w, ok := weight[label]; ok {
					score += w
				}
			}
		}
		viterbi[0][j] = score
		backpointers[0][j] = -1 // No previous state for the first word
	}

	// Recursion: Remaining words
	for i := 1; i < numWords; i++ {
		for j, currentLabel := range model.Labels {
			bestScore := math.Inf(-1)
			bestPrevious := -1

			for k, previousLabel := range model.Labels {
				// Transition score
				transitionScore := 0.0
				if weight, ok := model.Weights[previousLabel]; ok {
					if w, ok := weight[currentLabel]; ok {
						transitionScore = w
					}
				}

				// Feature score
				featureScore := 0.0
				for featureName, featureValue := range sentenceFeatures[i].toMap() {
					if weight, ok := model.Weights[featureName+"="+featureValue]; ok {
						if w, ok := weight[currentLabel]; ok {
							featureScore += w
						}
					}
				}

				score := viterbi[i-1][k] + transitionScore + featureScore

				if score > bestScore {
					bestScore = score
					bestPrevious = k
				}
			}
			viterbi[i][j] = bestScore
			backpointers[i][j] = bestPrevious
		}
	}

	// Termination: Find the best score in the last row
	bestFinalScore := math.Inf(-1)
	bestFinalLabel := -1
	for j, score := range viterbi[numWords-1] {
		if score > bestFinalScore {
			bestFinalScore = score
			bestFinalLabel = j
		}
	}

	// Backtracking: Get the best sequence of labels
	bestSequence := make([]string, numWords)
	currentLabel := bestFinalLabel
	for i := numWords - 1; i >= 0; i-- {
		bestSequence[i] = model.Labels[currentLabel]
		currentLabel = backpointers[i][currentLabel]
	}

	return ViterbiOutput{Labels: bestSequence, Score: bestFinalScore}
}

func Train(model *CRFModel, trainingData []TrainingExample) {
	learningRate := 0.1
	numIterations := 10

	for iter := 0; iter < numIterations; iter++ {
		fmt.Printf("Iteration %d\n", iter+1)

		for _, data := range trainingData {
			sentenceFeatures, correctLabels := data.WordExamples, data.WordExamples

			// 1. Calculate Gradients
			gradients := calculateGradients(model, sentenceFeatures, correctLabels)

			// 2. Update Weights
			for feature, labelMap := range gradients {
				for label, gradient := range labelMap {
					if _, ok := model.Weights[feature]; ok {
						if _, ok := model.Weights[feature][label]; ok {
							model.Weights[feature][label] -= learningRate * gradient
						}
					}
					if _, ok := model.Weights[label]; ok {
						feature2 := strings.Split(feature, "=")[0]
						if _, ok := model.Weights[label][feature2]; ok {
							model.Weights[label][feature2] -= learningRate * gradient
						}
					}
				}
			}
		}
	}
}

func calculateGradients(model *CRFModel, sentenceFeatures []WordExample, correctLabels []WordExample) map[string]map[string]float64 {
	numLabels := len(model.Labels)

	// Initialize gradients for each feature and label
	gradients := make(map[string]map[string]float64) //gradients := make(map[string]map[string]float64) // Initialize gradients for each feature and label
	for _, feature := range model.Features {
		gradients[feature] = make(map[string]float64) // Initialize inner map
		for _, label := range model.Labels {
			gradients[feature][label] = 0.0
		}
	}
	for _, label1 := range model.Labels {
		gradients[label1] = make(map[string]float64) // Initialize inner map
		for _, label2 := range model.Labels {
			gradients[label1][label2] = 0.0
		}
	}

	sentenceFeaturesToFeatures := make([]FeatureSet, len(sentenceFeatures))
	for i, example := range sentenceFeatures {
		sentenceFeaturesToFeatures[i] = FeatureSet{
			Word:   example.Features["word"],
			PosTag: example.Features["pos"],

			Prefix:  example.Features["prefix"],  // Add prefix
			Suffix:  example.Features["suffix"],  // Add suffix
			Bigram:  example.Features["bigram"],  // Add bigram
			Trigram: example.Features["trigram"], // Add trigram
			NerTag:  example.Features["ner"],
		}
	}

	// Run Viterbi to get the predicted label sequence
	viterbiOutput := Viterbi(model, sentenceFeaturesToFeatures)
	predictedLabels := viterbiOutput.Labels

	// Calculate the expected count for each feature and label (correct labels)
	expectedCountsCorrect := make(map[string]map[string]float64)
	for _, feature := range model.Features {
		expectedCountsCorrect[feature] = make(map[string]float64)
		for _, label := range model.Labels {
			expectedCountsCorrect[feature][label] = 0.0
		}
	}
	for _, label1 := range model.Labels {
		expectedCountsCorrect[label1] = make(map[string]float64)
		for _, label2 := range model.Labels {
			expectedCountsCorrect[label1][label2] = 0.0
		}
	}

	//Correct labels
	for i, label := range correctLabels {
		for featureName, featureValue := range label.Features {
			// Initialize the inner map if it doesn't exist
			if _, ok := expectedCountsCorrect[featureName+"="+featureValue]; !ok {
				expectedCountsCorrect[featureName+"="+featureValue] = make(map[string]float64)
			}
			expectedCountsCorrect[featureName+"="+featureValue][label.Label] += 1.0
		}
		//Fix: numWords -> len(correctLabels)
		if i < len(correctLabels)-1 {
			//Initialize if does not exist
			if _, ok := expectedCountsCorrect[label.Label]; !ok {
				expectedCountsCorrect[label.Label] = make(map[string]float64)
			}
			expectedCountsCorrect[label.Label][correctLabels[i+1].Label] += 1.0
		}
	}

	// Calculate the expected count for each feature and label (predicted labels)
	expectedCountsPredicted := make(map[string]map[string]float64)
	for _, feature := range model.Features {
		expectedCountsPredicted[feature] = make(map[string]float64)
		for _, label := range model.Labels {
			expectedCountsPredicted[feature][label] = 0.0
		}
	}
	for _, label1 := range model.Labels {
		expectedCountsPredicted[label1] = make(map[string]float64)
		for _, label2 := range model.Labels {
			expectedCountsPredicted[label1][label2] = 0.0
		}
	}

	// Calculate Viterbi values for each word
	viterbiValues := make([][]float64, len(sentenceFeatures))
	for i := range viterbiValues {
		viterbiValues[i] = make([]float64, numLabels)
		for j, label := range model.Labels {
			score := 0.0
			for featureName, featureValue := range sentenceFeatures[i].Features {
				if weight, ok := model.Weights[featureName+"="+featureValue]; ok {
					if w, ok := weight[label]; ok {
						score += w
					}
				}
			}
			viterbiValues[i][j] = score
		}
	}

	// Calculate predicted transitions
	for i, label := range predictedLabels {
		for featureName, featureValue := range sentenceFeatures[i].Features {
			if _, ok := expectedCountsPredicted[featureName+"="+featureValue]; !ok {
				expectedCountsPredicted[featureName+"="+featureValue] = make(map[string]float64)
			}
			expectedCountsPredicted[featureName+"="+featureValue][label] += 1.0
		}
		//Fix: numWords -> len(predictedLabels)
		if i < len(predictedLabels)-1 {
			//Initialize if does not exist
			if _, ok := expectedCountsPredicted[label]; !ok {
				expectedCountsPredicted[label] = make(map[string]float64)
			}
			// add transition values:
			expectedCountsPredicted[label][predictedLabels[i+1]] += 1.0
		}
	}

	// Calculate the gradients:
	// Feature gradients
	for feature, labelMap := range expectedCountsPredicted {
		//Initialize if it does not exist
		if _, ok := gradients[feature]; !ok {
			gradients[feature] = make(map[string]float64)
		}
		for label := range labelMap {
			if _, ok := gradients[feature]; !ok {
				gradients[feature] = make(map[string]float64)
			}
			gradients[feature][label] = 0.0
			if _, ok := expectedCountsCorrect[feature][label]; ok {
				gradients[feature][label] = expectedCountsCorrect[feature][label] - expectedCountsPredicted[feature][label]
			} else {
				gradients[feature][label] = 0 - expectedCountsPredicted[feature][label]
			}
		}
	}
	// Transition gradients
	for label1, labelMap := range expectedCountsPredicted {
		//Initialize if it does not exist
		if _, ok := gradients[label1]; !ok {
			gradients[label1] = make(map[string]float64)
		}
		for label2 := range labelMap {
			if _, ok := gradients[label1]; !ok {
				gradients[label1] = make(map[string]float64)
			}
			gradients[label1][label2] = 0.0
			if _, ok := expectedCountsCorrect[label1][label2]; ok {
				gradients[label1][label2] = expectedCountsCorrect[label1][label2] - expectedCountsPredicted[label1][label2]
			} else {
				gradients[label1][label2] = 0 - expectedCountsPredicted[label1][label2]
			}
		}
	}

	return gradients
}

// ExtractFeatures extracts features for a sentence and return the WordExamples
func ExtractFeatures(t tag.Tag) []WordExample {
	wordExamples := make([]WordExample, len(t.Tokens))
	for i, token := range t.Tokens {
		wordExamples[i] = WordExample{
			Features: map[string]string{
				"word": token,
				"pos":  t.PosTag[i],
				"ner":  t.NerTag[i],
			},
		}

		// Add prefix and suffix features
		if len(token) >= 3 {
			wordExamples[i].Features["prefix"] = token[:3]
			wordExamples[i].Features["suffix"] = token[len(token)-3:]
		} else {
			wordExamples[i].Features["prefix"] = token
			wordExamples[i].Features["suffix"] = token
		}

		// Add bigram feature
		if i > 0 {
			wordExamples[i].Features["bigram"] = t.Tokens[i-1] + "_" + token
		} else {
			wordExamples[i].Features["bigram"] = "<START>_" + token
		}

		// Add trigram feature
		if i > 1 {
			wordExamples[i].Features["trigram"] = t.Tokens[i-2] + "_" + t.Tokens[i-1] + "_" + token
		} else if i > 0 {
			wordExamples[i].Features["trigram"] = "<START>_" + t.Tokens[i-1] + "_" + token
		} else {
			wordExamples[i].Features["trigram"] = "<START>_<START>_" + token
		}
	}
	return wordExamples
}
func (fs FeatureSet) toMap() map[string]string {
	m := make(map[string]string)
	m["word"] = fs.Word
	m["pos"] = fs.PosTag
	m["ner"] = fs.NerTag
	m["prefix"] = fs.Prefix
	m["suffix"] = fs.Suffix
	m["bigram"] = fs.Bigram
	m["trigram"] = fs.Trigram

	return m
}

func loadTrainingData(filename string) ([]TrainingExample, error) {
	// Read the file
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("error reading file: %w", err)
	}

	// Unmarshal the JSON data
	var trainingData []TrainingExample
	err = json.Unmarshal(data, &trainingData)
	if err != nil {
		return nil, fmt.Errorf("error unmarshaling JSON: %w", err)
	}

	return trainingData, nil
}

// FeatureSet represents the features for a word
type FeatureSet struct {
	Word    string
	PosTag  string
	NerTag  string
	Prefix  string // New: Word prefix
	Suffix  string // New: Word suffix
	Bigram  string // New: Bigram
	Trigram string // New: Trigram

}

// getIndex return the index of the label in the list of labels
func getIndex(labels []string, label string) int {
	for i, l := range labels {
		if l == label {
			return i
		}
	}
	return -1
}

func CrfTrain() {

	trainingDatas, err := loadTrainingData("datas/crftrainingdata.json") // Replace with your file name
	if err != nil {
		fmt.Println("Error loading training data:", err)
		return
	}

	// Define features and labels
	features := []string{"word", "pos", "ner", "prefix", "suffix", "bigram", "trigram"} // Add new features
	labels := []string{"Arg0", "Arg1", "OTHER", "ArgM-LOC", "ArgM-MNR", "Arg2"}

	// Create the CRF model

	model := NewCRFModel(features, labels)

	// Train the model
	Train(model, trainingDatas)

	// Example sentence (tag.Tag)
	var t tag.Tag
	t.Tokens = []string{"The", "system", "handles", "user", "login"}
	t.PosTag = []string{"DT", "NN", "VBZ", "NN", "NN"}
	t.NerTag = []string{"OTHER", "OBJECT_TYPE", "COMMAND", "OBJECT_TYPE", "OBJECT_TYPE"}
	//Extract features:
	//Declare sentenceFeaturesWordExample in the correct scope

	sentenceFeaturesWordExample := ExtractFeatures(t)

	//Convert WordExample to FeaturesSet
	sentenceFeatures := make([]FeatureSet, len(sentenceFeaturesWordExample))
	for i, example := range sentenceFeaturesWordExample {
		sentenceFeatures[i] = FeatureSet{
			Word:    example.Features["word"],
			PosTag:  example.Features["pos"],
			Prefix:  example.Features["prefix"],  // Add prefix
			Suffix:  example.Features["suffix"],  // Add suffix
			Bigram:  example.Features["bigram"],  // Add bigram
			Trigram: example.Features["trigram"], // Add trigram
			NerTag:  example.Features["ner"],
		}
	}

	//Get dependencies

	// Inference (Viterbi)
	// The correct input for Viterbi is: sentenceFeatures
	viterbiOutput := Viterbi(model, sentenceFeatures)

	fmt.Println("Predicted Labels:", viterbiOutput.Labels)
	fmt.Println("Score:", viterbiOutput.Score)
}
