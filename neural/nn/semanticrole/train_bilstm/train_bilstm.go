// Package train_bilstm provides functions for training a BiLSTM model for Semantic Role Labeling.

package train_bilstm

import (
	"encoding/gob"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"

	"github.com/zendrulat/nlptagger/neural/nn/semanticrole"
	"github.com/zendrulat/nlptagger/neural/nn/semanticrole/bilstm_model"
	"github.com/zendrulat/nlptagger/neural/nnu/word2vec"
)

func LoadRoleData(filePath string) ([]semanticrole.SentenceRoleData, error) {
	return semanticrole.LoadRoleData(filePath)
}
func createRoleMap(data []semanticrole.SentenceRoleData) map[string]int {
	return semanticrole.CreateRoleMap(data)
}

func crossEntropyLoss(predictedProbabilities []float64, trueRoleID int) float64 {
	loss := 0.0
	for i, prob := range predictedProbabilities {
		if i == trueRoleID {
			loss -= math.Log(prob)
		}
	}
	return loss
}

// Hyperparameters
const (
	hiddenSize   = 128
	learningRate = 0.1
	epochs       = 10
	maxGrad      = 5.0
)

// Placeholder file paths
const (
	word2vecModelPath = "./gob_models/trained_model.gob"
	trainingDataPath  = "./trainingdata/roledata/training_data.json" // Assumed format: "token|ROLE token|ROLE ..." per line
	bilstmModelPath   = "./gob_models/bilstm_model.gob"
	roleMapPath       = "./gob_models/role_map.gob"
)

func Train_bilstm() {
	// 2. Load pre-trained word2vec model
	word2vecModel, err := word2vec.LoadModel(word2vecModelPath)
	if err != nil {
		fmt.Println("Error loading word2vec model: ", err)
	}
	vocabularySize := len(word2vecModel.Vocabulary)

	// 3. Load SRL training data
	trainingData, err := LoadRoleData(trainingDataPath)
	if err != nil {
		fmt.Println("Error loading training data: ", err)
		return
	}

	// 4. Create role map
	roleMap := createRoleMap(trainingData)
	// fmt.Printf("Role Map: %v\n", roleMap)
	model := bilstm_model.NewBiLSTMModel(vocabularySize, hiddenSize, len(roleMap))
	// Initialize output layer
	model.InitializeOutputLayer(hiddenSize)

	// Training loop with SentenceRoleData
	for epoch := 0; epoch < epochs; epoch++ { // start of epoch

		// Shuffle training data
		rand.Shuffle(len(trainingData), func(i, j int) {
			trainingData[i], trainingData[j] = trainingData[j], trainingData[i] // Swap elements
		})

		// Iterate through sentences
		for _, taggedSentence := range trainingData {
			// Skip sentences with no tokens
			if len(taggedSentence.Tokens) == 0 {
				fmt.Println("Skipping sentence with no tokens.")
				continue
			}

			// Convert sentence to input features
			embeddings := make([][]float64, len(taggedSentence.Tokens))
			roleIDs := make([]int, len(taggedSentence.Tokens))
			for i := range taggedSentence.Tokens {
				if i >= len(taggedSentence.Tokens) {
					fmt.Printf("Invalid index i=%d for taggedSentence.Tokens of length %d. Skipping.\n", i, len(taggedSentence.Tokens)) // Robust check to ensure i is a valid index
					continue
				}
				token := taggedSentence.Tokens[i] // Access token only after checking index
				embedding, ok := word2vecModel.WordVectors[word2vecModel.Vocabulary[token.Token]]
				if !ok {
					embedding = make([]float64, len(word2vecModel.WordVectors[word2vecModel.Vocabulary[word2vec.UNKToken]]))
					copy(embedding, word2vecModel.WordVectors[word2vecModel.Vocabulary[word2vec.UNKToken]])
				}

				embeddings[i] = embedding

				roleID, ok := roleMap[token.Role]
				if !ok {
					roleID = 0 // Assign role ID 0 for unknown roles
				}
				roleIDs[i] = roleID
				// if i < 5 {
				// 	fmt.Printf("  Token: %s, Role: %s, Role ID: %d\n", taggedSentence.Tokens[i].Token, taggedSentence.Tokens[i].Role, roleID)
				// }
			}

			tokenIDs := make([]int, len(taggedSentence.Tokens))
			for i, token := range taggedSentence.Tokens {
				tokenIDs[i] = word2vecModel.Vocabulary[token.Token]
			}
			// Forward pass
			probabilities := model.ForwardAndCalculateProbabilities(tokenIDs)

			// Calculate loss
			var loss float64 = 0.0
			for tokenIndex, prob := range probabilities {
				loss += crossEntropyLoss(prob, roleIDs[tokenIndex])
			}
			// Calculate gradients and update weights (placeholder)
			if len(roleIDs) > 0 {
				model.Backpropagate(probabilities, roleIDs, tokenIDs)
				model.UpdateWeights(learningRate)
			} else {
				fmt.Println("Skipping backpropagation for sentence with no roles.")
			}
			model.ClipGradients(maxGrad)

			// Print loss for this batch (for debugging)
			// if loss > 0 {
			// 	fmt.Printf("Loss: %f\n", loss)
			// }
		}
	}

	// 8. Save trained BiLSTMModel and role map

	saveModel(model, bilstmModelPath)
	saveRoleMap(roleMap, roleMapPath)

}

func saveModel(model *bilstm_model.BiLSTMModel, path string) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("error creating model file: %w", err)
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	if err := encoder.Encode(model); err != nil {
		return fmt.Errorf("error encoding model: %w", err)
	}
	return nil
}

func TrainWithActiveLearning(iterations, batchSize int, samplingMethod string) {
	// 2. Load pre-trained word2vec model
	word2vecModel, err := word2vec.LoadModel(word2vecModelPath)
	if err != nil {
		fmt.Println("Error loading word2vec model: ", err)
		return
	}
	vocabularySize := len(word2vecModel.Vocabulary)

	// 3. Load SRL training data (seed set and unlabeled)
	allTrainingData, err := LoadRoleData(trainingDataPath)
	if err != nil {
		fmt.Println("Error loading training data: ", err)
		return
	}

	// In a real scenario, you'd separate this into labeled (seed) and unlabeled data.
	// For this simulation, we'll treat all data as initially "labeled" but simulate
	// annotation by using the existing labels as "true" labels for selected examples.
	labeledData := make([]semanticrole.SentenceRoleData, 0) // Start with an empty labeled set
	unlabeledData := allTrainingData                        // Initially, all data is "unlabeled"

	// 4. Create role map
	roleMap := createRoleMap(allTrainingData) // Use all data to create the role map
	fmt.Printf("Role Map: %v\n", roleMap)

	// 5. Active learning loop
	for iteration := 0; iteration < iterations; iteration++ {
		fmt.Printf("Active Learning Iteration %d\n", iteration+1)

		// a. Train model on labeled data
		fmt.Println("Training model on labeled data...")
		model := bilstm_model.NewBiLSTMModel(vocabularySize, hiddenSize, len(roleMap))
		model.InitializeOutputLayer(hiddenSize)
		trainModel(model, labeledData, word2vecModel, roleMap) // Train on labeled data

		if len(unlabeledData) == 0 {
			fmt.Println("No more unlabeled data. Active learning complete.")
			break
		}

		// b. Predict on unlabeled data and select uncertain examples
		predictions := predictRoles(model, unlabeledData, word2vecModel, roleMap)
		var selectedIndices []int
		switch samplingMethod {
		case "least_confidence":
			selectedIndices = selectUncertainExamplesLeastConfidence(predictions, batchSize)
		default:
			fmt.Println("Invalid sampling method. Using least_confidence.")
			selectedIndices = selectUncertainExamplesLeastConfidence(predictions, batchSize)
		}

		// c. "Annotate" selected examples (simulation: use existing labels)
		fmt.Printf("Simulating annotation for %d examples...\n", len(selectedIndices))
		newlyLabeled := make([]semanticrole.SentenceRoleData, len(selectedIndices))
		for i, index := range selectedIndices {
			newlyLabeled[i] = unlabeledData[index]
		}

		// d. Update labeled and unlabeled data
		labeledData = append(labeledData, newlyLabeled...)
		unlabeledData = removeSelected(unlabeledData, selectedIndices)

		fmt.Printf("Labeled data size: %d, Unlabeled data size: %d\n", len(labeledData), len(unlabeledData))
	}

	// 6. Save the final trained model
	finalModel := bilstm_model.NewBiLSTMModel(vocabularySize, hiddenSize, len(roleMap))
	finalModel.InitializeOutputLayer(hiddenSize)
	trainModel(finalModel, labeledData, word2vecModel, roleMap) // Train on all labeled data
	saveModel(finalModel, bilstmModelPath)
	saveRoleMap(roleMap, roleMapPath)

	fmt.Println("Active learning complete. Final model and role map saved.")
}

// Helper function to train the model (similar to Train_bilstm's core logic)
func trainModel(model *bilstm_model.BiLSTMModel, trainingData []semanticrole.SentenceRoleData, word2vecModel *word2vec.SimpleWord2Vec, roleMap map[string]int) {
	// Training loop (similar to the original Train_bilstm, but adapted for the function signature)
	for epoch := 0; epoch < epochs; epoch++ {
		fmt.Printf("  Epoch %d\n", epoch+1)
		rand.Shuffle(len(trainingData), func(i, j int) {
			trainingData[i], trainingData[j] = trainingData[j], trainingData[i]
		})
		for _, taggedSentence := range trainingData {
			if len(taggedSentence.Tokens) == 0 {
				fmt.Println("Skipping sentence with no tokens.")
				continue
			}

			embeddings := make([][]float64, len(taggedSentence.Tokens))

			roleIDs := make([]int, len(taggedSentence.Tokens))
			for i, token := range taggedSentence.Tokens {
				if i >= len(taggedSentence.Tokens) {
					fmt.Printf("Invalid index i=%d for taggedSentence.Tokens of length %d. Skipping.\n", i, len(taggedSentence.Tokens)) // Robust check to ensure i is a valid index
					continue
				}
				embedding, ok := word2vecModel.WordVectors[word2vecModel.Vocabulary[token.Token]]
				if !ok { // UNK token
					embedding = make([]float64, len(word2vecModel.WordVectors[word2vecModel.Vocabulary[word2vec.UNKToken]]))
					copy(embedding, word2vecModel.WordVectors[word2vecModel.Vocabulary[word2vec.UNKToken]])
				}
				embeddings[i] = embedding

				roleID, ok := roleMap[token.Role]
				if !ok {
					roleID = 0 // Assign role ID 0 for unknown roles
				}
				roleIDs[i] = roleID
			}
			tokenIDs := make([]int, len(taggedSentence.Tokens))
			for i, token := range taggedSentence.Tokens {
				tokenIDs[i] = word2vecModel.Vocabulary[token.Token]
			}
			probabilities := model.ForwardAndCalculateProbabilities(tokenIDs)
			var loss float64 = 0.0
			for tokenIndex, prob := range probabilities {
				loss += crossEntropyLoss(prob, roleIDs[tokenIndex])
			}
			if len(roleIDs) > 0 {
				model.Backpropagate(probabilities, roleIDs, tokenIDs) // Pass tokenIDs instead of embeddings
				model.UpdateWeights(learningRate)
			} else {
				fmt.Println("Skipping backpropagation for sentence with no roles.")
			}
			model.ClipGradients(maxGrad)
			if loss > 0 {
				fmt.Printf("  Loss: %f\n", loss)
			}
		}
	}
}

// Helper function to predict roles on unlabeled data
func predictRoles(model *bilstm_model.BiLSTMModel, data []semanticrole.SentenceRoleData, word2vecModel *word2vec.SimpleWord2Vec, roleMap map[string]int) [][][]float64 {
	allProbabilities := make([][][]float64, len(data))
	for i, taggedSentence := range data {
		if len(taggedSentence.Tokens) == 0 {
			allProbabilities[i] = [][]float64{}
			continue
		}

		tokenProbabilities := make([][]float64, len(taggedSentence.Tokens))

		for k := range tokenProbabilities {
			tokenProbabilities[k] = make([]float64, len(roleMap))
		}
		embeddings := make([][]float64, len(taggedSentence.Tokens))
		for j, token := range taggedSentence.Tokens {
			embedding, ok := word2vecModel.WordVectors[word2vecModel.Vocabulary[token.Token]]
			if !ok {
				embedding = make([]float64, len(word2vecModel.WordVectors[word2vecModel.Vocabulary[word2vec.UNKToken]]))
				copy(embedding, word2vecModel.WordVectors[word2vecModel.Vocabulary[word2vec.UNKToken]])
			}
			embeddings[j] = embedding
		}
		tokenIDs := make([]int, len(taggedSentence.Tokens))
		for i, token := range taggedSentence.Tokens {
			tokenIDs[i] = word2vecModel.Vocabulary[token.Token]
		}
		probabilities := model.ForwardAndCalculateProbabilities(tokenIDs)

		for j, probs := range probabilities {
			for k, prob := range probs {
				tokenProbabilities[j][k] = prob
			}
		}
		allProbabilities[i] = tokenProbabilities
	}
	return allProbabilities
}

// Helper function for least confidence sampling
func selectUncertainExamplesLeastConfidence(probabilities [][][]float64, batchSize int) []int {
	scores := make([]float64, len(probabilities))
	for i, sentenceProbs := range probabilities {
		if len(sentenceProbs) == 0 {
			scores[i] = 1.0 // Assign highest uncertainty to empty sentences
			continue
		}
		sentenceScore := 0.0
		for _, tokenProbs := range sentenceProbs {
			maxProb := 0.0
			for _, prob := range tokenProbs {
				if prob > maxProb {
					maxProb = prob
				}
			}
			sentenceScore += (1 - maxProb) // Uncertainty = 1 - confidence
		}
		scores[i] = sentenceScore / float64(len(sentenceProbs)) // Average uncertainty per token
	}
	return getTopIndices(scores, batchSize)
}

// Helper function to get indices of top-scoring elements
func getTopIndices(scores []float64, topN int) []int {
	if topN >= len(scores) {
		indices := make([]int, len(scores))
		for i := range scores {
			indices[i] = i
		}
		return indices
	}
	pairs := make([]indexScorePair, len(scores))
	for i, score := range scores {
		pairs[i] = indexScorePair{i, score}
	}
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].score > pairs[j].score
	})
	topIndices := make([]int, topN)
	for i := 0; i < topN; i++ {
		topIndices[i] = pairs[i].index
	}
	return topIndices
}

type indexScorePair struct {
	index int
	score float64
}

// Helper function to remove elements at given indices from a slice
func removeSelected(data []semanticrole.SentenceRoleData, indices []int) []semanticrole.SentenceRoleData {
	if len(indices) == 0 {
		return data
	}
	sort.Sort(sort.Reverse(sort.IntSlice(indices))) // Sort in descending order for safe removal
	for _, index := range indices {
		data = append(data[:index], data[index+1:]...)
	}
	return data
}

func saveRoleMap(roleMap map[string]int, path string) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("error creating role map file: %w", err)
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	if err := encoder.Encode(roleMap); err != nil {
		return fmt.Errorf("error encoding role map: %w", err)
	}
	return nil
}
