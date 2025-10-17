package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math" // Keep math for softmax
	"math/rand"
	"os"
	"os/exec"
	"sort"
	"strings"

	"nlptagger/neural/moe"
	mainvocab "nlptagger/neural/nnu/vocab"
	"nlptagger/neural/tensor"
	"nlptagger/neural/tokenizer"
	"nlptagger/tagger/nertagger"
	"nlptagger/tagger/postagger"
	"nlptagger/tagger/tag"
)

// IntentTrainingExample represents a single training example for intent classification.
type IntentTrainingExample struct {
	Query        string `json:"query"`
	ParentIntent string `json:"parent_intent"`
	ChildIntent  string `json:"child_intent"`
	Description  string `json:"description"`
	Sentence     string `json:"sentence"`
}

// IntentTrainingData represents the structure of the intent training data JSON.
type IntentTrainingData []IntentTrainingExample

// ExpectedState defines the desired state of a filesystem entity.
type ExpectedState struct {
	Type string // "directory" or "file"
	Path string
}

// MoEModel defines the interface for a model that has a Forward method.
type MoEModel interface {
	Forward(inputs ...*tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, []*tensor.Tensor, *tensor.Tensor, error)
}

// LoadIntentTrainingData loads the intent training data from a JSON file.
func LoadIntentTrainingData(filePath string) (*IntentTrainingData, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open training data file %s: %w", filePath, err)
	}
	defer file.Close()

	bytes, err := io.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("failed to read training data file %s: %w", filePath, err)
	}

	var data IntentTrainingData
	err = json.Unmarshal(bytes, &data)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal training data JSON from %s: %w", filePath, err)
	}

	return &data, nil
}

type Prediction struct {
	TokenID    int
	Word       string
	Confidence float64
}

func getTopNPredictions(probabilities []float64, vocab []string, n int) []Prediction {
	predictions := make([]Prediction, 0, len(probabilities))
	for i, p := range probabilities {
		if i < 2 { // Skip <pad> and UNK
			continue
		}
		if i < len(vocab) {
			word := vocab[i]
			predictions = append(predictions, Prediction{
				TokenID:    i,
				Word:       word,
				Confidence: p * 100.0,
			})
		}
	}

	// Sort predictions by confidence
	sort.Slice(predictions, func(i, j int) bool {
		return predictions[i].Confidence > predictions[j].Confidence
	})

	if len(predictions) < n {
		return predictions
	}
	return predictions[:n]
}

var (
	query        = flag.String("query", "", "Query for MoE inference")
	maxSeqLength = flag.Int("maxlen", 32, "Maximum sequence length")
)

func main() {
	rand.Seed(1) // Seed the random number generator for deterministic behavior
	flag.Parse()

	// Define paths
	const vocabPath = "gob_models/query_vocabulary.gob"
	const moeModelPath = "gob_models/moe_classification_model.gob"
	const parentIntentVocabPath = "gob_models/parent_intent_vocabulary.gob"
	const childIntentVocabPath = "gob_models/child_intent_vocabulary.gob"

	// Load vocabularies
	vocabulary, err := mainvocab.LoadVocabulary(vocabPath)
	if err != nil {
		log.Fatalf("Failed to set up input vocabulary: %v", err)
	}

	// Setup parent intent vocabulary
	parentIntentVocabulary, err := mainvocab.LoadVocabulary(parentIntentVocabPath)
	if err != nil {
		log.Fatalf("Failed to set up parent intent vocabulary: %v", err)
	}

	// Setup child intent vocabulary
	childIntentVocabulary, err := mainvocab.LoadVocabulary(childIntentVocabPath)
	if err != nil {
		log.Fatalf("Failed to set up child intent vocabulary: %v", err)
	}

	// Create tokenizer
	tok, err := tokenizer.NewTokenizer(vocabulary)
	if err != nil {
		log.Fatalf("Failed to create tokenizer: %w", err)
	}

	// Load the trained MoEClassificationModel model
	model, err := moe.LoadIntentMoEModelFromGOB(moeModelPath)
	if err != nil {
		log.Fatalf("Failed to load MoE model: %v", err)
	}

	// Process initial query from flag, if provided
	if *query != "" {
		processQuery(*query, tok, model, vocabulary, parentIntentVocabulary, childIntentVocabulary, *maxSeqLength)
	}

	// Start interactive loop
	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("\nWhat operations do I need to run now? ")
		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)

		if input == "exit" || input == "quit" {
			break
		}

		if input != "" {
			processQuery(input, tok, model, vocabulary, parentIntentVocabulary, childIntentVocabulary, *maxSeqLength)
		}
	}
}

func processQuery(query string, tok *tokenizer.Tokenizer, model MoEModel, vocabulary *mainvocab.Vocabulary, parentIntentVocabulary *mainvocab.Vocabulary, childIntentVocabulary *mainvocab.Vocabulary, maxSeqLength int) {
	// Encode the query
	tokenIDs, err := tok.Encode(query)
	if err != nil {
		log.Printf("Failed to encode query: %v", err)
		return
	}

	// Pad or truncate the sequence to a fixed length
	if len(tokenIDs) > maxSeqLength {
		tokenIDs = tokenIDs[:maxSeqLength] // Truncate from the end
	} else {
		for len(tokenIDs) < maxSeqLength {
			tokenIDs = append(tokenIDs, vocabulary.PaddingTokenID) // Appends padding
		}
	}
	inputData := make([]float64, len(tokenIDs))
	for i, id := range tokenIDs {
		inputData[i] = float64(id)
	}
	inputTensor := tensor.NewTensor([]int{1, len(inputData)}, inputData, false) // RequiresGrad=false for inference

	// Create a dummy target tensor for inference, as the Forward method expects two inputs.
	dummyTargetTokenIDs := make([]float64, maxSeqLength)
	for i := 0; i < maxSeqLength; i++ {
		dummyTargetTokenIDs[i] = float64(vocabulary.PaddingTokenID)
	}
	dummyTargetTensor := tensor.NewTensor([]int{1, maxSeqLength}, dummyTargetTokenIDs, false)

	// Forward pass
	parentLogits, childLogits, _, _, err := model.Forward(inputTensor, dummyTargetTensor)
	if err != nil {
		log.Printf("MoE model forward pass failed: %v", err)
		return
	}

	// Interpret parent intent output
	parentProbabilities := softmax(parentLogits.Data)
	topParentPredictions := getTopNPredictions(parentProbabilities, parentIntentVocabulary.TokenToWord, 3)

	// Interpret child intent output
	childProbabilities := softmax(childLogits.Data)
	topChildPredictions := getTopNPredictions(childProbabilities, childIntentVocabulary.TokenToWord, 3)

	// Perform POS tagging
	posResult := postagger.Postagger(query)

	// Perform NER tagging
	nerResult := nertagger.Nertagger(posResult)

	var parentIntent string
	if len(topParentPredictions) > 0 {
		parentIntent = topParentPredictions[0].Word
	}

	var childIntent string
	if len(topChildPredictions) > 0 {
		childIntent = topChildPredictions[0].Word
	}

	var objectTypes []string
	var names []string
	for i, tag := range nerResult.NerTag {
		if tag == "OBJECT_TYPE" {
			objectTypes = append(objectTypes, nerResult.Tokens[i])
		}
		if tag == "NAME" {
			names = append(names, nerResult.Tokens[i])
		}
	}

	fmt.Println("\n| Identified Elements | Values in Query |")
	fmt.Println("| :--- | :--- |")
	fmt.Println("| Parent Intent |", parentIntent, "|")
	fmt.Println("| Child Intent |", childIntent, "|")
	fmt.Println("| Object Types |", strings.Join(objectTypes, ", "), "|")
	fmt.Println("| Names |", strings.Join(names, ", "), "|")

	command := generateCommand(parentIntent, childIntent, nerResult)
	expectedStates := generateExpectedState(parentIntent, childIntent, nerResult)

	fmt.Println("\n| Key Question | |")
	fmt.Println("| :--- | :--- |")
	fmt.Printf("| What operations do I need to run now? | %s |\n", command)
	fmt.Printf("| What should the environment look like after all changes? | %s |\n", describeExpectedStates(expectedStates))

	// Generate and execute command based on NER/POS tags and intent predictions
	fmt.Println("\n--- Generating Command ---")

	if command != "" {
		fmt.Printf("Generated Command: %s\n", command)
		// Execute the command
		cmd := exec.Command("bash", "-c", command)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		err := cmd.Run()
		if err != nil {
			log.Printf("Error executing command: %v", err)
		} else {
			// New verification step
			fmt.Println("\n--- Verifying State ---")
			if verifyState(expectedStates) {
				fmt.Println("Declarative state verification successful.")
			} else {
				fmt.Println("Declarative state verification failed.")
			}
		}
	} else {
		fmt.Println("Could not generate a command.")
	}
}

// softmax applies the softmax function to a slice of float64.
func softmax(logits []float64) []float64 {
	if len(logits) == 0 {
		return []float64{}
	}
	maxLogit := logits[0]
	for _, logit := range logits {
		if logit > maxLogit {
			maxLogit = logit
		}
	}
	expSum := 0.0
	for _, logit := range logits {
		expSum += math.Exp(logit - maxLogit)
	}

	probabilities := make([]float64, len(logits))
	for i, logit := range logits {
		probabilities[i] = math.Exp(logit-maxLogit) / expSum
	}
	return probabilities
}

func generateCommand(parentIntent, childIntent string, nerResult tag.Tag) string {
	switch parentIntent {
	case "file_system":
		switch childIntent {
		case "create":
			var fileName, folderName string
			for i, tag := range nerResult.NerTag {
				if tag == "OBJECT_TYPE" && nerResult.Tokens[i] == "file" {
					if i+2 < len(nerResult.Tokens) && nerResult.NerTag[i+1] == "NAME_PREFIX" && nerResult.NerTag[i+2] == "NAME" {
						fileName = nerResult.Tokens[i+2]
					}
				} else if tag == "OBJECT_TYPE" && nerResult.Tokens[i] == "folder" {
					if i+2 < len(nerResult.Tokens) && nerResult.NerTag[i+1] == "NAME_PREFIX" && nerResult.NerTag[i+2] == "NAME" {
						folderName = nerResult.Tokens[i+2]
					}
				}
			}
			if fileName != "" && folderName != "" {
				return fmt.Sprintf("mkdir -p %s && touch %s/%s", folderName, folderName, fileName)
			} else if fileName != "" {
				return fmt.Sprintf("touch %s", fileName)
			}
		}
		// Add other file_system child intents here (e.g., "delete", "read")
	case "webserver_creation":
		switch childIntent {
		case "create":
			var webserverName string
			for i, t := range nerResult.NerTag {
				if t == "OBJECT_TYPE" && nerResult.Tokens[i] == "webserver" {
					// Find the next NAME
					for j := i + 1; j < len(nerResult.NerTag); j++ {
						if nerResult.NerTag[j] == "NAME" {
							webserverName = nerResult.Tokens[j]
							break
						}
					}
				}
			}

			if webserverName != "" {
				return fmt.Sprintf(`
mkdir -p %s && \
cat > %s/main.go <<'EOF'
package main

import (
	"fmt"
	"log"
	"net/http"
)

func handler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Hello, World!")
}

func main() {
	http.HandleFunc("/", handler)
	log.Println("Starting web server on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
EOF
`, webserverName, webserverName)
			}
		}
		// Add other webserver_creation child intents here
	}
	// Add other parent intents here

	return ""
}


func generateExpectedState(parentIntent, childIntent string, nerResult tag.Tag) []ExpectedState {
	var states []ExpectedState
	switch parentIntent {
	case "file_system":
		switch childIntent {
		case "create":
			var fileName, folderName string
			for i, tag := range nerResult.NerTag {
				if tag == "OBJECT_TYPE" && nerResult.Tokens[i] == "file" {
					if i+2 < len(nerResult.Tokens) && nerResult.NerTag[i+1] == "NAME_PREFIX" && nerResult.NerTag[i+2] == "NAME" {
						fileName = nerResult.Tokens[i+2]
					}
				} else if tag == "OBJECT_TYPE" && nerResult.Tokens[i] == "folder" {
					if i+2 < len(nerResult.Tokens) && nerResult.NerTag[i+1] == "NAME_PREFIX" && nerResult.NerTag[i+2] == "NAME" {
						folderName = nerResult.Tokens[i+2]
					}
				}
			}
			if fileName != "" && folderName != "" {
				states = append(states, ExpectedState{Type: "directory", Path: folderName})
				states = append(states, ExpectedState{Type: "file", Path: fmt.Sprintf("%s/%s", folderName, fileName)})
			} else if fileName != "" {
				states = append(states, ExpectedState{Type: "file", Path: fileName})
			}
		}
	}
	return states
}

func verifyState(states []ExpectedState) bool {
	for _, state := range states {
		info, err := os.Stat(state.Path)
		if os.IsNotExist(err) {
			log.Printf("State verification failed: Path does not exist: %s", state.Path)
			return false
		}
		if err != nil {
			log.Printf("State verification failed: Error accessing path %s: %v", state.Path, err)
			return false
		}

		switch state.Type {
		case "directory":
			if !info.IsDir() {
				log.Printf("State verification failed: Path is not a directory: %s", state.Path)
				return false
			}
		case "file":
			if info.IsDir() {
				log.Printf("State verification failed: Path is not a file: %s", state.Path)
				return false
			}
		}
	}
	return true
}

func describeExpectedStates(states []ExpectedState) string {
	var descriptions []string
	for _, s := range states {
		descriptions = append(descriptions, fmt.Sprintf("A %s named '%s' should exist.", s.Type, s.Path))
	}
	if len(descriptions) == 0 {
		return "No changes to the environment."
	}
	return strings.Join(descriptions, " ")
}
