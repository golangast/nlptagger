package main

import (
	"bufio"
	"encoding/base64"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strings"

	"github.com/zendrulat/nlptagger/neural/moe"
	"github.com/zendrulat/nlptagger/neural/nnu/context" // Import the new context package
	mainvocab "github.com/zendrulat/nlptagger/neural/nnu/vocab"
	"github.com/zendrulat/nlptagger/neural/nnu/word2vec"
	"github.com/zendrulat/nlptagger/neural/tensor"
	"github.com/zendrulat/nlptagger/neural/tokenizer"
)

var (
	query        = flag.String("query", "", "Query for MoE inference")
	maxSeqLength = flag.Int("maxlen", 32, "Maximum sequence length")
)

func main() {
	fmt.Fprintf(os.Stderr, "DEBUG: Entering moe_inference main function.\n")
	rand.Seed(1) // Seed the random number generator for deterministic behavior
	flag.Parse()

	// Initialize ConversationContext
	conversationContext := context.NewConversationContext(3) // Store last 3 turns

	// Define paths
	const vocabPath = "gob_models/query_vocabulary.gob"
	const moeModelPath = "gob_models/moe_classification_model.gob"
	const semanticOutputVocabPath = "gob_models/semantic_output_vocabulary.gob"
	const word2vecModelPath = "gob_models/word2vec_model.gob"

	// Load the Word2Vec model
	log.Printf("Loading Word2Vec model from %s...", word2vecModelPath)
	word2vecModel, err := word2vec.LoadModel(word2vecModelPath)
	if err != nil {
		log.Fatalf("Failed to load Word2Vec model: %v", err)
	}
	log.Println("Word2vec model loaded successfully.")

	// Load vocabularies
	// Create a mainvocab.Vocabulary from the word2vecModel.Vocabulary (map[string]int)
	queryVocabulary := mainvocab.NewVocabulary()
	for word, id := range word2vecModel.Vocabulary {
		queryVocabulary.WordToToken[word] = id
		// Ensure TokenToWord is also populated correctly
		for len(queryVocabulary.TokenToWord) <= id {
			queryVocabulary.TokenToWord = append(queryVocabulary.TokenToWord, "")
		}
		queryVocabulary.TokenToWord[id] = word
	}
	queryVocabulary.PaddingTokenID = queryVocabulary.GetTokenID(word2vec.PaddingToken)
	queryVocabulary.UnkID = queryVocabulary.GetTokenID(word2vec.UNKToken)

	semanticOutputVocabulary, err := mainvocab.LoadVocabulary(semanticOutputVocabPath)
	if err != nil {
		log.Fatalf("Failed to set up semantic output vocabulary: %v", err)
	}
	for word, id := range semanticOutputVocabulary.WordToToken {
		log.Printf("Vocab: %s -> %d", word, id)
	}

	// Create tokenizer
	tok, err := tokenizer.NewTokenizer(queryVocabulary)
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
		processQuery(*query, model, queryVocabulary, semanticOutputVocabulary, tok, semanticOutputTokenizer, conversationContext, maxSeqLength)
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
		processQuery(input, model, queryVocabulary, semanticOutputVocabulary, tok, semanticOutputTokenizer, conversationContext, maxSeqLength)
	}
	return // Added return statement here
}

// processQuery encapsulates the logic for processing a single query
func processQuery(q string, model *moe.IntentMoE, vocabulary *mainvocab.Vocabulary, semanticOutputVocabulary *mainvocab.Vocabulary, tok *tokenizer.Tokenizer, semanticOutputTokenizer *tokenizer.Tokenizer, conversationContext *context.ConversationContext, maxSeqLength *int) {
	log.Printf("Input vocabulary size: %d", vocabulary.Size())
	log.Printf("Semantic output vocabulary size: %d", semanticOutputVocabulary.Size())
	log.Printf("Token ID for 'intent': %d", semanticOutputVocabulary.GetTokenID("intent"))
	log.Printf("Token ID for 'CREATE_FILESYSTEM_OBJECTS': %d", semanticOutputVocabulary.GetTokenID("CREATE_FILESYSTEM_OBJECTS"))
	log.Printf("Token ID for 'folder_name': %d", semanticOutputVocabulary.GetTokenID("folder_name"))
	log.Printf("Token ID for 'file_name': %d", semanticOutputVocabulary.GetTokenID("file_name"))

	log.Printf("Running MoE inference for query: \"%s\"", q)

	// Resolve co-references using the conversation context
	resolvedQuery := conversationContext.ResolveCoReference(q)
	log.Printf("Resolved query: \"%s\"", resolvedQuery)

	// Encode the resolved query
	tokenIDs, err := tok.Encode(resolvedQuery)
	if err != nil {
		log.Fatalf("Failed to encode query: %v", err)
	}
	log.Printf("Token IDs: %v", tokenIDs)

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

	log.Printf("Predicted token IDs: %v", predictedIDs)

	// Decode the predicted IDs to a sentence
	predictedSentence, err := semanticOutputTokenizer.Decode(predictedIDs)
	if err != nil {
		log.Fatalf("Failed to decode predicted IDs: %v", err)
	}

	// The model's predicted sentence will be used directly, without hardcoded overrides.

	// Parse the semantic output to extract intent and entities
	predictedIntent, predictedEntities := parseSemanticOutput(predictedSentence)

	// Update the conversation context
	conversationContext.AddTurn(predictedIntent, predictedEntities, q)

	fmt.Println("--- Predicted Semantic Output ---")
	fmt.Println(predictedSentence)
	fmt.Println("---------------------------------")

	fmt.Printf("Current Conversation Context: Intent = %s, Entities = %+v\n", conversationContext.CurrentIntent, conversationContext.CurrentEntities)

	performActionFromSemanticOutput(predictedIntent, predictedEntities, q, predictedSentence)
}



func parseSemanticOutput(semanticOutput string) (string, []context.Entity) {
	intent := ""
	entities := []context.Entity{}

	// Remove <s> and </s> tokens if present
	cleanedOutput := strings.ReplaceAll(semanticOutput, "<s> ", "")
	cleanedOutput = strings.ReplaceAll(cleanedOutput, " </s>", "")
	cleanedOutput = strings.TrimSpace(cleanedOutput)

	parts := strings.Split(cleanedOutput, ",")
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
		} else {
			log.Printf("Warning: Semantic output part '%s' could not be parsed into key:value pair.", part)
		}
	}

	return intent, entities
}


// performActionFromSemanticOutput interprets the predicted intent and entities to perform actions.
func performActionFromSemanticOutput(intent string, entities []context.Entity, originalQuery string, predictedSentence string) {
	switch intent {
	case "Create": // Handle "Create" as an alias for CREATE_FILESYSTEM_OBJECTS
	case "CREATE_FILESYSTEM_OBJECTS":
		var fileName, folderName string
		for _, entity := range entities {
			if entity.Type == "file_name" {
				fileName = entity.Value
			} else if entity.Type == "folder_name" {
				folderName = entity.Value
			}
		}

		// If fileName or folderName are still empty, try to extract them from the original query using heuristics
		if fileName == "" || folderName == "" {
			log.Println("Applying heuristic for file system object creation due to missing entities.")
			// Extract folder name
			folderIndex := strings.Index(originalQuery, "create folder")
			if folderIndex != -1 {
				remaining := originalQuery[folderIndex+len("create folder"):]
				andIndex := strings.Index(remaining, " and add a file")
				if andIndex != -1 {
					folderName = strings.TrimSpace(remaining[:andIndex])
				} else {
					// If "and add a file" is not present, assume folder name is till the end
					folderName = strings.TrimSpace(remaining)
				}
			}

			// Extract file name
			fileIndex := strings.Index(originalQuery, "add a file")
			if fileIndex != -1 {
				remaining := originalQuery[fileIndex+len("add a file"):]
				fileName = strings.TrimSpace(remaining)
			}
		}

		if fileName != "" && folderName != "" {
			filePath := fmt.Sprintf("%s/%s", folderName, fileName)
			// Ensure the directory exists
			err := os.MkdirAll(folderName, 0755)
			if err != nil {
				log.Printf("Error creating directory %s: %v", folderName, err)
				return
			}
			// Create the file
			file, err := os.Create(filePath)
			if err != nil {
				log.Printf("Error creating file %s: %v", filePath, err)
				return
			}
			file.Close()
			log.Printf("Created file: %s", filePath)
		} else {
			log.Printf("CREATE_FILESYSTEM_OBJECTS intent received, but file_name or folder_name is missing. Original query: %s", originalQuery)
		}

	case "UPDATE_FILE_CONTENT":
		var fileName, directory, content, contentType string
		for _, entity := range entities {
			if entity.Type == "file_name" {
				fileName = entity.Value
			} else if entity.Type == "directory" {
				directory = entity.Value
			} else if entity.Type == "content" {
				content = entity.Value
			} else if entity.Type == "content_type" {
				contentType = entity.Value
			}
		}

		if fileName != "" {
			if directory == "" {
				directory = "."
			}
			filePath := fmt.Sprintf("%s/%s", directory, fileName)
			var fileContent []byte

			if contentType == "webserver_go" {
				// Generate Go web server code
				webserverCode := `package main

import (
	"fmt"
	"log"
	"net/http"
)

func main() {
	http.HandleFunc("/", handler)
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func handler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Hello, Jack!")
}`
				fileContent = []byte(webserverCode)
			} else if content != "" {
				// If content is provided and not a specific content_type, assume it's base64 encoded
				decodedContent, err := base64.StdEncoding.DecodeString(content)
				if err != nil {
					log.Printf("Error decoding base64 content for file %s: %v", filePath, err)
					return
				}
				fileContent = decodedContent
			} else {
				log.Printf("UPDATE_FILE_CONTENT intent received for %s, but no content or content_type specified.", filePath)
				return
			}

			err := os.WriteFile(filePath, fileContent, 0644)
			if err != nil {
				log.Printf("Error writing to file %s: %v", filePath, err)
				return
			}
			log.Printf("Updated file: %s", filePath)
		} else {
			log.Printf("UPDATE_FILE_CONTENT intent received, but file_name or directory is missing.")
		}

	default:
		if intent == "" {
			log.Printf("Warning: No intent was extracted from the semantic output. Predicted sentence: \"%s\". Original query: \"%s\"", predictedSentence, originalQuery)
		} else {
			log.Printf("No action defined for intent: %s", intent)
		}
	}
}