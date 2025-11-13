package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strings"

	"encoding/json"

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
	log.Println("Semantic Output Vocabulary contents:")
	for word, id := range semanticOutputVocabulary.WordToToken {
		log.Printf("  %s -> %d", word, id)
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
	semanticOutput, err := parseSemanticOutput(predictedSentence, q)
	if err != nil {
		log.Fatalf("Failed to parse semantic output even with heuristics: %v", err)
	}

	// Update the conversation context
	conversationContext.AddTurn(semanticOutput.Operation, extractEntities(semanticOutput), q)

	fmt.Println("--- Predicted Semantic Output ---")
	jsonOutput, err := json.MarshalIndent(semanticOutput, "", "  ")
	if err != nil {
		log.Printf("Error marshalling semantic output to JSON: %v", err)
		fmt.Println(predictedSentence) // Fallback to raw if marshalling fails
	} else {
		fmt.Println(string(jsonOutput))
	}
	fmt.Println("---------------------------------")

	fmt.Printf("Current Conversation Context: Intent = %s, Entities = %+v\n", conversationContext.CurrentIntent, conversationContext.CurrentEntities)

	performActionFromSemanticOutput(semanticOutput, q, predictedSentence)
}



func extractEntities(so context.SemanticOutput) []context.Entity {
	var entities []context.Entity
	if so.TargetResource.Type != "" {
		entities = append(entities, context.Entity{Type: "target_resource_type", Value: so.TargetResource.Type})
	}
	if so.TargetResource.Name != "" {
		entities = append(entities, context.Entity{Type: "target_resource_name", Value: so.TargetResource.Name})
	}
	if so.TargetResource.Directory != "" {
		entities = append(entities, context.Entity{Type: "target_resource_directory", Value: so.TargetResource.Directory})
	}
	if so.TargetResource.Destination != "" {
		entities = append(entities, context.Entity{Type: "target_resource_destination", Value: so.TargetResource.Destination})
	}
	for key, value := range so.TargetResource.Properties {
		entities = append(entities, context.Entity{Type: key, Value: fmt.Sprintf("%v", value)})
	}
	for _, child := range so.TargetResource.Children {
		if child.Type != "" {
			entities = append(entities, context.Entity{Type: "child_type", Value: child.Type})
		}
		if child.Name != "" {
			entities = append(entities, context.Entity{Type: "child_name", Value: child.Name})
		}
		for key, value := range child.Properties {
			entities = append(entities, context.Entity{Type: "child_" + key, Value: fmt.Sprintf("%v", value)})
		}
	}
	return entities
}

func parseSemanticOutput(semanticOutput string, originalQuery string) (context.SemanticOutput, error) {
	var so context.SemanticOutput
	// Remove <s> and </s> tokens if present
	cleanedOutput := strings.ReplaceAll(semanticOutput, "<s> ", "")
	cleanedOutput = strings.ReplaceAll(cleanedOutput, " </s>", "")
	cleanedOutput = strings.TrimSpace(cleanedOutput)

	err := json.Unmarshal([]byte(cleanedOutput), &so)
	if err != nil {
		log.Printf("Failed to parse semantic output: %v. Attempting to use heuristics.", err)
		// Heuristic-based approach when semantic output is invalid
		so = context.SemanticOutput{} // Initialize an empty semantic output
		log.Printf("DEBUG: Original query (originalQuery) in heuristic: '%s'", originalQuery)

		var fileName, folderName, content string
		var pathFromHeuristic string

		// Determine intent based on keywords
		if strings.Contains(originalQuery, "create folder") || strings.Contains(originalQuery, "add file") || strings.Contains(originalQuery, "in folder") || strings.Contains(originalQuery, "make a new directory") || strings.Contains(originalQuery, "create a directory") {
			so.Operation = "CREATE_FILESYSTEM_OBJECTS"
			log.Printf("DEBUG: Heuristic set Operation to: %s", so.Operation)
		} else if strings.Contains(originalQuery, "update file") || strings.Contains(originalQuery, "write to file") || strings.Contains(originalQuery, "in file") || strings.Contains(originalQuery, "change the content") || strings.Contains(originalQuery, "append") || (strings.Contains(originalQuery, "add") && strings.Contains(originalQuery, "code")) {
			so.Operation = "UPDATE_FILE_CONTENT"
			log.Printf("DEBUG: Heuristic set Operation to: %s", so.Operation)
		} else if strings.Contains(originalQuery, "delete") || strings.Contains(originalQuery, "remove") || strings.Contains(originalQuery, "get rid of") || strings.Contains(originalQuery, "erase") {
			so.Operation = "Delete"
			log.Printf("DEBUG: Heuristic set Operation to: %s", so.Operation)
		} else if strings.Contains(originalQuery, "move") || strings.Contains(originalQuery, "relocate") || strings.Contains(originalQuery, "shift") || strings.Contains(originalQuery, "transfer") {
			so.Operation = "Move"
			log.Printf("DEBUG: Heuristic set Operation to: %s", so.Operation)
		} else if strings.Contains(originalQuery, "launch") || strings.Contains(originalQuery, "execute") || strings.Contains(originalQuery, "start") {
			so.Operation = "Run"
			log.Printf("DEBUG: Heuristic set Operation to: %s", so.Operation)
		} else if strings.Contains(originalQuery, "list") || strings.Contains(originalQuery, "show") || strings.Contains(originalQuery, "display") {
			so.Operation = "Read"
			log.Printf("DEBUG: Heuristic set Operation to: %s", so.Operation)
		}

		// Extract file and folder names using heuristics
		// Look for "named X" or "called X" for folder name
		if strings.Contains(originalQuery, "folder named") {
			folderName = extractAfterKeyword(originalQuery, "folder named")
		} else if strings.Contains(originalQuery, "directory called") {
			folderName = extractAfterKeyword(originalQuery, "directory called")
		} else if strings.Contains(originalQuery, "folder") {
			// Generic "folder X"
			folderName = extractAfterKeyword(originalQuery, "folder")
		}

		// Look for "file named X" or "file called X" for file name
		if strings.Contains(originalQuery, "file named") {
			fileName = extractAfterKeyword(originalQuery, "file named")
		} else if strings.Contains(originalQuery, "file called") {
			fileName = extractAfterKeyword(originalQuery, "file called")
		} else if strings.Contains(originalQuery, "file") {
			// Generic "file X"
			fileName = extractAfterKeyword(originalQuery, "file")
		}

		// New heuristic for "in <path>"
		if strings.Contains(originalQuery, "in ") {
			afterIn := extractAfterKeyword(originalQuery, "in")
			if afterIn != "" {
				// Check if it's a path with a slash
				if strings.Contains(afterIn, "/") {
					parts := strings.Split(afterIn, "/")
					if len(parts) > 1 {
						folderName = parts[0]
						fileName = parts[1]
						pathFromHeuristic = folderName // Set path from heuristic
					}
				} else {
					// If no slash, assume it's a file name or folder name
					// Prioritize file name if "file" is also in the query
					if strings.Contains(originalQuery, "file") {
						fileName = afterIn
					} else if strings.Contains(originalQuery, "folder") || strings.Contains(originalQuery, "directory") {
						folderName = afterIn
						pathFromHeuristic = folderName // Set path from heuristic
					}
				}
			}
		}

		log.Printf("DEBUG: Extracted fileName: '%s', folderName: '%s'", fileName, folderName)

		// Refine folderName and fileName if they contain "and add a file" or similar
		if strings.Contains(folderName, "and add a file") {
			folderName = strings.Split(folderName, "and add a file")[0]
			folderName = strings.TrimSpace(folderName)
		}
		if strings.Contains(folderName, "and create a file") {
			folderName = strings.Split(folderName, "and create a file")[0]
			folderName = strings.TrimSpace(folderName)
		}

		// Extract content if "add X in Y" pattern
		if strings.HasPrefix(originalQuery, "add ") && strings.Contains(originalQuery, " in ") {
			startIndex := strings.Index(originalQuery, "add ") + len("add ")
			endIndex := strings.Index(originalQuery, " in ")
			if startIndex < endIndex {
				content = originalQuery[startIndex:endIndex]
				content = strings.TrimSpace(content)
			}
		}

		// Populate semanticOutput.TargetResource based on extracted info
		if fileName != "" {
			so.TargetResource.Type = "Filesystem::File"
			so.TargetResource.Name = fileName
			so.TargetResource.Properties = make(map[string]interface{})
			if pathFromHeuristic != "" {
				so.TargetResource.Properties["path"] = pathFromHeuristic
			} else {
				so.TargetResource.Properties["path"] = "./" // Default path
			}
			if content != "" {
				so.TargetResource.Properties["content"] = content
			}
		}
		if folderName != "" {
			if so.TargetResource.Type == "" { // If not already set by file
				so.TargetResource.Type = "Filesystem::Folder"
			}
			if so.TargetResource.Name == "" { // If not already set by file
				so.TargetResource.Name = folderName
			}
			if so.TargetResource.Properties == nil {
				so.TargetResource.Properties = make(map[string]interface{})
			}
			if pathFromHeuristic != "" {
				so.TargetResource.Properties["path"] = pathFromHeuristic
			} else if _, ok := so.TargetResource.Properties["path"]; !ok {
				so.TargetResource.Properties["path"] = "./" // Default path
			}

			// If both file and folder are present, make the file a child of the folder
			if fileName != "" && so.Operation == "CREATE_FILESYSTEM_OBJECTS" {
				fileChild := context.TargetResource{
					Type: "Filesystem::File",
					Name: fileName,
					Properties: map[string]interface{}{
						"path": folderName,
					},
				}
				so.TargetResource.Children = []context.TargetResource{fileChild}
			}
		}

		// Default operation if not set but file/folder info is present
		if so.Operation == "" && (fileName != "" || folderName != "") {
			so.Operation = "UPDATE_FILE_CONTENT"
			log.Printf("DEBUG: Heuristic set Operation to: %s (default for file/folder with no explicit operation)", so.Operation)
		} else if so.Operation == "" {
			log.Printf("Could not determine intent from query using heuristics for query: '%s'", originalQuery)
			so = context.SemanticOutput{} // Will trigger 'no intent' warning
		}

		log.Printf("DEBUG: Final semanticOutput from heuristic: %+v", so)
		return so, nil // Return the heuristically determined semanticOutput
	}
	return so, nil
}
func extractAfterKeyword(text, keyword string) string {
	idx := strings.Index(text, keyword)
	if idx == -1 {
		return ""
	}
	extracted := text[idx+len(keyword):]
	extracted = strings.TrimSpace(extracted)

	// Remove surrounding quotes if present
	if strings.HasPrefix(extracted, "'") && strings.HasSuffix(extracted, "'") {
		extracted = strings.Trim(extracted, "'")
	} else if strings.HasPrefix(extracted, "\"") && strings.HasSuffix(extracted, "\"") {
		extracted = strings.Trim(extracted, "\"")
	}

	// Take only the first "word" or quoted phrase
	if strings.Contains(extracted, " ") && !strings.HasPrefix(extracted, "\"") && !strings.HasPrefix(extracted, "'") {
		extracted = strings.Split(extracted, " ")[0]
	}
	return extracted
}

// performActionFromSemanticOutput interprets the predicted intent and entities to perform actions.
func performActionFromSemanticOutput(semanticOutput context.SemanticOutput, originalQuery string, predictedSentence string) {
	intent := semanticOutput.Operation
	entities := extractEntities(semanticOutput)

	switch intent {
	case "Create": // Handle "Create" as an alias for CREATE_FILESYSTEM_OBJECTS
	case "CREATE_FILESYSTEM_OBJECTS":
		var fileName, folderName string
		// Prioritize entities from semanticOutput if available
		for _, entity := range entities {
			if entity.Type == "file_name" {
				fileName = entity.Value
			} else if entity.Type == "folder_name" {
				folderName = entity.Value
			} else if entity.Type == "target_resource_name" && semanticOutput.TargetResource.Type == "Filesystem::Folder" {
				folderName = entity.Value
			} else if entity.Type == "target_resource_name" && semanticOutput.TargetResource.Type == "Filesystem::File" {
				fileName = entity.Value
			} else if entity.Type == "child_path" { // Add this condition
				folderName = entity.Value
			}
		}

		// If entities are still missing, use the heuristic from the original query
		if fileName == "" && semanticOutput.TargetResource.Children != nil && len(semanticOutput.TargetResource.Children) > 0 {
			for _, child := range semanticOutput.TargetResource.Children {
				if child.Type == "Filesystem::File" {
					fileName = child.Name
					break
				}
			}
		}
		if folderName == "" && semanticOutput.TargetResource.Type == "Filesystem::Folder" {
			folderName = semanticOutput.TargetResource.Name
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
		var fileName, content, contentType string
		var filePath string
		for _, entity := range entities {
			if entity.Type == "file_name" {
				fileName = entity.Value
			} else if entity.Type == "content_type" {
				contentType = entity.Value
			} else if entity.Type == "target_resource_name" && semanticOutput.TargetResource.Type == "Filesystem::File" {
				fileName = entity.Value
			} else if entity.Type == "target_resource_directory" {
				// This is now handled by semanticOutput.TargetResource.Properties["path"]
			} else if entity.Type == "content_type" {
				contentType = entity.Value
			}
		}
		// Extract content from semanticOutput.TargetResource.Properties
		if contentVal, ok := semanticOutput.TargetResource.Properties["content"]; ok {
			if contentStr, isString := contentVal.(string); isString {
				content = contentStr
			}
		}

		// Fallback to original query heuristics if semanticOutput didn't provide enough info
		if fileName == "" {
			if strings.Contains(originalQuery, "update") {
				fileName = extractAfterKeyword(originalQuery, "update")
			} else if strings.Contains(originalQuery, "change the content of") {
				fileName = extractAfterKeyword(originalQuery, "change the content of")
			} else if strings.Contains(originalQuery, "append") {
				fileName = extractAfterKeyword(originalQuery, "append")
			}
		}
		// Extract content from semanticOutput.TargetResource.Properties
		if contentVal, ok := semanticOutput.TargetResource.Properties["content"]; ok {
			if contentStr, isString := contentVal.(string); isString {
				content = contentStr
			}
		}
		if contentType == "" {
			if strings.Contains(originalQuery, "as") {
				contentType = extractAfterKeyword(originalQuery, "as")
			}
		}

		if fileName != "" {
			// Determine the directory from semanticOutput.TargetResource.Properties["path"]
			dir := "."
			if pathVal, ok := semanticOutput.TargetResource.Properties["path"]; ok {
				if pathStr, isString := pathVal.(string); isString {
					dir = pathStr
				}
			}
			filePath = fmt.Sprintf("%s/%s", dir, fileName)
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
				// If content is provided and not a specific content_type, assume it's plain text
				fileContent = []byte(content)
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
			log.Printf("UPDATE_FILE_CONTENT intent received, but file_name is missing.")
		}

	default:
		if intent == "" {
			log.Printf("Warning: No intent was extracted from the semantic output. Predicted sentence: \"%s\". Original query: \"%s\"", predictedSentence, originalQuery)
		} else {
			log.Printf("No action defined for intent: %s", intent)
		}
	}
}