package intent

import (
	"fmt"
	"math"
	"regexp"
	"strings"

	"github.com/golangast/nlptagger/commands"
	"github.com/golangast/nlptagger/neural/nn/g"
	"github.com/golangast/nlptagger/neural/nn/semanticrole"
	"github.com/golangast/nlptagger/neural/nn/semanticrole/train_bilstm"
	"github.com/golangast/nlptagger/neural/nnu/train"
	"github.com/golangast/nlptagger/tagger"
	"github.com/golangast/nlptagger/tagger/tag"
)

// TrainingData struct
type IntentClassifier struct {
	// ... other fields
	SemanticRoleModel *semanticrole.SemanticRoleModel
}

// Define maps for actions and objects
var actionMap = map[string]string{
	"add":        "add",        // Add
	"generate":   "generate",   // Generate
	"create":     "create",     // Create
	"delete":     "delete",     // Delete
	"remove":     "remove",     // Remove
	"convert":    "convert",    // Convert
	"read":       "read",       // Read
	"write":      "write",      // Write
	"modify":     "modify",     // Modify
	"update":     "update",     // Update
	"analyze":    "analyze",    // Analyze
	"start":      "start",      // Start
	"stop":       "stop",       // Stop
	"restart":    "restart",    // Restart
	"deploy":     "deploy",     // Deploy
	"configure":  "configure",  // Configure
	"backup":     "backup",     // Backup
	"restore":    "restore",    // Restore
	"search":     "search",     // Search
	"list":       "list",       // List
	"move":       "move",       // Move
	"copy":       "copy",       // Copy
	"monitor":    "monitor",    // Monitor
	"install":    "install",    // install
	"uninstall":  "uninstall",  // uninstall
	"debug":      "debug",      // debug
	"connect":    "connect",    // connect
	"disconnect": "disconnect", // disconnect
	"export":     "export",     //export
	"import":     "import",     //import
}

var objectMap = map[string]string{
	"webserver":   "webserver",   //Webserver
	"file":        "file",        //File
	"document":    "document",    //Document
	"data":        "data",        //Data
	"script":      "script",      //Script
	"image":       "image",       //Image
	"database":    "database",    //Database
	"service":     "service",     //Service
	"application": "application", //Application
	"container":   "container",   //Container
	"network":     "network",     //Network
	"user":        "user",        //user
	"permission":  "permission",  //permission
	"setting":     "setting",     //setting
	"log":         "log",         //log
	"process":     "process",     //process
	"connection":  "connection",  //connection
	"backup":      "backup",      //Backup
	"report":      "report",      //Report
	"security":    "security",    //Security
}

// Phrase Tag Map: Maps specific phrase tags to corresponding intents.
// Phrase tags are derived from dependency analysis and provide a higher level of
// context, allowing for more accurate intent detection.
// Example of how a phrase tag is mapped to an intent: `verbPhrase:generate_a_webserver`: "generate a webserver"
var phraseTagMap = map[string]string{
	`verbPhrase:generate_a_webserver`:        "generate a webserver",      // Generate a webserver
	`command:create_file`:                    "create a file",             // Create file
	`command:delete_file`:                    "delete a file",             // Delete file
	`command:convert_file_format`:            "convert a file",            // Convert file format
	`verbPhrase:read_a_json_file`:            "read a json file",          // Read JSON file
	`verbPhrase:remove_empty_lines`:          "remove empty lines",        // Remove empty lines
	`command:overwrite_file`:                 "overwrite a file",          // Overwrite file
	`command:extract_monetary_values`:        "extract monetary values",   // Extract monetary values
	`command:automate_image_processing`:      "automate image processing", // Automate image processing
	`command:start_service`:                  "start a service",           // Start service
	`command:stop_service`:                   "stop a service",            // Stop service
	`command:restart_service`:                "restart a service",         // Restart service
	`verbPhrase:configure_the_(.*)_settings`: "configure settings",        // Configure settings
	`command:search_(.*)_logs`:               "search in the logs",        // Search in the logs
	`command:monitor_(.*)_status`:            "monitor the status",        // Monitor the status
	`command:install_(.*)_package`:           "install a package",         // install a package
	`command:uninstall_(.*)_package`:         "uninstall a package",       // uninstall a package
	`command:connect_(.*)_network`:           "connect to a network",      // connect to a network
}

func LoadRoleData(filePath string) ([]semanticrole.SentenceRoleData, error) {
	return semanticrole.LoadRoleData(filePath)
}

// ProcessCommand: The entry point for processing a command.
// This function takes a raw command string, a word-to-vector index, and context relevance
// information, then performs dependency analysis, intent interpretation, and returns the detected intent.
//
// Parameters:
// - command: The raw command string.
// - index: The word-to-vector index used for context vector calculation.
// - c: Contextual relevance information.
//
// Returns the detected intent string and any errors encountered.
func (ic *IntentClassifier) ProcessCommand(command string, index map[string][]float64, c train.ContextRelevance) (string, error) {

	tag := tagger.Tagging(command)
	trainingdata, err := g.LoadTrainingData("datas/training_data.json")
	if err != nil {
		fmt.Println("Error loading training data:", err)
		return "", err
	}

	// 1. Obtain tokens from the command (replace with your actual tokenization logic)
	tokens := strings.Split(command, " ")
	// 2. Train or load the Word2Vec embedding model
	if ic.SemanticRoleModel == nil {
		ic.SemanticRoleModel, err = semanticrole.NewSemanticRoleModel(
			"word2vec_model.gob",
			"bilstm_model.gob",
			"role_map.gob",
		)
		if err != nil {
			fmt.Println("failed to load semantic role model: %w", err)
		}
	}
	// 3. Predict roles
	roles, err := ic.SemanticRoleModel.PredictRoles(tokens)
	if err != nil {
		fmt.Println("failed to predict roles: %w", err)
	}

	// 4. Intent Interpretation (This is where the new logic goes)
	intent, err := ic.InterpretIntent(tag, trainingdata, index, command, c, roles)
	if err != nil {
		return "", err
	}

	fmt.Println("Tokens:", tokens)
	fmt.Println("Roles:", roles)

	return intent, nil
}

// InterpretIntent: Interprets the intent behind a given command by analyzing its
// dependency structure, comparing it against training data, and considering contextual relevance.
//
// Parameters:
//   - dependencyAnalysis: The dependency analysis results of the command, including tokens,
//     NER tags, POS tags, phrase tags, and dependency relationships.
//   - trainingData: A slice of training data used for comparing the command against known intents.
//   - index: A map used for word embedding lookup, which is used in similarity calculations.
//   - command: The original command string.
//   - c: Contextual relevance information, including iteration, nearest context word, similarity,
//     and contextual relevance.
//
// Returns:
// - A string describing the detected intent, or an error if one occurs.
//
// InterpretIntent: Interprets the intent behind a given command by analyzing its structure, training data, roles and context.
func (i *IntentClassifier) InterpretIntent(dependencyAnalysis tag.Tag, trainingData []g.TrainingData, index map[string][]float64, command string, c train.ContextRelevance, roles []string) (string, error) {
	var intent string
	var actionDetected bool = false
	var objectDetected bool = false
	// Variables to store extracted action and object
	var extractedAction string
	var extractedObject string
	// Variable to store the main verb found in the sentence.
	var verbToken string

	commandVector := g.CalculateContextVector(command, index)

	var mostSimilarIndex int
	var maxSimilarity float64 = -1.0
	// The maximum similarity between the command and training sentences.

	// Iterate through training data to find most similar sentence
	for i, data := range trainingData {
		// Create a training vector from the first context string of the data.Context array
		trainingVector := g.CalculateContextVector(data.Sentence, index)

		// Calculate the similarity between the command vector and the training vector
		vecA := NewVecDense(len(commandVector))
		vecA.data = commandVector
		vecB := NewVecDense(len(trainingVector))
		vecB.data = trainingVector
		similarity := CosineSimilarityVecDense(vecA, vecB)

		// Update if this similarity is higher than the previous max
		if similarity > maxSimilarity {
			maxSimilarity = similarity
			mostSimilarIndex = i
		}
	}
	train_bilstm.Train_bilstm()
	// Assuming you have functions to load the model and role map

	fmt.Println("Most similar training data vector:", trainingData[mostSimilarIndex].Sentence) // Outputting the sentence that is most similar.

	fmt.Println("Similarity: ", maxSimilarity) // Outputting the similarity value between command and the most similar sentence.

	// Filter sentences by similarity
	similarSentences := g.FilterBySentenceSimilarity(command, trainingData, index)

	fmt.Println("Contextual Information:")
	fmt.Println("  Iteration:", c.Iteration)
	fmt.Println("  Nearest Context Word:", c.NearestContextWord)
	fmt.Println("  Similarity:", c.Similarity)
	fmt.Println("  Contextual Relevance:", c.ContextualRelevance)

	for i := 0; i < len(dependencyAnalysis.Tokens); i++ {
		// Iterates through each token in the analyzed command and determines if
		// the current token could be the action or object of the intent. It also uses
		// any found phrase tags or verb tags to match against the intent.
		var depTag tag.Dependency
		token := dependencyAnalysis.Tokens[i]
		nerTag := dependencyAnalysis.NerTag[i]
		posTag := dependencyAnalysis.PosTag[i]
		phraseTag := dependencyAnalysis.PhraseTag[i]
		if len(dependencyAnalysis.Dependencies) > 2 {
			depTag = dependencyAnalysis.Dependencies[i]
		} else {
			depTag = dependencyAnalysis.Dependencies[0]
		}

		fmt.Printf("Word: %-15s Ner: %-15s Pos: %-15s Phrase: %-15s Dep: %-15v\n",
			token, nerTag, posTag, phraseTag, depTag)
		// Prioritize phrase tags - these are more specific
		if phraseTag != "" {
			if intentFromPhrase, ok := phraseTagMap[phraseTag]; ok {
				intent = intentFromPhrase
				actionDetected = true
				objectDetected = true
				if intent == "generate a webserver" {
					extractedAction = "generate"
					extractedObject = "webserver"
				}
				if intent == "create a file" {
					commands.Createfile()
				}
				continue // Skip other checks if a phrase is matched
			} else {
				// Check for regular expression matches
				for pattern, intentFromPattern := range phraseTagMap {
					matched, _ := regexp.MatchString(pattern, phraseTag)
					if matched {
						intent = intentFromPattern
						actionDetected = true
						objectDetected = true
						continue
					}
				}
				intent = "Phrase Detected: " + phraseTag

			}

		} // Check for actions using NER tags
		// Check for actions using NER tags
		if nerTag == "ACTION" && extractedAction == "" { // Only assign if not already assigned
			if intentFromAction, ok := actionMap[token]; ok {
				extractedAction = intentFromAction // Store the action token
			} else {
				extractedAction = "unknown"

			}
			intent = fmt.Sprintf("%s the %s", actionMap[extractedAction], extractedObject) // Example: create the file
			if intent == "create the file" {
				commands.Createfile()
			}
		} else if extractedAction != "" {
			intent = fmt.Sprintf("%s", actionMap[extractedAction]) // Example: create
			if intent == "create" {
				commands.Createfile()
			}

			verbToken = token
			actionDetected = true
			intent = extractedAction
		}

		// Check for objects using NER tags
		if nerTag == "OBJECT_TYPE" && extractedObject == "" { // Only assign if not already assigned
			extractedObject = token // Store the object token
			objectDetected = true
			intent = objectMap[token] // Store the object token
		}

		if nerTag == "NAME" {
			// Consider "NAME" as a potential object
			extractedObject = token // Store the object token
			objectDetected = true
			intent = "About " + token
		}

		// Check for specific object names, object names are always objects
		if nerTag == "OBJECT_NAME" {
			objectDetected = true // An object name was detected
			intent = "Intent: Object Name detected: " + token
			extractedObject = token
		} // Use dependency relationships to extract action and object
		// Use dependency relationships
		if posTag == "VERB" && verbToken == "" {
			// If it is a verb, and the verbToken is not assigned, assign it
			verbToken = token
			// verbToken variable is set to the current token, which is a verb.

		}
		if depTag.Dep == "nsubj" && posTag == "NOUN" {
			// Find the verb that this subject relates to.
			for j := 0; j < len(dependencyAnalysis.Dependencies); j++ {
				if dependencyAnalysis.DepRelationsTag[j] == "ROOT" {
					// Check if this ROOT is related to the nsubj by comparing indices.
					if depTag.Dependent == j {
						if _, ok := actionMap[dependencyAnalysis.Tokens[j]]; ok {
							extractedAction = dependencyAnalysis.Tokens[j]
						}
					}
				} else {
					if depTag.Head == j {
						if _, ok := actionMap[dependencyAnalysis.Tokens[j]]; ok {
							extractedAction = dependencyAnalysis.Tokens[j]
						}
					}
					if depTag.Head == i {
						extractedAction = dependencyAnalysis.Tokens[j]
					}
				}
			}

		} else if depTag.Dep == "dobj" {
			// If there is a direct object, see if it is a known object.
			if _, ok := objectMap[token]; ok {
				extractedObject = token
			}
		}
		// Determine intent based on extracted action and object from
		// dependencies
		// Check if we have extracted an action and object from dependencies
		if extractedAction != "" && extractedObject != "" {
			intent = fmt.Sprintf("%s the %s", actionMap[extractedAction], extractedObject) // Example: create the file
		} else if extractedAction != "" {
			intent = fmt.Sprintf("%s", actionMap[extractedAction]) // Example: create
		} else if extractedObject != "" {
			intent = fmt.Sprintf("About the %s", extractedObject) // Example: About the file
		} // Using training data similarity to improve intent
		// If we have similar data, use it to improve the intent
		if len(similarSentences) > 0 {
			// If there are similar sentences, they can improve the intent detection
			if intent == "" { // No intent was set yet
				// If no intent has been determined, use the context of the most similar sentence as intent
				intent = similarSentences[0].Context // No intent was set, use similar sentence as intent
			} else {
				intent = "The command is similar to " + similarSentences[0].Context + ". " + intent // Add similar sentence info to existing intent
			}

		}
		// Context awareness
		// If the ContextualRelevance value is higher than a certain threshold,
		// this indicates a significant contextual connection.
		if c.ContextualRelevance > 0.8 {
			intent = "Considering the context: " + intent
		} else if c.ContextualRelevance > 0.5 && c.ContextualRelevance <= 0.8 {
			intent = "It is possible that:" + intent
		}
		//Unclear intent
		if !actionDetected && !objectDetected && intent == "" { // If no action or object detected, suggest using them
			intent = fmt.Sprintf("Unclear Intent: Could not determine action or object. Try using verbs like: %s and nouns like: %s", strings.Join(mapsKeys(actionMap), ", "), strings.Join(mapsKeys(objectMap), ","))
		} else if !actionDetected && !objectDetected { // If there is intent, but no action or object
			intent = "Unclear Intent: It was not possible to clearly determine the action or object, but, considering: " + intent //If there is intent, show it
		} else if actionDetected && !objectDetected {
			// If we have an action but no object
			suggestedObjects := getSuggestedObjects(extractedAction)
			// Contextual Inference
			// Check if the most similar training data vector has an object related to this action
			similarSentence := trainingData[mostSimilarIndex].Sentence
			var inferredObject string
			for obj := range objectMap {
				if strings.Contains(strings.ToLower(similarSentence), obj) {
					inferredObject = obj
					break
				}
			}
			//If a object is not inferred, check if the nearest context word is a known object
			if inferredObject == "" && c.NearestContextWord != "" {
				if _, ok := objectMap[c.NearestContextWord]; ok {
					inferredObject = c.NearestContextWord
				}
			}

			if inferredObject != "" {
				intent = fmt.Sprintf("%s the %s based on the context.", actionMap[extractedAction], inferredObject)
				objectDetected = true
			} else {
				// Fallback to suggestion if inference fails
				intent = fmt.Sprintf("Unclear Intent: Action Detected but no Object. Try using objects like: %s", strings.Join(suggestedObjects, ", ")) // Suggest objects
			}

		} else if !actionDetected && objectDetected {

			suggestedActions := getSuggestedActions(extractedObject)                                                                              // Get suggested actions based on the object
			intent = fmt.Sprintf("Unclear Intent: Object Detected but no Action. Try using verbs like: %s", strings.Join(suggestedActions, ", ")) // Suggest actions
		}

	}
	fmt.Printf("Roles: %-15v\n", roles)

	//Unclear intent
	if !actionDetected && !objectDetected && intent == "" { // If no action or object detected, suggest using them
		intent = fmt.Sprintf("Unclear Intent: Could not determine action or object. Try using verbs like: %s and nouns like: %s", strings.Join(mapsKeys(actionMap), ", "), strings.Join(mapsKeys(objectMap), ","))
	} else if !actionDetected && !objectDetected { // If there is intent, but no action or object
		intent = "Unclear Intent: It was not possible to clearly determine the action or object, but, considering: " + intent //If there is intent, show it
	} else if actionDetected && !objectDetected {
		// If we have an action but no object
		suggestedObjects := getSuggestedObjects(extractedAction)
		intent = fmt.Sprintf("Unclear Intent: Action Detected but no Object. Try using objects like: %s", strings.Join(suggestedObjects, ", ")) // Suggest objects
	} else if !actionDetected && objectDetected {
		// If we have an object but no action
		suggestedActions := getSuggestedActions(extractedObject) // Get suggested actions based on the object
		intent = fmt.Sprintf("Unclear Intent: Object Detected but no Action. Try using verbs like: %s", strings.Join(suggestedActions, ", "))
	}

	// Context awareness
	return "The intent is to " + intent + ".", nil // Add final format
}

// VecDense represents a dense vector.
type VecDense struct {
	data []float64
	N    int
}

// NewVecDense creates a new VecDense of length n with all values set to zero.
func NewVecDense(n int) *VecDense {
	return &VecDense{
		data: make([]float64, n),
		N:    n,
	}
}

// At returns the value at the given index.
func (v *VecDense) At(i int) float64 {
	if i < 0 || i >= v.N {
		panic(fmt.Sprintf("index out of bounds: %d (length: %d)", i, v.N))
	}
	return v.data[i]
}

// Set sets the value at the given index.
func (v *VecDense) Set(i int, value float64) {
	if i < 0 || i >= v.N {
		panic(fmt.Sprintf("index out of bounds: %d (length: %d)", i, v.N))
	}
	v.data[i] = value
}

// CosineSimilarityVecDense calculates the cosine similarity between two VecDense vectors.
func CosineSimilarityVecDense(v1, v2 *VecDense) float64 {
	// Check if vectors have different lengths
	if v1.N != v2.N {
		return 0.0 // Or handle the error as you prefer
	}

	dotProduct := 0.0
	magV1 := 0.0
	magV2 := 0.0

	if v1.N != 0 {
		magV1 := 0.0
		magV2 := 0.0
		for i := 0; i < v1.N; i++ {
			dotProduct += v1.At(i) * v2.At(i)
			magV1 += math.Pow(v1.At(i), 2)
			magV2 += math.Pow(v2.At(i), 2)
		}
		magV1 = math.Sqrt(magV1)
	}
	return dotProduct / (magV1 * magV2)
}

// mapsKeys retrieves all keys from a map and returns them as a string slice.
// It's useful for diagnostic or informative outputs where map keys need to be listed.
func mapsKeys[K comparable, V any](m map[K]V) []string {
	keys := make([]string, len(m))
	var i int
	for k := range m {
		keys[i] = fmt.Sprintf("%v", k)
		i++
	}
	return keys
}

// getSuggestedObjects returns a list of objects that are typically associated with
// the given action. It aids in providing suggestions when an object is not explicitly specified.
func getSuggestedObjects(action string) []string {
	switch action {
	case "create":
		return []string{"file", "document", "database", "script"}
	case "generate":
		return []string{"webserver", "script", "document", "file", "data"}
	case "delete", "remove":
		return []string{"file", "data", "user", "permission", "log"}
	case "start", "stop", "restart":
		return []string{"service", "process", "application", "webserver", "container"}
	case "configure":
		return []string{"setting", "user", "service", "network", "application"}
	case "backup", "restore":
		return []string{"database", "file", "data"}
	case "search", "list":
		return []string{"log", "file", "user", "process", "data"}
	case "add":
		return []string{"user", "permission", "data"}
	case "deploy":
		return []string{"application", "container", "service"}
	case "convert":
		return []string{"file", "document", "data"}
	case "read", "write", "modify":
		return []string{"file", "document", "data", "setting"}
	case "move", "copy":
		return []string{"file", "data"}
	case "analyze":
		return []string{"log", "data"}
	default:
		return []string{"file", "data", "service", "application", "user"}
	}
}

// getSuggestedActions returns a list of actions that are typically performed on
// the given object. This function is useful for providing suggestions when an action is not explicitly defined.
func getSuggestedActions(object string) []string {
	switch object {
	case "file", "document", "script", "data":
		return []string{"create", "read", "write", "modify", "delete", "remove", "convert", "move", "copy", "search", "analyze"}
	case "webserver", "service", "application", "container", "process":
		return []string{"start", "stop", "restart", "configure", "deploy", "list", "analyze", "monitor", "debug"}
	case "user", "permission":
		return []string{"add", "remove", "modify", "configure", "list"}
	case "database":
		return []string{"create", "delete", "backup", "restore", "configure", "list"}
	case "network", "setting":
		return []string{"configure", "modify", "list"}
	case "log":
		return []string{"search", "analyze", "read", "delete", "remove"}
	case "backup", "report", "security":
		return []string{"generate", "create", "export", "import", "analyze"}
	default:
		return []string{"add", "generate", "create", "delete", "remove", "convert", "read", "write", "modify", "update", "analyze", "start", "stop", "restart", "deploy", "configure", "backup", "restore", "search", "list", "move", "copy"}
	}
}
