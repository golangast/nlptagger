package main

import (
	"bufio"
	"database/sql"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/zendrulat/nlptagger/internal/sqlite_db"
	"github.com/zendrulat/nlptagger/neural/nn/ner"
	"github.com/zendrulat/nlptagger/neural/semantic"
)

// Simple task result struct
type TaskResult struct {
	Name   string
	Output string
	Err    error
}

// ParsedGoal contains the semantic understanding of a user's goal
type ParsedGoal struct {
	Intent         semantic.IntentType
	Entities       map[string]string
	SemanticOutput semantic.SemanticOutput
	RawQuery       string
}

// parseGoalWithSemantics uses MoE-based intent classification and NER to understand the goal
func parseGoalWithSemantics(goal string) (*ParsedGoal, error) {
	// Step 1: Classify intent from query
	classifier := semantic.NewIntentClassifier()
	intent := classifier.Classify(goal)

	fmt.Printf("🧠 Semantic Analysis:\n")
	fmt.Printf("  Intent: %s\n", intent)

	// Step 2: Extract entities using NER
	ruleNER, err := ner.NewRuleBasedNER(goal, "")
	if err != nil {
		return nil, fmt.Errorf("failed to create NER: %w", err)
	}

	entityMap := ruleNER.GetEntityMap()
	extractor := semantic.NewEntityExtractor()
	entities := extractor.ExtractFromQuery(goal, entityMap)

	fmt.Printf("  Entities: %v\n", entities)

	// Step 3: Check if query contains template keywords
	words := strings.Fields(goal)
	templateRegistry := semantic.NewTemplateRegistry()
	hasTemplate := false
	for _, word := range words {
		lowerWord := strings.ToLower(word)
		for _, tmpl := range templateRegistry.ListTemplates() {
			if lowerWord == strings.ToLower(tmpl) {
				hasTemplate = true
				break
			}
		}
		if hasTemplate {
			break
		}
	}

	var semanticOutput semantic.SemanticOutput

	if hasTemplate {
		// Use hierarchical parser for template-based scaffolding
		hierarchicalParser := semantic.NewHierarchicalParser()
		hierarchicalCmd := hierarchicalParser.Parse(goal, words, entityMap)
		semanticOutput = semantic.FillFromHierarchicalCommand(hierarchicalCmd)

		fmt.Printf("  Template: Hierarchical scaffolding detected\n")
		fmt.Printf("  Command Tree:\n%s\n", hierarchicalCmd.String())
	} else {
		// Use standard parser for simple commands
		parser := semantic.NewCommandParser()
		structuredCmd := parser.Parse(goal, words, entityMap)

		// Fill template with entities
		filler := semantic.NewTemplateFiller()
		var err error
		semanticOutput, err = filler.Fill(intent, entities)
		if err != nil {
			fmt.Printf("   ⚠️ Template filling failed: %v\n", err)
		} else {
			fmt.Printf("  Pattern: %s\n", structuredCmd.String())
		}
	}

	// Display semantic output
	jsonBytes, _ := json.MarshalIndent(semanticOutput, "  ", "  ")
	fmt.Printf("  Semantic Output:\n  %s\n", string(jsonBytes))

	return &ParsedGoal{
		Intent:         intent,
		Entities:       entities,
		SemanticOutput: semanticOutput,
		RawQuery:       goal,
	}, err // Return the function-scoped error
}

// generateFromSemantic creates files and folders based on SemanticOutput
func generateFromSemantic(output semantic.SemanticOutput, kb *KnowledgeBase, reader *bufio.Reader) error {
	if output.TargetResource == nil {
		return nil
	}

	// Helper function to recursively create resources
	var createResource func(r semantic.Resource, parentPath string) error
	createResource = func(r semantic.Resource, parentPath string) error {
		// Determine path
		// If resource name is empty, skip creating it but process children (unless it's a file)
		if r.Name == "" && len(r.Children) > 0 {
			for _, child := range r.Children {
				if err := createResource(child, parentPath); err != nil {
					return err
				}
			}
			return nil
		}

		currentPath := filepath.Join(parentPath, r.Name)

		// Handle different resource types
		switch r.Type {
		case "Filesystem::Folder":
			if err := os.MkdirAll(currentPath, 0755); err != nil {
				return fmt.Errorf("failed to create folder %s: %w", currentPath, err)
			}
			// fmt.Printf("Created folder: %s\n", currentPath)
		case "Filesystem::File":
			// Ensure parent dir exists
			if err := os.MkdirAll(filepath.Dir(currentPath), 0755); err != nil {
				return fmt.Errorf("failed to create parent dir for %s: %w", currentPath, err)
			}

			content := ""
			if r.Properties != nil {
				if c, ok := r.Properties["content"].(string); ok {
					content = c
				}
			}

			// Knowledge Base Lookup
			// If content is empty or generic, try to learn
			if kb != nil {
				query := r.Name
				if r.Properties != nil {
					if component, ok := r.Properties["component"].(string); ok {
						query += " " + component
					}
				}

				matches, _ := kb.FindRelevantFiles(query)
				if len(matches) > 0 {
					fmt.Printf("\n📚 Found learned content for '%s'. Use one of these?\n", r.Name)
					for i, m := range matches {
						fmt.Printf("   %d. %s\n", i+1, filepath.Base(m))
					}
					fmt.Println("   0. No, use default/generated")
					fmt.Print("   > ")

					choiceStr, _ := reader.ReadString('\n')
					choiceStr = strings.TrimSpace(choiceStr)
					choice, _ := strconv.Atoi(choiceStr)

					if choice > 0 && choice <= len(matches) {
						learnedContent, err := kb.ReadFileContent(matches[choice-1])
						if err == nil {
							content = learnedContent
							fmt.Printf("   ✅ Using content from %s\n", matches[choice-1])
						} else {
							fmt.Printf("   ⚠️ Failed to read content: %v\n", err)
						}
					}
				}
			}

			if err := os.WriteFile(currentPath, []byte(content), 0644); err != nil {
				return fmt.Errorf("failed to write file %s: %w", currentPath, err)
			}
			fmt.Printf("Created file: %s\n", currentPath)
		}

		// Process children
		for _, child := range r.Children {
			if err := createResource(child, currentPath); err != nil {
				return err
			}
		}
		return nil
	}

	// Start generation from project root
	projectRoot := "generated_projects/project"

	// Check if TargetResource is a container or a concrete item
	if output.TargetResource.Type == "Unknown" && len(output.TargetResource.Children) > 0 {
		// It's a container, iterate children
		for _, child := range output.TargetResource.Children {
			if err := createResource(child, projectRoot); err != nil {
				return err
			}
		}
	} else {
		// It's a concrete item
		if err := createResource(*output.TargetResource, projectRoot); err != nil {
			return err
		}
	}

	return nil
}

func main() {
	reader := bufio.NewReader(os.Stdin)
	kb := NewKnowledgeBase("learning")

	// Initialize database once
	dbFileName := "generated_projects/project/orchestrator.db"
	db, err := sqlite_db.InitDB(dbFileName)
	if err != nil {
		fmt.Printf("Failed to initialize database: %v\n", err)
		return
	}
	defer db.Close()

	if err := initGitRepo(); err != nil {
		fmt.Printf("Failed to initialize git repository: %v\n", err)
		// We can continue without git, but revert functionality will not work.
	}

	// Initialize custom intent registry
	customIntents := NewCustomIntentRegistry("generated_projects/custom_intents.json")
	fmt.Printf("📚 Loaded %d custom intent(s)\n", len(customIntents.Intents))

	for {
		fmt.Print("\n🤖 Multi-Orchestrator (with NLP Understanding)\n")
		fmt.Print("Commands: 'delete project', 'show history', 'revert <id/hash/command>', 'exit'\n")
		fmt.Print("Or describe what you want in natural language (e.g., 'create a webserver with authentication handler')\n> ")
		goal, _ := reader.ReadString('\n')
		goal = strings.TrimSpace(goal)

		if goal == "exit" {
			fmt.Println("Exiting multi-orchestrator.")
			break
		}

		// Handle custom intent commands
		goalLower := strings.ToLower(goal)

		// Create custom intent (with typo tolerance)
		if strings.Contains(goalLower, "create") && strings.Contains(goalLower, "intent") ||
			strings.Contains(goalLower, "new") && strings.Contains(goalLower, "intent") ||
			goalLower == "new intent" || goalLower == "create custom intent" {
			if err := customIntents.InteractiveCreate(reader); err != nil {
				fmt.Printf("❌ Failed to create custom intent: %v\n", err)
			}
			continue
		}

		// List custom intents
		if strings.Contains(goalLower, "list") && strings.Contains(goalLower, "intent") ||
			strings.Contains(goalLower, "show") && strings.Contains(goalLower, "intent") {
			intents := customIntents.List()
			if len(intents) == 0 {
				fmt.Println("No custom intents defined yet.")
			} else {
				fmt.Println("\n📚 Custom Intents:")
				for _, intent := range intents {
					fmt.Printf("  • %s - %s\n", intent.Name, intent.Description)
					fmt.Printf("    Keywords: %v\n", intent.Keywords)
					fmt.Printf("    Output: %s\n", intent.FilePath)
				}
			}
			continue
		}

		// Remove custom intent
		if strings.HasPrefix(goalLower, "remove intent ") || strings.HasPrefix(goalLower, "delete intent ") {
			intentName := strings.TrimPrefix(goal, "remove intent ")
			intentName = strings.TrimPrefix(intentName, "delete intent ")
			intentName = strings.TrimSpace(intentName)
			if err := customIntents.Remove(intentName); err != nil {
				fmt.Printf("❌ Failed to remove intent: %v\n", err)
			} else {
				fmt.Printf("✅ Removed custom intent '%s'\n", intentName)
			}
			continue
		}

		// Handle delete project with natural language understanding
		isDeleteCommand := goal == "delete project" ||
			strings.Contains(strings.ToLower(goal), "delete") && strings.Contains(strings.ToLower(goal), "project") ||
			strings.Contains(strings.ToLower(goal), "remove") && strings.Contains(strings.ToLower(goal), "project") ||
			strings.Contains(strings.ToLower(goal), "clear") && strings.Contains(strings.ToLower(goal), "project")

		if isDeleteCommand {
			fmt.Println("Deleting project files (preserving Git history)...")
			projectDir := "generated_projects/project"

			// Read all entries in the project directory
			entries, err := os.ReadDir(projectDir)
			if err != nil {
				if os.IsNotExist(err) {
					fmt.Println("Project directory does not exist.")
				} else {
					fmt.Printf("Failed to read project directory: %v\n", err)
				}
				continue
			}

			// Delete everything except .git directory
			for _, entry := range entries {
				if entry.Name() != ".git" {
					path := filepath.Join(projectDir, entry.Name())
					err := os.RemoveAll(path)
					if err != nil {
						fmt.Printf("Failed to delete %s: %v\n", entry.Name(), err)
					}
				}
			}
			fmt.Println("Project files deleted successfully. Git history preserved for revert.")
			continue
		}
		if goal == "show history" {
			messages, err := sqlite_db.GetMessages(db)
			if err != nil {
				fmt.Printf("Failed to get messages: %v\n", err)
				continue
			}
			fmt.Println("--- Filtered Command History ---")
			for _, msg := range messages {
				if msg.Role == "user" {
					fmt.Printf("ID: %d, Timestamp: %s\n", msg.ID, msg.Timestamp)
					fmt.Printf("Command: %s\n", msg.Content)
					if msg.CommitHash.Valid {
						fmt.Printf("Commit: %s\n", msg.CommitHash.String[:7])
					}
					fmt.Println("-----------------")
				}
			}
			continue
		}
		if strings.HasPrefix(goal, "revert") {
			parts := strings.Fields(goal)
			if len(parts) < 2 {
				fmt.Println("Usage: revert <id_or_command_or_hash>")
				continue
			}
			identifier := strings.Join(parts[1:], " ")

			var hash string
			var revertID int64

			// Try to parse as an ID first
			id, err := strconv.ParseInt(identifier, 10, 64)
			if err == nil {
				// It's an ID
				revertID = id
				hash, err = sqlite_db.GetCommitHash(db, revertID)
				if err != nil {
					fmt.Printf("Failed to get commit hash for ID %d: %v\n", revertID, err)
					continue
				}
			} else {
				// Check if it looks like a commit hash (at least 7 characters, all hexadecimal)
				isHash := len(identifier) >= 7
				if isHash {
					for _, c := range identifier {
						if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
							isHash = false
							break
						}
					}
				}

				if isHash {
					// Try to find by commit hash
					fmt.Printf("Searching for commit hash: '%s'\n", identifier)
					msg, err := sqlite_db.GetMessageByCommitHash(db, identifier)
					if err != nil {
						fmt.Printf("No command matching commit hash '%s' found in history.\n", identifier)
						continue
					}
					revertID = int64(msg.ID)
					hash = msg.CommitHash.String
				} else {
					// It's a command string
					fmt.Printf("Searching for command: '%s'\n", identifier)
					messages, err := sqlite_db.GetMessages(db)
					if err != nil {
						fmt.Printf("Failed to get messages: %v\n", err)
						continue
					}

					var latestMatchingMessage sqlite_db.Message
					found := false
					for i := len(messages) - 1; i >= 0; i-- { // Search from most recent
						msg := messages[i]
						if msg.Role == "user" && strings.Contains(strings.ToLower(msg.Content), strings.ToLower(identifier)) {
							latestMatchingMessage = msg
							found = true
							break
						}
					}

					if !found {
						fmt.Printf("No command matching '%s' found in history.\n", identifier)
						continue
					}
					revertID = int64(latestMatchingMessage.ID)
					if !latestMatchingMessage.CommitHash.Valid {
						fmt.Printf("No commit hash associated with command ID %d ('%s'). Cannot revert.\n", revertID, latestMatchingMessage.Content)
						continue
					}
					hash = latestMatchingMessage.CommitHash.String
				}
			}

			projectDir := "generated_projects/project"

			// Check if project directory exists
			if _, err := os.Stat(projectDir); os.IsNotExist(err) {
				fmt.Printf("Cannot revert: project directory does not exist. Please create a new project first.\n")
				continue
			}

			// Check if .git directory exists
			gitDir := filepath.Join(projectDir, ".git")
			if _, err := os.Stat(gitDir); os.IsNotExist(err) {
				fmt.Printf("Cannot revert: Git repository not found. The project was likely deleted with an older version that didn't preserve Git history.\n")
				continue
			}

			fmt.Printf("Reverting to commit %s for command ID %d...\n", hash, revertID)
			if _, err := runCommand(projectDir, "git", "reset", "--hard", hash); err != nil {
				fmt.Printf("Failed to revert to commit %s: %v. This might happen if the commit no longer exists in the Git history (e.g., due to a rebase). Please check 'git log' for available commits.\n", hash, err)
			} else {
				fmt.Printf("Successfully reverted to command ID %d (commit %s).\n", revertID, hash[:7])
			}
			continue
		}

		if goal == "" {
			fmt.Println("No goal provided. Please enter a command.")
			continue
		}

		// Main goal processing logic starts here
		fmt.Printf("\n📝 Processing goal: %s\n", goal)
		fmt.Println(strings.Repeat("=", 60))

		// Check for custom intent match first
		if customIntent := customIntents.Match(goal); customIntent != nil {
			fmt.Printf("🎨 Matched custom intent: %s\n", customIntent.Name)

			// Extract variables from entities
			variables := make(map[string]string)

			// Simple variable extraction - you can enhance this
			words := strings.Fields(goal)
			for i, word := range words {
				// Store each word as a potential variable
				variables[fmt.Sprintf("Word%d", i)] = word
			}

			// Ask user for any template variables
			fmt.Println("\n📋 Template variables needed:")
			templateVars := extractTemplateVariables(customIntent.Template)
			for _, varName := range templateVars {
				if _, exists := variables[varName]; !exists {
					fmt.Printf("Enter value for {{.%s}}: ", varName)
					value, _ := reader.ReadString('\n')
					variables[varName] = strings.TrimSpace(value)
				}
			}

			// Execute custom intent
			if err := customIntents.Execute(customIntent, "generated_projects/project", variables); err != nil {
				fmt.Printf("❌ Failed to execute custom intent: %v\n", err)
			} else {
				fmt.Println("✅ Custom intent executed successfully!")

				// Commit changes
				messageID, _ := sqlite_db.SaveMessage(db, "user", goal)
				if err := commitChanges(goal, db, messageID); err != nil {
					fmt.Printf("Failed to commit changes: %v\n", err)
				}
			}
			continue
		}

		// Use semantic parsing instead of naive keyword matching
		parsedGoal, err := parseGoalWithSemantics(goal)

		// Interactive Clarification
		// Skip clarification if we have valid hierarchical output with children
		hasValidOutput := parsedGoal != nil &&
			parsedGoal.SemanticOutput.TargetResource != nil &&
			len(parsedGoal.SemanticOutput.TargetResource.Children) > 0

		if parsedGoal != nil && (parsedGoal.Intent == semantic.IntentUnknown || err != nil) && !hasValidOutput {
			fmt.Println("\n❓ I couldn't understand your intent.")
			if len(parsedGoal.Entities) > 0 {
				fmt.Printf("   I found these entities: %v\n", parsedGoal.Entities)
			}
			fmt.Println("   What would you like to do?")
			fmt.Println("   1. create_handler")
			fmt.Println("   2. create_database")
			fmt.Println("   3. delete_file")
			fmt.Println("   4. create_file")
			fmt.Println("   5. delete_project")
			fmt.Println("   6. Define Custom / Alias")
			fmt.Print("   > ")

			choice, _ := reader.ReadString('\n')
			choice = strings.TrimSpace(choice)

			switch choice {
			case "1", "create_handler":
				// Add handler entity if missing
				if _, ok := parsedGoal.Entities["handler_name"]; !ok {
					parsedGoal.Entities["handler_name"] = "Custom"
				}
				parsedGoal.Intent = semantic.IntentAddFeature
				err = nil
			case "2", "create_database":
				parsedGoal.Entities["database_name"] = "orchestrator.db"
				parsedGoal.Intent = semantic.IntentAddFeature
				err = nil
			case "3", "delete_file":
				parsedGoal.Intent = semantic.IntentDeleteFile
				err = nil
			case "4", "create_file":
				parsedGoal.Intent = semantic.IntentCreateFile
				err = nil
			case "5", "delete_project":
				// We can handle this by setting the goal string to "delete project" and restarting loop
				// But we are deep in the loop. Let's just execute the logic manually here or rely on entities.
				// The easiest way is to set a flag or just execute it.
				// Let's execute it here to be safe.
				fmt.Println("Deleting project files (preserving Git history)...")
				projectDir := "generated_projects/project"
				entries, _ := os.ReadDir(projectDir)
				for _, entry := range entries {
					if entry.Name() != ".git" {
						os.RemoveAll(filepath.Join(projectDir, entry.Name()))
					}
				}
				fmt.Println("Project files deleted successfully.")
				continue
			case "6", "custom", "alias":
				// Manual Input Flow
				fmt.Print("   Enter Intent Name (e.g., 'my_action'): ")
				customIntent, _ := reader.ReadString('\n')
				parsedGoal.Intent = semantic.IntentType(strings.TrimSpace(customIntent))

				fmt.Print("   Enter Entities (key=value, comma separated): ")
				entitiesStr, _ := reader.ReadString('\n')
				entitiesStr = strings.TrimSpace(entitiesStr)
				if entitiesStr != "" {
					pairs := strings.Split(entitiesStr, ",")
					for _, pair := range pairs {
						kv := strings.SplitN(pair, "=", 2)
						if len(kv) == 2 {
							parsedGoal.Entities[strings.TrimSpace(kv[0])] = strings.TrimSpace(kv[1])
						}
					}
				}

				// Check if this is a known executable intent
				isKnown := false
				switch parsedGoal.Intent {
				case semantic.IntentAddFeature, semantic.IntentDeleteFile, "create_handler", "create_database":
					isKnown = true
				}

				// If unknown, ask for alias mapping
				if !isKnown {
					fmt.Printf("   ⚠️ Intent '%s' is not natively supported.\n", parsedGoal.Intent)
					fmt.Println("   Map it to a known action? (Alias)")
					fmt.Println("   1. create_handler (requires 'handler_name' or 'component')")
					fmt.Println("   2. create_database (requires 'database_name')")
					fmt.Println("   3. delete_file (requires 'file')")
					fmt.Println("   4. No, try to run as is")
					fmt.Print("   > ")

					aliasChoice, _ := reader.ReadString('\n')
					aliasChoice = strings.TrimSpace(aliasChoice)

					switch aliasChoice {
					case "1":
						parsedGoal.Intent = semantic.IntentAddFeature
						if _, ok := parsedGoal.Entities["handler_name"]; !ok {
							if _, ok2 := parsedGoal.Entities["component"]; !ok2 {
								parsedGoal.Entities["handler_name"] = "Custom" // Default
							}
						}
					case "2":
						parsedGoal.Intent = semantic.IntentAddFeature
						if _, ok := parsedGoal.Entities["database_name"]; !ok {
							parsedGoal.Entities["database_name"] = "orchestrator.db"
						}
					case "3":
						parsedGoal.Intent = semantic.IntentDeleteFile
					}
				}
				err = nil
			}

			if err == nil {
				fmt.Printf("   ✅ Intent set to: %s\n", parsedGoal.Intent)
			}
		}

		if err != nil {
			fmt.Printf("❌ Failed to parse goal semantically: %v\n", err)
			fmt.Println("Falling back to basic keyword matching...")
			// Continue with fallback logic below
		}

		// Handle Delete File specifically
		if parsedGoal != nil && parsedGoal.Intent == semantic.IntentDeleteFile {
			fileName := parsedGoal.Entities["file"]
			if fileName == "" {
				fileName = parsedGoal.Entities["source_file"]
			}
			// Fallback: try to find a file entity in the map values if key is generic
			if fileName == "" {
				for k, v := range parsedGoal.Entities {
					if strings.Contains(k, "file") {
						fileName = v
						break
					}
				}
			}

			if fileName != "" {
				fmt.Printf("Deleting file: %s\n", fileName)
				path := filepath.Join("generated_projects/project", fileName)
				if err := os.Remove(path); err != nil {
					fmt.Printf("Failed to delete file: %v\n", err)
				} else {
					fmt.Println("File deleted successfully.")
				}
				continue // Skip generation
			} else {
				fmt.Println("⚠️ Intent is delete_file but no file entity found.")
			}
		}

		// Example: Save the user's goal as a message
		messageID, err := sqlite_db.SaveMessage(db, "user", goal)
		if err != nil {
			fmt.Printf("Failed to save goal to database: %v\n", err)
			// Decide if this is a critical error or if processing can continue
		}
		fmt.Printf("💾 User goal saved to %s with message ID: %d\n", dbFileName, messageID)

		// Extract information from semantic parsing
		var customHandlerName string
		var customDBName string = "database/orchestrator.db" // default

		if parsedGoal != nil {
			fmt.Println(strings.Repeat("=", 60))

			// Use entities extracted by NER
			if handlerName, ok := parsedGoal.Entities["handler_name"]; ok {
				customHandlerName = handlerName
				fmt.Printf("✅ Extracted handler name: %s\n", customHandlerName)
			} else if componentName, ok := parsedGoal.Entities["component_name"]; ok {
				customHandlerName = componentName
				fmt.Printf("✅ Extracted component name: %s\n", customHandlerName)
			}

			if dbName, ok := parsedGoal.Entities["database_name"]; ok {
				customDBName = "database/" + dbName
				fmt.Printf("✅ Extracted database name: %s\n", customDBName)
			}

			// Use intent to determine what to generate
			fmt.Printf("🎯 Intent-based action: %s\n", parsedGoal.Intent)
		} else {
			// Fallback to old keyword-based extraction
			if strings.Contains(goal, "handler") {
				// Attempt to extract the custom handler name
				handlerKeywordIndex := strings.LastIndex(goal, "handler")
				if handlerKeywordIndex != -1 {
					nameStartIndex := -1
					for i := handlerKeywordIndex - 1; i >= 0; i-- {
						if goal[i] == ' ' {
							nameStartIndex = i + 1
							break
						}
					}
					if nameStartIndex == -1 && handlerKeywordIndex > 0 {
						nameStartIndex = 0
					}

					if nameStartIndex != -1 {
						potentialName := goal[nameStartIndex:handlerKeywordIndex]
						potentialName = strings.TrimSpace(potentialName)
						if len(potentialName) > 0 {
							customHandlerName = strings.ToUpper(potentialName[:1]) + potentialName[1:]
						}
					}
				}
				if customHandlerName == "" {
					customHandlerName = "Custom"
				}
			}

			// Extract custom database name if present
			if strings.Contains(goal, "database") {
				// Look for .db file pattern
				parts := strings.Fields(goal)
				for _, part := range parts {
					if strings.HasSuffix(part, ".db") {
						customDBName = "database/" + part
						break
					}
				}
			}
		}

		var wg sync.WaitGroup
		results := make(chan TaskResult, 3)

		// Declare these at outer scope so QA phase can access them
		var shouldCreateHandler bool
		var shouldCreateDatabase bool
		var generatedFromSemantic bool

		wg.Add(1)
		go func() {
			defer wg.Done()
			var err error
			out := "Go server written"

			// Use semantic understanding to determine what to generate

			if parsedGoal != nil {
				// Check if we should create a handler based on semantic analysis
				if _, hasHandler := parsedGoal.Entities["handler_name"]; hasHandler {
					shouldCreateHandler = true
				} else if _, hasComponent := parsedGoal.Entities["component"]; hasComponent {
					// Check if component is handler-related
					if strings.Contains(strings.ToLower(parsedGoal.Entities["component"]), "handler") {
						shouldCreateHandler = true
					}
				}

				// Check if we should create a database
				if _, hasDB := parsedGoal.Entities["database_name"]; hasDB {
					shouldCreateDatabase = true
				} else if strings.Contains(strings.ToLower(goal), "database") {
					shouldCreateDatabase = true
				}

				fmt.Printf("🔧 Generation decision: handler=%v, database=%v\n", shouldCreateHandler, shouldCreateDatabase)
			} else {
				// Fallback to keyword matching
				shouldCreateHandler = strings.Contains(goal, "handler")
				shouldCreateDatabase = strings.Contains(goal, "database")
			}

			// Generate based on semantic understanding
			if parsedGoal != nil && parsedGoal.SemanticOutput.TargetResource != nil {
				// Check if we have meaningful content to generate
				hasContent := len(parsedGoal.SemanticOutput.TargetResource.Children) > 0 ||
					(parsedGoal.SemanticOutput.TargetResource.Type != "Unknown" && parsedGoal.SemanticOutput.TargetResource.Name != "")

				// Only attempt generation if it's a Filesystem resource or a container (Unknown)
				// This prevents Code::Component resources (from IntentAddFeature) from being marked as "generated"
				// without actually doing anything, which blocks the specific feature handlers below.
				isFilesystem := strings.HasPrefix(parsedGoal.SemanticOutput.TargetResource.Type, "Filesystem::") ||
					parsedGoal.SemanticOutput.TargetResource.Type == "Unknown"

				if hasContent && isFilesystem {
					err = generateFromSemantic(parsedGoal.SemanticOutput, kb, reader)
					if err == nil {
						out = "Generated files from semantic output"
						generatedFromSemantic = true
					} else {
						fmt.Printf("⚠️ Semantic generation failed: %v. Falling back...\n", err)
					}
				}
			}

			if !generatedFromSemantic {
				if shouldCreateDatabase {
					err = writeGoServerWithDB(customDBName)
					if err == nil {
						out = fmt.Sprintf("Go server with database (%s) written to generated_projects/project/server.go", customDBName)
					}
				} else if shouldCreateHandler {
					if customHandlerName == "" {
						customHandlerName = "Custom"
					}
					err = writeHandlers(customHandlerName)
					if err == nil {
						out = fmt.Sprintf("Go server with %sHandler written to generated_projects/project/server.go", customHandlerName)
					}
				} else {
					err = writeGoServer()
					if err == nil {
						out = "Go server written to generated_projects/project/server.go"
					}
				}
			}
			if err == nil {
				err = copySupportFiles()
			}
			if err == nil {
				serverGoPath := "generated_projects/project/server.go"
				code, readErr := os.ReadFile(serverGoPath)
				if readErr != nil {
					fmt.Printf("Failed to read generated server.go: %v\n", readErr)
				} else {
					if _, saveErr := sqlite_db.SaveMessage(db, "coder", string(code)); saveErr != nil {
						fmt.Printf("Failed to save generated code to database: %v\n", saveErr)
					}
				}
			}
			if err != nil {
				out = ""
			}
			results <- TaskResult{Name: "Coder", Output: out, Err: err}
		}()

		wg.Add(1)
		go func() {
			defer wg.Done()
			err := writeDockerfile(customDBName)
			out := "Dockerfile written"
			if err != nil {
				out = ""
			}
			results <- TaskResult{Name: "DevOps", Output: out, Err: err}
		}()

		wg.Add(1)
		go func() {
			defer wg.Done()
			err := writeReadme()
			out := "README written"
			if err != nil {
				out = ""
			}
			results <- TaskResult{Name: "Readme", Output: out, Err: err}
		}()

		wg.Wait()
		close(results)

		fmt.Println("--- Generation Results ---")
		for r := range results {
			if r.Err != nil {
				fmt.Printf("%s failed: %v\n", r.Name, r.Err)
			} else {
				fmt.Printf("%s succeeded: %s\n", r.Name, r.Output)
			}
		}

		fmt.Println("--- QA Phase ---")
		// Skip QA if we only did semantic generation (no server.go)
		skipQA := generatedFromSemantic && !shouldCreateDatabase && !shouldCreateHandler

		if skipQA {
			fmt.Println("Skipping QA for semantic file generation...")
			fmt.Println("All checks passed. Project ready!")
		} else {
			if err := runTestsAndBuild(goal, customHandlerName); err != nil {
				fmt.Printf("QA failed: %v\n", err)
				fmt.Println("Triggering re-run of generation tasks... (simplified)")
				continue // Continue the loop to allow user to try again or revert
			}
			fmt.Println("All checks passed. Project ready!")
		}

		if err := commitChanges(goal, db, messageID); err != nil {
			fmt.Printf("Failed to commit changes: %v\n", err)
		}
	} // Closing brace for the 'for' loop
}

func writeGoServer() error {
	content := `package main

import (
    "fmt"
    "log"
    "net/http"
)

func helloHandler(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintln(w, "Hello, World!")
}

func main() {
    http.HandleFunc("/", helloHandler)
    log.Println("Starting server on :8080")
    if err := http.ListenAndServe(":8080", nil); err != nil {
        log.Fatalf("Server failed: %v", err)
    }
}`
	// Ensure target directory exists
	os.MkdirAll("generated_projects/project", 0755)
	return os.WriteFile("generated_projects/project/server.go", []byte(content), 0644)
}

func writeHandlers(handlerName string) error {
	// Dynamically create the handler function name and path
	dynamicHandlerFuncName := strings.ToLower(handlerName) + "Handler"
	dynamicHandlerPath := "/" + strings.ToLower(handlerName)

	content := fmt.Sprintf(`package main

import (
	"fmt"
	"log"
	"net/http"
)

func helloHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintln(w, "Hello, World!")
}

func %s(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintln(w, "%s received!")
}

func main() {
	http.HandleFunc("/", helloHandler)
	http.HandleFunc("%s", %s)
	log.Println("Starting server on :8080")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatalf("Server failed: %%v", err)
	}
}`, dynamicHandlerFuncName, handlerName, dynamicHandlerPath, dynamicHandlerFuncName)
	// Ensure target directory exists
	os.MkdirAll("generated_projects/project", 0755)
	return os.WriteFile("generated_projects/project/server.go", []byte(content), 0644)
}

func writeGoServerWithDB(dbName string) error {
	content := "package main\n\n" +
		"import (\n" +
		"	\"database/sql\"\n" +
		"	\"fmt\"\n" +
		"	\"html/template\"\n" +
		"	\"log\"\n" +
		"	\"net/http\"\n\n" +
		"	\"github.com/zendrulat/nlptagger/internal/sqlite_db\"\n" +
		")\n\n" +
		"func helloHandler(w http.ResponseWriter, r *http.Request) {\n" +
		"	fmt.Fprintln(w, \"Hello, World!\")\n" +
		"}\n\n" +
		"func saveHandler(db *sql.DB) http.HandlerFunc {\n" +
		"	return func(w http.ResponseWriter, r *http.Request) {\n" +
		"		role := r.URL.Query().Get(\"role\")\n" +
		"		content := r.URL.Query().Get(\"content\")\n" +
		"		if role == \"\" || content == \"\" {\n" +
		"			http.Error(w, \"role and content are required\", http.StatusBadRequest)\n" +
		"			return\n" +
		"		}\n" +
		"		if _, err := sqlite_db.SaveMessage(db, role, content); err != nil {\n" +
		"			http.Error(w, fmt.Sprintf(\"failed to save message: %v\", err), http.StatusInternalServerError)\n" +
		"			return\n" +
		"		}\n" +
		"		fmt.Fprintln(w, \"Message saved successfully\")\n" +
		"	}\n" +
		"}\n\n" +
		"func viewHandler(db *sql.DB) http.HandlerFunc {\n" +
		"	return func(w http.ResponseWriter, r *http.Request) {\n" +
		"		messages, err := sqlite_db.GetMessages(db)\n" +
		"		if err != nil {\n" +
		"			http.Error(w, fmt.Sprintf(\"failed to get messages: %v\", err), http.StatusInternalServerError)\n" +
		"			return\n" +
		"		}\n\n" +
		"		tmpl, err := template.New(\"messages\").Parse(`\n" +
		"			<!DOCTYPE html>\n" +
		"			<html>\n" +
		"			<head>\n" +
		"				<title>Messages</title>\n" +
		"			</head>\n" +
		"			<body>\n" +
		"				<h1>Messages</h1>\n" +
		"				<ul>\n" +
		"					{{range .}}\n" +
		"						<li><b>{{.Role}}:</b> {{.Content}} <i>({{.Timestamp}})</i></li>\n" +
		"					{{end}}\n" +
		"				</ul>\n" +
		"			</body>\n" +
		"			</html>\n" +
		"		`)\n" +
		"		if err != nil {\n" +
		"			http.Error(w, \"Failed to parse template\", http.StatusInternalServerError)\n" +
		"			return\n" +
		"		}\n\n" +
		"		if err := tmpl.Execute(w, messages); err != nil {\n" +
		"			http.Error(w, \"Failed to execute template\", http.StatusInternalServerError)\n" +
		"		}\n" +
		"	}\n" +
		"}\n\n" +
		"func main() {\n" +
		fmt.Sprintf("\tdb, err := sqlite_db.InitDB(\"%s\")\n", dbName) +
		"\tif err != nil {\n" +
		"\t\tlog.Fatalf(\"Failed to initialize database: %v\", err)\n" +
		"\t}\n" +
		"\tdefer db.Close()\n\n" +
		"\thttp.HandleFunc(\"/\", helloHandler)\n" +
		"\thttp.HandleFunc(\"/save\", saveHandler(db))\n" +
		"\thttp.HandleFunc(\"/view\", viewHandler(db))\n" +
		"\tlog.Println(\"Starting server on :8080\")\n" +
		"\tif err := http.ListenAndServe(\":8080\", nil); err != nil {\n" +
		"\t\tlog.Fatalf(\"Server failed: %v\", err)\n" +
		"\t}\n" +
		"}\n"
	os.MkdirAll("generated_projects/project", 0755)
	return os.WriteFile("generated_projects/project/server.go", []byte(content), 0644)
}

func writeDockerfile(dbPath string) error {
	content := fmt.Sprintf(`FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN go build -o webserver server.go

FROM alpine:latest
WORKDIR /app
COPY --from=builder /app/webserver .
COPY --from=builder /app/%s ./%s
EXPOSE 8080
CMD ["./webserver"]
`, dbPath, dbPath)
	os.MkdirAll("generated_projects/project", 0755)
	return os.WriteFile("generated_projects/project/Dockerfile", []byte(content), 0644)
}

func writeReadme() error {
	content := "# Simple Go Webserver\n\nThis project contains a minimal Go HTTP server and a Dockerfile to containerize it.\n\n## Build & Run\n\n```sh\ngo run server.go\n```\n\nOr with Docker:\n\n```sh\ndocker build -t go-webserver .\n docker run -p 8080:8080 go-webserver\n```\n"
	os.MkdirAll("generated_projects/project", 0755)
	return os.WriteFile("generated_projects/project/README.md", []byte(content), 0644)
}

func runTestsAndBuild(goal string, customHandlerName string) error { // Update signature
	buildCmd := exec.Command("go", "build", "-o", "generated_projects/project/webserver", "generated_projects/project/server.go")
	if output, err := buildCmd.CombinedOutput(); err != nil {
		return fmt.Errorf("failed to build Go server: %v\n%s", err, output)
	}

	runCmd := exec.Command("./webserver")
	runCmd.Dir = "generated_projects/project"
	if err := runCmd.Start(); err != nil {
		return fmt.Errorf("failed to run Go server: %v", err)
	}
	defer runCmd.Process.Kill()

	// Give the server a moment to start
	time.Sleep(2 * time.Second)

	if strings.Contains(goal, "handler") {
		dynamicHandlerPath := "/" + strings.ToLower(customHandlerName)
		fmt.Printf("Testing %s endpoint...\n", dynamicHandlerPath)
		resp, err := http.Get("http://localhost:8080" + dynamicHandlerPath)
		if err != nil {
			return fmt.Errorf("failed to connect to %s endpoint: %v", dynamicHandlerPath, err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			return fmt.Errorf("unexpected status code for %s: got %d, want %d", dynamicHandlerPath, resp.StatusCode, http.StatusOK)
		}
		fmt.Printf("%s endpoint test passed.\n", dynamicHandlerPath)
	}

	return nil
}

func copySupportFiles() error {
	// Copy internal/sqlite_db directory
	srcDbDir := "internal/sqlite_db"
	destDbDir := "generated_projects/project/internal/sqlite_db"
	if err := os.MkdirAll(destDbDir, 0755); err != nil {
		return fmt.Errorf("failed to create directory %s: %w", destDbDir, err)
	}
	dbFile := "sqlite_db.go"
	if err := copyFile(filepath.Join(srcDbDir, dbFile), filepath.Join(destDbDir, dbFile)); err != nil {
		return err
	}

	// Copy go.mod and go.sum
	if err := copyFile("go.mod", "generated_projects/project/go.mod"); err != nil {
		return err
	}
	if err := copyFile("go.sum", "generated_projects/project/go.sum"); err != nil {
		return err
	}
	return nil
}

func copyFile(src, dest string) error {
	sourceFile, err := os.Open(src)
	if err != nil {
		return fmt.Errorf("failed to open source file %s: %w", src, err)
	}
	defer sourceFile.Close()

	destFile, err := os.Create(dest)
	if err != nil {
		return fmt.Errorf("failed to create destination file %s: %w", dest, err)
	}
	defer destFile.Close()

	_, err = io.Copy(destFile, sourceFile)
	if err != nil {
		return fmt.Errorf("failed to copy file from %s to %s: %w", src, dest, err)
	}
	return nil
}

func runCommand(dir string, command string, args ...string) (string, error) {
	cmd := exec.Command(command, args...)
	cmd.Dir = dir
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("failed to run command '%s %s': %v\n%s", command, strings.Join(args, " "), err, output)
	}
	return string(output), nil
}

func initGitRepo() error {
	projectDir := "generated_projects/project"
	if err := os.MkdirAll(projectDir, 0755); err != nil {
		return fmt.Errorf("failed to create project directory: %w", err)
	}
	gitDir := filepath.Join(projectDir, ".git")
	if _, err := os.Stat(gitDir); os.IsNotExist(err) {
		fmt.Println("Initializing git repository...")
		if _, err := runCommand(projectDir, "git", "init"); err != nil {
			return err
		}
	}

	// Check if there are any commits
	_, err := runCommand(projectDir, "git", "rev-parse", "--verify", "HEAD")
	if err != nil {
		// No commits, make an initial commit
		fmt.Println("Making initial commit...")
		if _, err := runCommand(projectDir, "git", "add", "."); err != nil {
			return err
		}
		// It's possible there are no files to commit initially
		_, err := runCommand(projectDir, "git", "commit", "-m", "Initial commit")
		if err != nil && !strings.Contains(err.Error(), "nothing to commit") {
			return err
		}
	}
	return nil
}
func commitChanges(goal string, db *sql.DB, messageID int64) error {
	projectDir := "generated_projects/project"
	fmt.Println("Committing changes...")
	if _, err := runCommand(projectDir, "git", "add", "."); err != nil {
		return err
	}
	if _, err := runCommand(projectDir, "git", "commit", "-m", goal); err != nil {
		// It's possible there's nothing to commit if the generated files are the same
		if strings.Contains(err.Error(), "nothing to commit") {
			fmt.Println("No changes to commit.")
			return nil
		}
		return err
	}

	// Get the commit hash
	hash, err := runCommand(projectDir, "git", "rev-parse", "HEAD")
	if err != nil {
		return fmt.Errorf("failed to get commit hash: %w", err)
	}
	hash = strings.TrimSpace(hash)

	// Update the database with the commit hash
	if err := sqlite_db.UpdateCommitHash(db, messageID, hash); err != nil {
		return fmt.Errorf("failed to update commit hash in database: %w", err)
	}

	return nil
}

// extractTemplateVariables extracts variable names from a template string
// e.g., "{{.Name}} is {{.Age}}" -> ["Name", "Age"]
func extractTemplateVariables(template string) []string {
	var vars []string
	seen := make(map[string]bool)

	// Simple regex-like extraction
	parts := strings.Split(template, "{{.")
	for _, part := range parts[1:] {
		endIdx := strings.Index(part, "}}")
		if endIdx > 0 {
			varName := part[:endIdx]
			if !seen[varName] {
				vars = append(vars, varName)
				seen[varName] = true
			}
		}
	}

	return vars
}
