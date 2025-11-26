package ner

import (
	"encoding/json"
	"strings"
)

// SemanticOutput represents the parsed JSON structure from MoE prediction
type SemanticOutput struct {
	Operation      string `json:"operation"`
	TargetResource struct {
		Type       string `json:"type"`
		Name       string `json:"name"`
		Properties struct {
			Path string `json:"path"`
		} `json:"properties"`
		Children []struct {
			Type       string                 `json:"type"`
			Name       string                 `json:"name"`
			Properties map[string]interface{} `json:"properties"`
		} `json:"children"`
	} `json:"target_resource"`
	Context struct {
		UserRole string `json:"user_role"`
	} `json:"context"`
}

// EntityType represents the type of named entity
type EntityType string

const (
	EntityTypeFileName      EntityType = "FILE_NAME"
	EntityTypeFolderName    EntityType = "FOLDER_NAME"
	EntityTypeOperation     EntityType = "OPERATION"
	EntityTypeFileType      EntityType = "FILE_TYPE"
	EntityTypeCodeType      EntityType = "CODE_TYPE"
	EntityTypeFeatureName   EntityType = "FEATURE_NAME"
	EntityTypeComponentName EntityType = "COMPONENT_NAME"
	EntityTypeOutside       EntityType = "O"
)

// Entity represents a detected named entity
type Entity struct {
	Word  string
	Type  EntityType
	Start int // word index in query
	End   int // word index in query
}

// RuleBasedNER performs rule-based named entity recognition using semantic output
type RuleBasedNER struct {
	semanticOutput *SemanticOutput
	queryWords     []string
}

// NewRuleBasedNER creates a new rule-based NER instance
func NewRuleBasedNER(query string, semanticOutputJSON string) (*RuleBasedNER, error) {
	// Parse semantic output
	var semOut SemanticOutput

	// Clean the JSON string (remove <s> and </s> tags if present)
	cleanJSON := strings.TrimSpace(semanticOutputJSON)
	cleanJSON = strings.TrimPrefix(cleanJSON, "<s>")
	cleanJSON = strings.TrimSuffix(cleanJSON, "</s>")
	cleanJSON = strings.TrimSpace(cleanJSON)

	// Try to parse JSON
	err := json.Unmarshal([]byte(cleanJSON), &semOut)
	if err != nil {
		// If parsing fails, return NER with nil semantic output (will use fallback rules)
		return &RuleBasedNER{
			semanticOutput: nil,
			queryWords:     strings.Fields(strings.ToLower(query)),
		}, nil
	}

	return &RuleBasedNER{
		semanticOutput: &semOut,
		queryWords:     strings.Fields(strings.ToLower(query)),
	}, nil
}

// ExtractEntities extracts named entities from the query using semantic context
func (r *RuleBasedNER) ExtractEntities() []Entity {
	entities := make([]Entity, 0)

	if r.semanticOutput == nil {
		// Fallback to pattern-based rules if no semantic output
		return r.extractWithPatterns()
	}

	// Extract entities based on semantic output
	targetName := strings.ToLower(r.semanticOutput.TargetResource.Name)
	targetType := r.semanticOutput.TargetResource.Type

	// Determine entity type based on resource type
	var entityType EntityType
	if strings.Contains(targetType, "File") {
		entityType = EntityTypeFileName
	} else if strings.Contains(targetType, "Folder") {
		entityType = EntityTypeFolderName
	} else {
		entityType = EntityTypeOutside
	}

	// Find the target name in the query
	if targetName != "" && targetName != "." {
		nameWords := strings.Fields(targetName)
		entities = append(entities, r.findEntityInQuery(nameWords, entityType)...)
	}

	// Extract child resource names
	for _, child := range r.semanticOutput.TargetResource.Children {
		childName := strings.ToLower(child.Name)
		if childName != "" {
			childType := EntityTypeFileName
			if strings.Contains(child.Type, "Folder") {
				childType = EntityTypeFolderName
			}
			nameWords := strings.Fields(childName)
			entities = append(entities, r.findEntityInQuery(nameWords, childType)...)
		}
	}

	// Mark operation words
	operation := strings.ToLower(r.semanticOutput.Operation)
	if operation != "" {
		opWords := []string{operation}
		entities = append(entities, r.findEntityInQuery(opWords, EntityTypeOperation)...)
	}

	return entities
}

// findEntityInQuery finds entity words in the query
func (r *RuleBasedNER) findEntityInQuery(entityWords []string, entityType EntityType) []Entity {
	entities := make([]Entity, 0)

	// Handle single-word entities
	if len(entityWords) == 1 {
		word := entityWords[0]
		for i, queryWord := range r.queryWords {
			// Match exact word or word without quotes/punctuation
			cleanQueryWord := strings.Trim(queryWord, "\"',.;:")
			if cleanQueryWord == word || queryWord == word {
				entities = append(entities, Entity{
					Word:  queryWord,
					Type:  entityType,
					Start: i,
					End:   i,
				})
			}
		}
		return entities
	}

	// Handle multi-word entities
	for i := 0; i <= len(r.queryWords)-len(entityWords); i++ {
		match := true
		for j, entityWord := range entityWords {
			cleanQueryWord := strings.Trim(r.queryWords[i+j], "\"',.;:")
			if cleanQueryWord != entityWord {
				match = false
				break
			}
		}
		if match {
			// Found multi-word entity
			for j := range entityWords {
				entities = append(entities, Entity{
					Word:  r.queryWords[i+j],
					Type:  entityType,
					Start: i + j,
					End:   i + j,
				})
			}
			break
		}
	}

	return entities
}

// extractWithPatterns uses pattern-based rules when semantic output is unavailable
func (r *RuleBasedNER) extractWithPatterns() []Entity {
	entities := make([]Entity, 0)

	// Pattern 1a: "folder <name>" or "directory <name>"
	for i := 0; i < len(r.queryWords)-1; i++ {
		if r.queryWords[i] == "folder" || r.queryWords[i] == "directory" || r.queryWords[i] == "dir" {
			entities = append(entities, Entity{
				Word:  r.queryWords[i+1],
				Type:  EntityTypeFolderName,
				Start: i + 1,
				End:   i + 1,
			})
		}
	}

	// Pattern 1b: "<name> folder" or "<name> directory" (reverse pattern)
	for i := 1; i < len(r.queryWords); i++ {
		if r.queryWords[i] == "folder" || r.queryWords[i] == "directory" || r.queryWords[i] == "dir" {
			// Check if previous word is not already a keyword
			prevWord := r.queryWords[i-1]
			if prevWord != "move" && prevWord != "to" && prevWord != "into" &&
				prevWord != "create" && prevWord != "add" && prevWord != "delete" {
				entities = append(entities, Entity{
					Word:  prevWord,
					Type:  EntityTypeFolderName,
					Start: i - 1,
					End:   i - 1,
				})
			}
		}
	}

	// Pattern 1c: word after "into" or "to" is likely a destination folder
	for i := 0; i < len(r.queryWords)-1; i++ {
		if r.queryWords[i] == "into" || r.queryWords[i] == "to" {
			nextWord := r.queryWords[i+1]
			// Only if it's not already identified and not a keyword
			if nextWord != "folder" && nextWord != "directory" && nextWord != "file" {
				entities = append(entities, Entity{
					Word:  nextWord,
					Type:  EntityTypeFolderName,
					Start: i + 1,
					End:   i + 1,
				})
			}
		}
	}

	// Pattern 1d: "<type> file" - detect file types
	fileTypes := []string{
		"html", "css", "js", "javascript", "typescript", "ts",
		"python", "py", "java", "go", "c", "cpp", "c++",
		"json", "yaml", "yml", "xml", "toml", "ini",
		"markdown", "md", "txt", "csv", "log",
		"sql", "sh", "bash", "zsh",
		"php", "ruby", "rb", "swift", "kt", "kotlin",
		"rust", "rs", "scala", "r",
	}
	for i := 0; i < len(r.queryWords)-1; i++ {
		if r.queryWords[i+1] == "file" {
			word := r.queryWords[i]
			for _, ft := range fileTypes {
				if word == ft {
					entities = append(entities, Entity{
						Word:  word,
						Type:  EntityTypeFileType,
						Start: i,
						End:   i,
					})
					break
				}
			}
		}
	}

	// Pattern 2: "file <name>" or words ending with file extensions
	for i, word := range r.queryWords {
		cleanWord := strings.Trim(word, "\"',.;:")
		if strings.Contains(cleanWord, ".") {
			// Likely a filename
			entities = append(entities, Entity{
				Word:  word,
				Type:  EntityTypeFileName,
				Start: i,
				End:   i,
			})
		}
	}

	// Pattern 3: Operation words
	operations := []string{"create", "delete", "add", "remove", "make", "list", "read", "move", "relocate", "transfer", "mv", "rename", "renaming"}
	for i, word := range r.queryWords {
		for _, op := range operations {
			if word == op {
				entities = append(entities, Entity{
					Word:  word,
					Type:  EntityTypeOperation,
					Start: i,
					End:   i,
				})
			}
		}
	}

	// Pattern 4: Code type detection (e.g., "handler code", "test code")
	codeTypeKeywords := []string{"handler", "test", "config", "util", "helper", "service", "controller", "model", "middleware"}
	for i := 0; i < len(r.queryWords); i++ {
		word := r.queryWords[i]

		// Check for "word code" pattern (e.g., "handler code")
		if i < len(r.queryWords)-1 && r.queryWords[i+1] == "code" {
			for _, keyword := range codeTypeKeywords {
				if word == keyword {
					// Mark both words as CODE_TYPE
					entities = append(entities, Entity{
						Word:  word,
						Type:  EntityTypeCodeType,
						Start: i,
						End:   i,
					})
					entities = append(entities, Entity{
						Word:  "code",
						Type:  EntityTypeCodeType,
						Start: i + 1,
						End:   i + 1,
					})
					break
				}
			}
		}
	}

	// Pattern 5: Feature detection (e.g., "jwt", "authentication", "logging")
	featureKeywords := []string{
		"jwt", "auth", "authentication", "authorization",
		"logging", "logger", "log",
		"caching", "cache",
		"database", "db",
		"validation", "validator",
		"encryption", "decrypt",
		"oauth", "session",
		"cors", "csrf",
		"rate-limiting", "throttle",
		"websocket", "ws",
		"graphql", "rest",
		"metrics", "monitoring",
		"tracing", "telemetry",
	}
	for i, word := range r.queryWords {
		cleanWord := strings.Trim(word, "\"',.;:")
		for _, feature := range featureKeywords {
			if cleanWord == feature {
				entities = append(entities, Entity{
					Word:  word,
					Type:  EntityTypeFeatureName,
					Start: i,
					End:   i,
				})
				break
			}
		}
	}

	// Pattern 6: Component detection (e.g., "webserver", "api", "service")
	componentKeywords := []string{
		"webserver", "server", "web-server",
		"api", "rest-api", "graphql-api",
		"service", "microservice",
		"handler", "controller",
		"middleware", "interceptor",
		"router", "route",
		"gateway", "proxy",
		"worker", "job",
		"queue", "stream",
		"database", "db", "store",
		"repository", "repo",
		"model", "entity",
		"client", "consumer",
		"producer", "publisher",
		"application", "app",
		"module", "component",
	}
	for i, word := range r.queryWords {
		cleanWord := strings.Trim(word, "\"',.;:")
		for _, component := range componentKeywords {
			if cleanWord == component {
				entities = append(entities, Entity{
					Word:  word,
					Type:  EntityTypeComponentName,
					Start: i,
					End:   i,
				})
				break
			}
		}
	}

	return entities
}

// GetEntityMap returns a map of word index to entity type for easy lookup
func (r *RuleBasedNER) GetEntityMap() map[int]EntityType {
	entities := r.ExtractEntities()
	entityMap := make(map[int]EntityType)

	// Initialize all words as Outside
	for i := range r.queryWords {
		entityMap[i] = EntityTypeOutside
	}

	// Set detected entities
	for _, entity := range entities {
		entityMap[entity.Start] = entity.Type
	}

	return entityMap
}
