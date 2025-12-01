package semantic

import (
	"strings"

	"github.com/zendrulat/nlptagger/neural/nn/ner"
)

// EntityExtractor extracts entities from NER results
type EntityExtractor struct{}

// NewEntityExtractor creates a new entity extractor
func NewEntityExtractor() *EntityExtractor {
	return &EntityExtractor{}
}

// Extract converts NER entity map to a simple key-value map
// entityMap: map[wordIndex]entityType
// words: the tokenized words from the query
func (ee *EntityExtractor) Extract(words []string, entityMap map[int]ner.EntityType) map[string]string {
	entities := make(map[string]string)

	// Find directional keywords (for move operations)
	directionalIdx := -1
	for i, word := range words {
		lowerWord := strings.ToLower(word)
		if lowerWord == "into" || lowerWord == "to" {
			directionalIdx = i
			break
		}
	}

	// Track counts for multi-entity handling
	folderCount := 0
	fileCount := 0

	// Prepositions and articles to filter out
	stopWords := map[string]bool{
		"of": true, "the": true, "a": true, "an": true,
		"at": true, "by": true, "for": true, "from": true,
		"on": true, "with": true, "in": true,
	}

	// Check if there is a database component in the query
	hasDatabaseComponent := false
	for i, word := range words {
		if entityMap[i] == "COMPONENT_NAME" || entityMap[i] == "FEATURE_NAME" {
			lower := strings.ToLower(word)
			if lower == "database" || lower == "db" || lower == "sqlite" {
				hasDatabaseComponent = true
				break
			}
		}
	}

	for i, word := range words {
		entityType, exists := entityMap[i]
		if !exists || entityType == "O" {
			continue
		}

		// Skip directional keywords - they should not be entity values
		lowerWord := strings.ToLower(word)
		if lowerWord == "into" || lowerWord == "to" {
			continue
		}

		// Skip prepositions and articles
		if stopWords[lowerWord] {
			continue
		}

		switch entityType {
		case "FOLDER_NAME":
			folderCount++
			if directionalIdx != -1 {
				// Use position relative to directional keyword
				if i < directionalIdx {
					entities["source_folder"] = word
					if folderCount == 1 {
						entities["folder"] = word // backward compatibility
					}
				} else {
					entities["destination_folder"] = word
				}
			} else {
				// No directional keyword: first is source, second is destination
				if folderCount == 1 {
					entities["source_folder"] = word
					entities["folder"] = word // backward compatibility
				} else if folderCount == 2 {
					entities["destination_folder"] = word
				} else {
					// Fallback for single folder
					entities["folder"] = word
				}
			}
		case "FILE_NAME":
			fileCount++
			if directionalIdx != -1 && i < directionalIdx {
				entities["source_file"] = word
				entities["file"] = word // backward compatibility
			} else if directionalIdx != -1 && i > directionalIdx {
				// File after directional keyword is destination
				entities["destination_file"] = word
			} else {
				// No directional keyword: first file is source, second is destination
				if fileCount == 1 {
					entities["source_file"] = word
					entities["file"] = word // backward compatibility
				} else if fileCount == 2 {
					entities["destination_file"] = word
				} else {
					// Fallback for single file
					entities["file"] = word
				}
			}

			// Check for database name
			if hasDatabaseComponent || strings.HasSuffix(word, ".db") {
				entities["database_name"] = word
			}
		case "FILE_TYPE":
			entities["file_type"] = word
		case "PATH":
			entities["path"] = word
		case "CODE_TYPE":
			// Combine multi-word code types (e.g., "handler code")
			if existingCodeType, ok := entities["code_type"]; ok {
				entities["code_type"] = existingCodeType + " " + word
			} else {
				entities["code_type"] = word
			}
		case "FEATURE_NAME":
			entities["feature"] = word
		case "COMPONENT_NAME":
			entities["component"] = word
		}
	}

	return entities
}

// ExtractFromQuery is a convenience method that splits the query and extracts
func (ee *EntityExtractor) ExtractFromQuery(query string, entityMap map[int]ner.EntityType) map[string]string {
	words := strings.Fields(query)
	return ee.Extract(words, entityMap)
}
