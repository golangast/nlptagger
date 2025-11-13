package parser

import (
	"fmt"
	"log"
	"strings"

	"github.com/zendrulat/nlptagger/neural/semantic"
)

// Helper to get resource name from token and type
func getResourceName(tokens []string, nerTags []string, idx int, objType string) string {
	if idx < len(tokens) {
		// If explicitly tagged as NAME
		if nerTags[idx] == "NAME" {
			return tokens[idx]
		}
		// If the next token is tagged as NAME
		if idx+1 < len(tokens) && nerTags[idx+1] == "NAME" {
			return tokens[idx+1]
		}

		// Fallback: If current token is an OBJECT_TYPE, assume the next token is the name
		if nerTags[idx] == "OBJECT_TYPE" && idx+1 < len(tokens) {
			// Ensure the next token is not another OBJECT_TYPE or PREPOSITION or DETERMINER
			if nerTags[idx+1] != "OBJECT_TYPE" && nerTags[idx+1] != "PREPOSITION" && nerTags[idx+1] != "DETERMINER" {
				return tokens[idx+1]
			}
		}
	}
	return ""
}

// ParsingRule defines a rule for mapping tokens and NER tags to semantic output.
type ParsingRule struct {
	Name    string
	Pattern []string // A sequence of NER tags or specific tokens to match
	Action  func(tokens, posTags, nerTags []string, i int, output *semantic.SemanticOutput, 
		fileResource **semantic.Resource, folderResource **semantic.Resource, 
		webserverResource **semantic.Resource, lastFolderResource **semantic.Resource, 
		lastProcessedResource **semantic.Resource, expectingDependencyTarget *bool, 
		expectingDependencySource *bool, dependencyType *string) (bool, int, error) // Function to apply if pattern matches, returns (applied, tokensConsumed, error)
}

// ParsingRuleEngine manages and applies a set of parsing rules.
type ParsingRuleEngine struct {
	rules []ParsingRule
}

// NewParsingRuleEngine creates a new ParsingRuleEngine.
func NewParsingRuleEngine() *ParsingRuleEngine {
	return &ParsingRuleEngine{}
}

// RegisterRule adds a new parsing rule to the engine.
func (pre *ParsingRuleEngine) RegisterRule(rule ParsingRule) {
	pre.rules = append(pre.rules, rule)
}

// ApplyRules attempts to apply the registered rules to the given tokens and tags.
// It returns true if a rule was applied, the number of tokens consumed by the rule, and an error if one occurred.
func (pre *ParsingRuleEngine) ApplyRules(tokens, posTags, nerTags []string, i int, output *semantic.SemanticOutput, 
	fileResource **semantic.Resource, folderResource **semantic.Resource, 
	webserverResource **semantic.Resource, lastFolderResource **semantic.Resource, 
	lastProcessedResource **semantic.Resource, expectingDependencyTarget *bool, 
	expectingDependencySource *bool, dependencyType *string) (bool, int, error) {
	for _, rule := range pre.rules {
		if pre.matchPattern(tokens, nerTags, posTags, i, rule.Pattern) {
			applied, tokensConsumed, err := rule.Action(tokens, posTags, nerTags, i, output, 
				fileResource, folderResource, webserverResource, lastFolderResource, 
				lastProcessedResource, expectingDependencyTarget, expectingDependencySource, dependencyType)
			if err != nil {
				return false, 0, fmt.Errorf("rule '%s' failed: %w", rule.Name, err)
			}
			if applied {
				return true, tokensConsumed, nil
			}
		}
	}
	return false, 0, nil
}

// matchPattern checks if the given NER tags (or tokens) match the rule's pattern starting from index i.
func (pre *ParsingRuleEngine) matchPattern(tokens, nerTags, posTags []string, i int, pattern []string) bool {
	if len(pattern) == 0 {
		return false
	}
	if i+len(pattern) > len(tokens) {
		return false
	}

	for j, p := range pattern {
		// Check for NER tag, POS tag, or direct token match
		if nerTags[i+j] != p && posTags[i+j] != p && tokens[i+j] != p {
			return false
		}
	}
	return true
}

// RegisterDefaultParsingRules registers a set of default parsing rules.
func (pre *ParsingRuleEngine) RegisterDefaultParsingRules() {
	// Rule for "add" command
	pre.RegisterRule(ParsingRule{
		Name:    "AddCommand",
		Pattern: []string{"VB"}, // Match a verb
		Action: func(tokens, posTags, nerTags []string, i int, output *semantic.SemanticOutput,
			fileResource **semantic.Resource, folderResource **semantic.Resource,
			webserverResource **semantic.Resource, lastFolderResource **semantic.Resource,
			lastProcessedResource **semantic.Resource, expectingDependencyTarget *bool,
			expectingDependencySource *bool, dependencyType *string) (bool, int, error) {
			if strings.ToLower(tokens[i]) == "add" || strings.ToLower(tokens[i]) == "make" || strings.ToLower(tokens[i]) == "create" {
				output.Operation = "CREATE"
				return true, 1, nil
			}
			return false, 0, nil
		},
	})

	// Rule for "move" command
	pre.RegisterRule(ParsingRule{
		Name:    "MoveCommand",
		Pattern: []string{"COMMAND", "NAME", "PREPOSITION", "NAME"}, // "move", "a/b.txt", "to", "c/d.txt"
		Action: func(tokens, posTags, nerTags []string, i int, output *semantic.SemanticOutput,
			fileResource **semantic.Resource, folderResource **semantic.Resource,
			webserverResource **semantic.Resource, lastFolderResource **semantic.Resource,
			lastProcessedResource **semantic.Resource, expectingDependencyTarget *bool,
			expectingDependencySource *bool, dependencyType *string) (bool, int, error) {
			if strings.ToLower(tokens[i]) == "move" || strings.ToLower(tokens[i]) == "mv" {
				sourcePath := tokens[i+1]
				destinationPath := tokens[i+3]

				*fileResource = &semantic.Resource{
					Type:        "Filesystem::File",
					Name:        sourcePath,
					Destination: destinationPath,
				}
				output.TargetResource = *fileResource
				output.Operation = "MOVE"
				log.Printf("MoveCommand rule applied. Source: %s, Destination: %s", sourcePath, destinationPath)
				return true, 4, nil // Consumed "move", "source", "to", "destination"
			}
			return false, 0, nil
		},
	})

	// Rule for "create folder <name>"
	pre.RegisterRule(ParsingRule{
		Name:    "CreateFolder",
		Pattern: []string{"OBJECT_TYPE", "NN"}, // "folder", "jim"
		Action: func(tokens, posTags, nerTags []string, i int, output *semantic.SemanticOutput,
			fileResource **semantic.Resource, folderResource **semantic.Resource,
			webserverResource **semantic.Resource, lastFolderResource **semantic.Resource,
			lastProcessedResource **semantic.Resource, expectingDependencyTarget *bool,
			expectingDependencySource *bool, dependencyType *string) (bool, int, error) {
			if strings.ToLower(tokens[i]) == "folder" {
				folderName := getResourceName(tokens, nerTags, i, "folder")
				if folderName != "" {
					*folderResource = &semantic.Resource{Type: "Filesystem::Folder", Name: folderName, Properties: map[string]interface{}{"permissions": 0755}}
					*lastFolderResource = *folderResource
					*folderResource = &semantic.Resource{Type: "Filesystem::Folder", Name: folderName, Properties: map[string]interface{}{"permissions": 0755}}
					*lastFolderResource = *folderResource
					output.TargetResource = *folderResource // Set as target for now, will be adjusted by parent-child rules
					log.Printf("CreateFolder rule applied. folderName: %s, folderResource: %+v", folderName, **folderResource)
					return true, 2, nil                   // Consumed "folder" and "name"
				}
			}
			return false, 0, nil
		},
	})

	// Rule for "file <name>"
	pre.RegisterRule(ParsingRule{
		Name:    "CreateFile",
		Pattern: []string{"OBJECT_TYPE", "NAME"}, // "file", "jack.go"
		Action: func(tokens, posTags, nerTags []string, i int, output *semantic.SemanticOutput,
			fileResource **semantic.Resource, folderResource **semantic.Resource,
			webserverResource **semantic.Resource, lastFolderResource **semantic.Resource,
			lastProcessedResource **semantic.Resource, expectingDependencyTarget *bool,
			expectingDependencySource *bool, dependencyType *string) (bool, int, error) {
			if strings.ToLower(tokens[i]) == "file" {
				fileName := getResourceName(tokens, nerTags, i, "file")
				if fileName != "" {
					log.Printf("Inside CreateFile rule: fileName=%s", fileName)
					*fileResource = &semantic.Resource{Type: "Filesystem::File", Name: fileName}
					log.Printf("Inside CreateFile rule: *fileResource.Type=%s", (*fileResource).Type)
					output.TargetResource = *fileResource // Set as target for now
					log.Printf("Inside CreateFile rule: output.TargetResource.Type=%s", output.TargetResource.Type)
					output.TargetResource.Type = "Filesystem::File"
					output.Operation = "CREATE"
					log.Printf("CreateFile rule applied. fileName: %s, fileResource: %+v", fileName, **fileResource)
					return true, 2, nil // Consumed "file" and "name"
				}
			}
			return false, 0, nil
		},
	})

	// Rule for "in folder <name>" - this will need to be more sophisticated to handle nesting
	// For now, a simplified version that sets the directory of the last created file/folder
	pre.RegisterRule(ParsingRule{
		Name:    "InFolder",
		Pattern: []string{"PREPOSITION", "OBJECT_TYPE", "NN"}, // "in", "folder", "jim"
		Action: func(tokens, posTags, nerTags []string, i int, output *semantic.SemanticOutput,
			fileResource **semantic.Resource, folderResource **semantic.Resource,
			webserverResource **semantic.Resource, lastFolderResource **semantic.Resource,
			lastProcessedResource **semantic.Resource, expectingDependencyTarget *bool,
			expectingDependencySource *bool, dependencyType *string) (bool, int, error) {
			if strings.ToLower(tokens[i]) == "in" && strings.ToLower(tokens[i+1]) == "folder" {
				folderName := getResourceName(tokens, nerTags, i+1, "folder")
				if folderName != "" {
					// If there's a target resource already identified, set its directory.
					if output.TargetResource != nil && output.TargetResource.Name != "" {
						output.TargetResource.Directory = folderName
					} else {
						// If no target resource yet, create a folder resource and set it as target.
						*folderResource = &semantic.Resource{Type: "Filesystem::Folder", Name: folderName, Properties: map[string]interface{}{"permissions": 0755}}
						output.TargetResource = *folderResource
					}
					return true, 3, nil // Consumed "in", "folder" and "name"
				}
			}
			return false, 0, nil
		},
	})
}