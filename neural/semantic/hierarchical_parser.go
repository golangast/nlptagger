package semantic

import (
	"strings"

	"github.com/zendrulat/nlptagger/neural/nn/ner"
)

// HierarchicalParser parses complex queries into hierarchical command trees
type HierarchicalParser struct {
	classifier       *IntentClassifier
	extractor        *EntityExtractor
	templateRegistry *TemplateRegistry
}

// NewHierarchicalParser creates a new hierarchical parser
func NewHierarchicalParser() *HierarchicalParser {
	return &HierarchicalParser{
		classifier:       NewIntentClassifier(),
		extractor:        NewEntityExtractor(),
		templateRegistry: NewTemplateRegistry(),
	}
}

// Parse converts a complex query into a hierarchical command tree
func (hp *HierarchicalParser) Parse(query string, words []string, entityMap map[int]ner.EntityType) *HierarchicalCommand {
	// Classify intent
	intent := hp.classifier.Classify(query)

	// Extract entities
	entities := hp.extractor.Extract(words, entityMap)

	// Build hierarchical command
	return hp.BuildHierarchicalCommand(query, words, intent, entities, entityMap)
}

// BuildHierarchicalCommand constructs a hierarchical command from parsed elements
func (hp *HierarchicalParser) BuildHierarchicalCommand(
	query string,
	words []string,
	intent IntentType,
	entities map[string]string,
	entityMap map[int]ner.EntityType,
) *HierarchicalCommand {

	cmd := NewHierarchicalCommand()

	// Set action from intent
	cmd.Action = IntentToAction(intent)
	cmd.ObjectType = IntentToObjectType(intent)

	// Set primary name - always try to get it from entities first
	if cmd.ObjectType == ObjectFolder {
		if name, ok := entities["folder"]; ok && name != "" {
			cmd.Name = name
		} else if name, ok := entities["source_folder"]; ok && name != "" {
			cmd.Name = name
		}
	} else if cmd.ObjectType == ObjectFile {
		if name, ok := entities["file"]; ok && name != "" {
			cmd.Name = name
		} else if name, ok := entities["source_file"]; ok && name != "" {
			cmd.Name = name
		}
	}

	// Detect template keyword
	template := hp.detectTemplate(words)
	if template != "" && hp.templateRegistry.HasTemplate(template) {
		cmd.Template = template
		// Apply template to populate children
		hp.templateRegistry.ApplyTemplate(cmd, template)
	}

	// Parse explicit file/folder children from query
	children := hp.parseChildren(query, words, entityMap)

	// Merge explicit children with template children
	// Explicit children override template defaults
	explicitNames := make(map[string]bool)
	for _, child := range children {
		explicitNames[child.Name] = true
		cmd.AddChild(child)
	}

	// Remove template children that are explicitly specified
	if template != "" {
		filteredChildren := make([]*HierarchicalCommand, 0)
		for _, child := range cmd.Children {
			if !explicitNames[child.Name] {
				filteredChildren = append(filteredChildren, child)
			}
		}
		cmd.Children = append(children, filteredChildren...)
	}

	return cmd
}

// detectTemplate detects project template keywords in the query
func (hp *HierarchicalParser) detectTemplate(words []string) string {
	templates := hp.templateRegistry.ListTemplates()

	for _, word := range words {
		lowerWord := strings.ToLower(word)
		for _, template := range templates {
			if lowerWord == strings.ToLower(template) {
				return template
			}
		}
	}

	return ""
}

// parseChildren extracts child files/folders from query
// Handles patterns like:
//
//	"in the file main.go"
//	"with handler.go"
//	"and a folder for templates"
func (hp *HierarchicalParser) parseChildren(query string, words []string, entityMap map[int]ner.EntityType) []*HierarchicalCommand {
	children := make([]*HierarchicalCommand, 0)

	// Look for file/folder patterns
	for i := 0; i < len(words); i++ {
		word := strings.ToLower(words[i])

		// Pattern: "file <name>" or "the file <name>"
		if word == "file" || (word == "the" && i+1 < len(words) && strings.ToLower(words[i+1]) == "file") {
			if word == "the" {
				i++ // Skip "the"
			}

			// Next word should be the filename
			if i+1 < len(words) {
				entityType, hasEntity := entityMap[i+1]
				if hasEntity && entityType == "FILE_NAME" {
					child := NewHierarchicalCommand()
					child.Action = ActionCreate
					child.ObjectType = ObjectFile
					child.Name = words[i+1]
					children = append(children, child)
				}
			}
		}

		// Pattern: "folder <name>" or "a folder for <name>"
		if word == "folder" {
			// Check if it's "a folder for <name>"
			if i >= 1 && strings.ToLower(words[i-1]) == "a" {
				// Look for "for <name>"
				if i+1 < len(words) && strings.ToLower(words[i+1]) == "for" && i+2 < len(words) {
					child := NewHierarchicalCommand()
					child.Action = ActionCreate
					child.ObjectType = ObjectFolder
					child.Name = words[i+2]
					children = append(children, child)
				}
			} else {
				// Simple "folder <name>"
				if i+1 < len(words) {
					entityType, hasEntity := entityMap[i+1]
					if hasEntity && entityType == "FOLDER_NAME" {
						child := NewHierarchicalCommand()
						child.Action = ActionCreate
						child.ObjectType = ObjectFolder
						child.Name = words[i+1]
						children = append(children, child)
					}
				}
			}
		}
	}

	return children
}

// ParseQuery is a convenience method that parses a query string directly
func (hp *HierarchicalParser) ParseQuery(query string, entityMap map[int]ner.EntityType) *HierarchicalCommand {
	words := strings.Fields(query)
	return hp.Parse(query, words, entityMap)
}
