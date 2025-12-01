package semantic

import (
	"strings"

	"github.com/zendrulat/nlptagger/neural/nn/ner"
)

// CommandParser parses queries into structured commands
type CommandParser struct {
	classifier *IntentClassifier
	extractor  *EntityExtractor
}

// NewCommandParser creates a new command parser
func NewCommandParser() *CommandParser {
	return &CommandParser{
		classifier: NewIntentClassifier(),
		extractor:  NewEntityExtractor(),
	}
}

// Parse converts a query into a structured command
func (cp *CommandParser) Parse(query string, words []string, entityMap map[int]ner.EntityType) *StructuredCommand {
	// Classify intent
	intent := cp.classifier.Classify(query)

	// Extract entities
	entities := cp.extractor.Extract(words, entityMap)

	// Build structured command
	return cp.BuildCommand(intent, entities, words)
}

// BuildCommand constructs a StructuredCommand from intent and entities
func (cp *CommandParser) BuildCommand(intent IntentType, entities map[string]string, words []string) *StructuredCommand {
	cmd := NewStructuredCommand()

	// Set action from intent
	cmd.Action = IntentToAction(intent)

	// Set primary object type
	cmd.ObjectType = IntentToObjectType(intent)

	// Set name based on object type
	switch cmd.ObjectType {
	case ObjectFile:
		if name, ok := entities["file"]; ok {
			cmd.Name = name
		} else if name, ok := entities["source_file"]; ok {
			cmd.Name = name
		}
	case ObjectFolder:
		if name, ok := entities["folder"]; ok {
			cmd.Name = name
		} else if name, ok := entities["source_folder"]; ok {
			cmd.Name = name
		}
	case ObjectComponent:
		if name, ok := entities["component"]; ok {
			cmd.Name = name
		}
	case ObjectCode:
		if name, ok := entities["code_type"]; ok {
			cmd.Name = name
		}
	}

	// Set path
	if path, ok := entities["path"]; ok {
		cmd.Path = path
	} else if dest, ok := entities["destination_folder"]; ok {
		cmd.Path = dest
	}

	// Detect keyword and secondary operation
	keyword := cp.detectKeyword(words)
	if keyword != "" {
		cmd.Keyword = keyword
	}

	// Set secondary object type from intent
	secondaryType := IntentToSecondaryObjectType(intent)
	if secondaryType != "" {
		cmd.ArgumentType = secondaryType

		// Set argument name based on type
		switch secondaryType {
		case ObjectFile:
			if name, ok := entities["file"]; ok {
				cmd.ArgumentName = name
			} else if name, ok := entities["destination_file"]; ok {
				cmd.ArgumentName = name
			}
		case ObjectFolder:
			if name, ok := entities["destination_folder"]; ok {
				cmd.ArgumentName = name
			}
		}
	}

	// For rename and move operations, set argument name as destination
	if cmd.Action == ActionRename {
		if newName, ok := entities["destination_file"]; ok {
			cmd.ArgumentName = newName
			cmd.ArgumentType = cmd.ObjectType // Same type as source
		} else if newName, ok := entities["destination_folder"]; ok {
			cmd.ArgumentName = newName
			cmd.ArgumentType = cmd.ObjectType
		}
	} else if cmd.Action == ActionMove {
		if dest, ok := entities["destination_folder"]; ok {
			cmd.ArgumentName = dest
			cmd.ArgumentType = ObjectFolder
		}
	}

	// Set properties
	cmd.Properties = make(map[string]string)
	for key, value := range entities {
		// Skip already-mapped entities
		if key == "file" || key == "folder" || key == "source_file" || key == "source_folder" ||
			key == "destination_file" || key == "destination_folder" || key == "path" ||
			key == "component" || key == "code_type" {
			continue
		}
		cmd.Properties[key] = value
	}

	return cmd
}

// detectKeyword identifies keywords in the query
func (cp *CommandParser) detectKeyword(words []string) CommandKeyword {
	for _, word := range words {
		lowerWord := strings.ToLower(word)
		switch lowerWord {
		case "with":
			return KeywordWith
		case "and":
			return KeywordAnd
		case "in":
			return KeywordIn
		case "into":
			return KeywordInto
		case "to":
			return KeywordTo
		case "inside":
			return KeywordInside
		}
	}
	return ""
}

// ParseQuery is a convenience method that parses a query string directly
func (cp *CommandParser) ParseQuery(query string, entityMap map[int]ner.EntityType) *StructuredCommand {
	words := strings.Fields(query)
	return cp.Parse(query, words, entityMap)
}
