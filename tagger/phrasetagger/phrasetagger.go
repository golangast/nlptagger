package phrasetagger

import (
	"github.com/zendrulat/nlptagger/tagger/tag"
)

// PhraseTagger is a struct that holds any necessary state for phrase tagging.
type PhraseTagger struct {
	// Add any fields here if the tagger needs state (e.g., loaded models, vocabularies)
}

// NewPhraseTagger creates and returns a new instance of PhraseTagger.
func NewPhraseTagger() *PhraseTagger {
	return &PhraseTagger{}
}

// GenerateCommands processes a sequence of words and generates commands.
// For now, it simply returns the input sequence as commands.
// In a real scenario, this would involve more sophisticated logic using POS/NER tags.
func (pt *PhraseTagger) GenerateCommands(words []string) []string {
	// In a real implementation, you would create a tag.Tag struct,
	// populate its Tokens, PosTag, and NerTag fields (perhaps from other taggers),
	// and then call CheckPhrase.
	// For this example, we'll just return the words as commands.
	return words
}

// ParseCommand parses a command string and extracts relevant information.
func CheckPhrase(text string, t tag.Tag) tag.Tag {
	if len(t.PhraseTag) < len(t.Tokens) {
		paddingSize := len(t.Tokens) - len(t.PhraseTag)
		padding := make([]string, paddingSize)
		t.PhraseTag = append(t.PhraseTag, padding...)
	}
	tokens := t.Tokens
	for i, token := range tokens {
		// if 1 < i && i < len(t.Tokens)-1 {

		switch {
		case t.PosTag[i] == "COMMAND_VERB" && i < len(tokens)-2 && t.PosTag[i+1] == "DET" && t.PosTag[i+2] == "NN":
			t.PhraseTag[i] = "command:" + tokens[i] + "_" + tokens[i+1] + "_" + tokens[i+2]
		case t.PosTag[i] == "COMMAND_VERB" && i < len(tokens)-2 && t.PosTag[i+1] == "DET" && t.PosTag[i+2] == "NN": // Verb Phrase: "generate a webserver"
			t.PhraseTag[i] = "command:" + tokens[i] + "_" + tokens[i+1] + "_" + tokens[i+2]

		case t.NerTag[i] == "ACTION" && token == "named" && i < len(tokens)-1 && t.NerTag[i+1] == "NAME": // Named Entity: "named dog"
			t.PhraseTag[i] = "objectName:" + tokens[i+1]

		case t.NerTag[i] == "OBJECT_TYPE" && i > 0 && i < len(tokens)-1 && t.NerTag[i-1] == "DET" && t.NerTag[i+1] == "OBJECT_TYPE": // Data Structure: "the data structure"
			t.PhraseTag[i-1] = "dataStructure:" + tokens[i-1] + "_" + tokens[i] + "_" + tokens[i+1]
		case t.PosTag[i] == "VB" && i < len(tokens)-1 && t.PosTag[i+1] == "NN": // Verb + Noun phrase: "create file"
			t.PhraseTag[i] = "command:" + tokens[i] + "_" + tokens[i+1]

		case t.PosTag[i] == "NN" && i < len(tokens)-1 && t.PosTag[i+1] == "NN": // Noun + Noun phrase: "file name"
			t.PhraseTag[i] = "object:" + tokens[i] + "_" + tokens[i+1]

		case t.NerTag[i] == "DATA_TYPE" && i < len(tokens)-1 && t.NerTag[i+1] == "OBJECT_TYPE": // Data type + Object type: "integer field"
			t.PhraseTag[i] = "dataType:" + tokens[i] + "_" + tokens[i+1]

		case t.PosTag[i] == "JJ" && i < len(tokens)-1 && t.PosTag[i+1] == "NN": // Adjective + Noun phrase: "new file"
			t.PhraseTag[i] = "object:" + tokens[i] + "_" + tokens[i+1]
		}

		switch {

		case t.Tokens[i] == "Create" && i < len(tokens)-1 && t.NerTag[i+1] == "OBJECT_NAME":
			t.PhraseTag[i] = "command:" + t.Tokens[i] + "_" + t.Tokens[i+1]
		case t.Tokens[i] == "Create" && i < len(tokens)-1 && t.NerTag[i+1] == "OBJECT_TYPE":
			t.PhraseTag[i] = "command:" + t.Tokens[i] + "_" + t.Tokens[i+1]
		case t.Tokens[i] == "Create" && i < len(tokens)-1 && t.NerTag[i+1] == "DATA_TYPE":
			t.PhraseTag[i] = "command:" + t.Tokens[i] + "_" + t.Tokens[i+1]
		case t.Tokens[i] == "Create" && i < len(tokens)-1 && t.NerTag[i+1] == "NUMBER":
			t.PhraseTag[i] = "command:" + t.Tokens[i] + "_" + t.Tokens[i+1]
		case t.Tokens[i] == "Create" && i < len(tokens)-1 && t.NerTag[i+1] == "ACTION":
			t.PhraseTag[i] = "command:" + t.Tokens[i] + "_" + t.Tokens[i+1] // Only tag if the current token is "Create"
			for j := i + 1; j < len(t.Tokens); j++ {
				if t.NerTag[j] == "OBJECT_NAME" {
					t.PhraseTag[i] = "command:" + t.Tokens[i] + "_" + t.Tokens[j]
					break // Exit the inner loop once the object name is found
				}
			}

		case i < len(tokens)-1 && t.NerTag[i+1] == "ACTION":
			for j := i + 1; j < len(t.Tokens); j++ {
				if t.NerTag[j] == "OBJECT_NAME" {
					t.PhraseTag[i] = "command:" + t.Tokens[i] + "_" + t.Tokens[j]
					break
				}
			}
		case t.Tokens[i] == "Create": // Check for "Create" verb
			for j := i + 1; j < len(t.Tokens); j++ {
				if t.NerTag[j] == "OBJECT_NAME" {
					t.PhraseTag[i] = "command:" + t.Tokens[i] + "_" + t.Tokens[j]
					break
				}
			}
		case t.Tokens[i] == "generate" && i < len(tokens)-1 && t.NerTag[i+1] == "OBJECT_TYPE":
			// Check for the verb phrase using NER and POS tags
			if t.NerTag[i] == "ACTION" && t.PosTag[i] == "COMMAND_VERB" &&
				t.NerTag[i+1] == "DETERMINER" && t.PosTag[i+1] == "DET" &&
				t.NerTag[i+2] == "ACTION" && t.PosTag[i+2] == "NN" {
				t.PhraseTag[i] = "verbPhrase:generate_a_webserver"
			}
		case t.Tokens[i] == "generate" && i+4 < len(tokens) && t.Tokens[i+1] == "a" && t.Tokens[i+2] == "handler" && t.Tokens[i+3] == "named" && t.NerTag[i+4] == "OBJECT_NAME":
			// Check for "generate a handler named <object_name>"
			t.PhraseTag[i] = "verbPhrase:generate_a_webserver"
		case t.PosTag[i] == "COMMAND_VERB" && i < len(tokens)-5 && t.PosTag[i+1] == "DET" && t.PosTag[i+2] == "NN" && tokens[i+3] == "named" && t.PosTag[i+4] == "NN":
			t.PhraseTag[i] = "command:" + tokens[i] + "_" + tokens[i+1] + "_" + tokens[i+2] + "_named_" + tokens[i+4]

		case token == "named" && i > 0 && t.NerTag[i] == "ACTION":
			t = handleNamedAction(tokens, i, t)
		case token == "fields" && i > 0 && t.NerTag[i] == "ACTION":
			t = handleFields(tokens, i, t)
		case token == "fields" && i > 0 && t.NerTag[i] == "OBJECT_TYPE":
			t = handleFields(tokens, i, t)
		case token == "fields" && i > 0 && t.NerTag[i] == "OBJECT_NAME":
			t = handleFields(tokens, i, t)
		case token == "fields" && i > 0 && t.NerTag[i] == "DATA_TYPE":
			t = handleFields(tokens, i, t)
		case token == "fields" && i > 0 && t.NerTag[i] == "NUMBER":
			t = handleFields(tokens, i, t)
		case i < len(tokens)-1 && t.NerTag[i] == "COMMAND": // Only consider command verbs
			for j := i + 1; j < len(t.Tokens); j++ {
				if t.NerTag[j] == "OBJECT_NAME" {
					t.PhraseTag[i] = "command:" + t.Tokens[i] + "_" + t.Tokens[j]
					break
				}
			}

		case i < len(tokens)-1 && t.NerTag[i+1] == "COMMAND_VERB":
			for j := i + 1; j < len(t.Tokens); j++ {
				if t.NerTag[j] == "OBJECT_TYPE" || t.PosTag[j] == "NN" { // Find object
					t.PhraseTag[i] = t.Tokens[i] + "Name:" + t.Tokens[j]
					break
				}
			}
			for j := i + 1; j < len(t.Tokens); j++ {
				if t.NerTag[j] == "OBJECT_NAME" {
					t.PhraseTag[i] = "command:" + t.Tokens[i] + "_" + t.Tokens[j]
					break
				}
			}
			t.PhraseTag[i] = "fieldType:" + token
		case token == "fields" && i > 0 && t.NerTag[i] == "NNS":
			t = handleFields(tokens, i, t)
		case t.Tokens[i] == "generate" && i < len(tokens)-2 && t.Tokens[i+1] == "a" && t.Tokens[i+2] == "webserver":
			// Check for "generate a webserver"
			t.PhraseTag[i] = "verbPhrase:generate_a_webserver"
		}

		// Assign a default tag if no specific rule was matched
		if t.PhraseTag[i] == "" {
			t.PhraseTag[i] = "WORD:" + token
		}
	}
	return t
}

// handleNamedAction handles the "named" action and extracts object names.
func handleNamedAction(tokens []string, i int, t tag.Tag) tag.Tag {
	if i < len(tokens)-1 {
		if i < len(tokens)-1 && t.NerTag[i] == "ACTION" && t.Tokens[i] == "named" {
			if i+1 < len(tokens) && t.NerTag[i+1] == "OBJECT_NAME" {
				t.PhraseTag[i] = "objectName:" + t.Tokens[i+1]
			}
		}
		if i < len(tokens)-2 && (t.PosTag[i] == "VB" || t.PosTag[i] == "COMMAND_VERB") && (t.PosTag[i+2] == "NN" || t.PosTag[i+2] == "NNP") {
			// Add phrase tag...
			t.PhraseTag[i] = tokens[i+2] + "Name:" + tokens[i+4] // Corrected index for name
		} else if i > 0 && t.NerTag[i-1] == "OBJECT_TYPE" && (i >= len(tokens)-2 || t.NerTag[i+2] != "ACTION") {
			t.PhraseTag[i-1] = tokens[i-1] + "Name:" + tokens[i+1]
		} else if i < len(tokens)-2 && t.NerTag[i+1] == "OBJECT_TYPE" && (i >= len(tokens)-3 || t.NerTag[i+3] != "ACTION") {
			t.PhraseTag[i+1] = tokens[i+1] + "Name:" + tokens[i+2]
		} else if i < len(tokens)-2 && t.PosTag[i+2] == "NN" {
			if i > 0 && t.NerTag[i-1] == "OBJECT_TYPE" {
				t.PhraseTag[i-1] = tokens[i-1] + "Name:" + tokens[i+2]
			}
		}
	}
	return t
}

// handleFields handles the "fields" token and extracts the number of fields.
func handleFields(tokens []string, i int, t tag.Tag) tag.Tag {
	for j := i - 1; j >= 0; j-- {
		switch {
		case t.Tokens[i] == "fields" && i > 0 && i < len(t.Tokens)-1 && t.Tokens[i+1] == "and":
			t.PhraseTag[i] = "fieldTypes:" + t.Tokens[i-1] + "_" + t.Tokens[i+3]
		case t.Tokens[i] == "fields" && i > 0 && i < len(t.Tokens)-1 && t.Tokens[i+1] == "in":
			t.PhraseTag[i] = "fieldTypes:" + t.Tokens[i-1] + "_" + t.Tokens[i+2]
		case t.PosTag[j] == "CD":
			t.PhraseTag[i] = "numFields:" + tokens[j]
		case t.Tokens[i] == "fields":
			if t.NerTag[j] == "NUMBER" {
				t.PhraseTag[i] = "numFields:" + t.Tokens[j]
			} else if t.NerTag[j] == "DATA_TYPE" {
				t.PhraseTag[i] = "fieldType:" + t.Tokens[j]
			}
		}

	}
	return t // Return the updated tag.Tag
}
