package phrasetagger

import (
	"fmt"

	"github.com/golangast/nlptagger/tagger/tag"
)

// ParseCommand parses a command string and extracts relevant information.
func CheckPhrase(text string, t tag.Tag) tag.Tag {
	if len(t.PhraseTag) < len(t.Tokens) {
		paddingSize := len(t.Tokens) - len(t.PhraseTag)
		padding := make([]string, paddingSize)
		t.PhraseTag = append(t.PhraseTag, padding...)
	}
	tokens := t.Tokens
	for i, token := range tokens {
		switch {
		case i < len(tokens)-1 && t.NerTag[i+1] == "COMMAND_VERB":
			for j := i + 1; j < len(t.Tokens); j++ {
				if t.NerTag[j] == "OBJECT_TYPE" || t.PosTag[j] == "NN" { // Find object
					t.PhraseTag[i] = t.Tokens[i] + "Name:" + t.Tokens[j]
					break
				}
			}
		case i < len(tokens)-1 && t.NerTag[i+1] == "OBJECT_TYPE":
			t.PhraseTag[i] = tokens[i] + "Name:" + tokens[i+1]
		case t.NerTag[i] == "OBJECT_TYPE":
			// Update t.PhraseTag or another appropriate field within tag.Tag
			t.PhraseTag[i] = token + "Name:" + tokens[i+1]
		case t.NerTag[i] == "ACTION": // Call handleNamedAction for all "ACTION" tokens
			t = handleNamedAction(tokens, i, t)
		case t.NerTag[i] == "DATA_TYPE":
			// Update t.PhraseTag or another appropriate field within tag.Tag
			t.PhraseTag[i] = "fieldType:" + token
		case token == "fields" && i > 0 && t.NerTag[i] == "NNS":
			t = handleFields(tokens, i, t)
		}
	}
	return t
}

// handleNamedAction handles the "named" action and extracts object names.
func handleNamedAction(tokens []string, i int, t tag.Tag) tag.Tag {
	if i < len(tokens)-1 {
		if i < len(tokens)-2 && (t.PosTag[i] == "VB" || t.PosTag[i] == "COMMAND_VERB") && (t.PosTag[i+2] == "NN" || t.PosTag[i+2] == "NNP") {
			// Add phrase tag...
			t.PhraseTag[i] = tokens[i+2] + "Name:" + tokens[i+4] // Corrected index for name
		} else if i > 0 && t.NerTag[i-1] == "OBJECT_TYPE" && (i >= len(tokens)-2 || t.NerTag[i+2] != "ACTION") {
			fmt.Printf("Updating PhraseTag at index: %d, value: %sName:%s\n", i-1, tokens[i-1], tokens[i+1])
			t.PhraseTag[i-1] = tokens[i-1] + "Name:" + tokens[i+1]
		} else if i < len(tokens)-2 && t.NerTag[i+1] == "OBJECT_TYPE" && (i >= len(tokens)-3 || t.NerTag[i+3] != "ACTION") {
			fmt.Printf("Updating PhraseTag at index: %d, value: %sName:%s\n", i+1, tokens[i+1], tokens[i+2])
			t.PhraseTag[i+1] = tokens[i+1] + "Name:" + tokens[i+2]
		} else if i < len(tokens)-2 && t.PosTag[i+2] == "NN" {
			if i > 0 && t.NerTag[i-1] == "OBJECT_TYPE" {
				fmt.Printf("Updating PhraseTag at index: %d, value: %sName:%s\n", i-1, tokens[i-1], tokens[i+2])
				t.PhraseTag[i-1] = tokens[i-1] + "Name:" + tokens[i+2]
			}
		}
	}
	return t
}

// handleFields handles the "fields" token and extracts the number of fields.
func handleFields(tokens []string, i int, t tag.Tag) tag.Tag {
	for j := i - 1; j >= 0; j-- {
		if t.PosTag[j] == "CD" {
			// Update t.PhraseTag or another appropriate field within tag.Tag
			t.PhraseTag[i] = "numFields:" + tokens[j]
			break
		}
	}
	return t // Return the updated tag.Tag
}
