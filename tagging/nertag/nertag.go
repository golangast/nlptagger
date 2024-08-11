package nertag

import (
	"regexp"
	"strconv"
	"strings"
)

func NerTagger(nertags map[string]string, token, originalToken, posTag string, i int, tokens []string, actionVerbs map[string]bool) string {

	// NER tagging with lookahead and lookbehind (improved)
	var nerTag string
	for pattern, tag := range nertags {
		re := regexp.MustCompile(pattern)
		if re.MatchString(token) {
			nerTag = tag
			// Lookahead for multi-word entities and context
			if i < len(tokens)-1 {
				nextToken := strings.ToLower(tokens[i+1])
				if nextToken == "structure" && nerTag == "DATA" {
					nerTag = "DATA_STRUCTURE"
					token += " " + nextToken
					tokens[i+1] = "" // Consume the next token
				} else if nextToken == "name" && nerTag == "OBJECT" {
					nerTag = "OBJECT_NAME"
					token += " " + nextToken
					tokens[i+1] = ""
				} else if nerTag == "PERSON" && (nextToken == "said" || nextToken == "says") { // Check for verbs indicating speech
					nerTag = "PERSON_SPEAKER"
				}
			}
			// Lookbehind for context
			if i > 0 {
				prevToken := strings.ToLower(tokens[i-1])
				if prevToken == "create" && nerTag == "OBJECT" {
					nerTag = "ACTION_CREATE"
				} else if (prevToken == "with" || prevToken == "using") && nerTag == "TOOL" { // More context for tools
					nerTag = "TOOL_WITH"
				}
			}
			// Check if the word is a potential action verb
			if posTag == "VB" || posTag == "VBD" || posTag == "VBG" || posTag == "VBN" || posTag == "VBZ" {
				// Check if the word is in the list of known action verbs
				if actionVerbs[token] {
					nerTag = "ACTION"
				} else {
					// Apply further logic based on context (example)
					if i > 0 && tokens[i-1] == "a" { // Check if preceded by "a"
						nerTag = "ACTION"
					}
					// ... more contextual rules for action verbs ...
				}
			}
			break
		}
	}
	// Handle unknown tokens with context
	if posTag == "" && nerTag == "" {
		if strings.ContainsAny(originalToken, "ABCDEFGHIJKLMNOPQRSTUVWXYZ") {
			if i > 0 && (tokens[i-1] == "a" || tokens[i-1] == "an" || tokens[i-1] == "the") {
				nerTag = "OBJECT_NAME"
			} else if i < len(tokens)-1 && tokens[i+1] == "of" {
				nerTag = "OBJECT_TYPE"
			} else if i > 0 && tokens[i-1] == "named" {
				nerTag = "NAMED_ENTITY"
			} else if i > 0 && tokens[i-1] == "with" { // e.g., "with handler"
				nerTag = "OBJECT_PROPERTY"
			} else if i < len(tokens)-1 && (tokens[i+1] == "handler" || tokens[i+1] == "handlers") { // e.g., "customer handler"
				nerTag = "HANDLER_TYPE"
			} else if i > 1 && tokens[i-2] == "data" && tokens[i-1] == "structure" { // e.g., "car data structure"
				nerTag = "DATA_STRUCTURE_TYPE"
			}
		} else if strings.HasPrefix(token, "-") {
			nerTag = "FLAG"
		} else if _, err := strconv.Atoi(token); err == nil { // Check for numbers that weren't caught earlier
			posTag = "CD" // Cardinal number
			nerTag = "NUMBER"
		} else {
			// More sophisticated unknown word handling (can be improved further)
			if i > 0 && (tokens[i-1] == "a" || tokens[i-1] == "an") {
				posTag = "NN" // Assume singular noun if preceded by an article
			} else if i > 0 && tokens[i-1] == "to" {
				posTag = "VB" // Assume base form verb if preceded by "to"
			} else {
				posTag = "UNK" // Unknown
			}
		}
	}
	return nerTag
}
