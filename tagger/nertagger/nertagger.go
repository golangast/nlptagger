package nertagger

import (
	"regexp"
	"strconv"
	"strings"

	"github.com/golangast/nlptagger/tagger/tag"
)

var nerTags = map[string]string{
	`\bcreate\b`: "COMMAND",
	`\b(webserver|database|handler|structure|field|data)\b`: "OBJECT_TYPE",
	`\bnamed\s+([a-zA-Z]+)\b`:                               "NAME",
	`\bthe\b`:                                               "DETERMINER", // Add this
	`\b(int|int8|int16|int32|int64|uint|uint8|uint16|uint32|uint64|float32|float64|complex64|complex128|string|byte|rune|integer|boolean|float|double|char|bool|int)\b`: "DATA_TYPE",
	`\bwith\s+the\s+handler\s+([a-zA-Z]+)\b`:                     "HANDLER_NAME",
	`\b(?:that\s+)?has\s+the\s+data\s+structure\s+([a-zA-Z]+)\b`: "DATA_STRUCTURE_NAME",
	`\bwith\s+(\d+)\s+([a-zA-Z]+)\s+fields?\b`:                   "FIELDS",
	`\b(?:with|has|using|containing|contain)\b`:                  "RELATION",
	`\b(?:of|for|in|on|to|from)\b`:                               "PREPOSITION",
	`\b(?:and|or|but|nor|yet|so)\b`:                              "CONJUNCTION",
	`\b(?:is|are|was|were|be|being|been|have|has|had|do|does|did|shall|will|should|would|may|might|must|can|could)\b`: "AUXILIARY_VERB",
	`\b(?:[A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*\s+(?:&|and)\s+[A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*)\b`:                         "ORG",
	`\b(?:[Tt]he\s+)?[A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*\s+(?:[Oo]f|[Ii]n)\s+[A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*\b`:         "ORG",
	`\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s+[A-Z]{2})\b`:                                                              "GPE",
	`\b(?:[A-Z]{2,})\b`: "GPE",
	`\b(?:[0-9]{1,3}\s+[A-Za-z]+(?:\s+(?:[Ss]t\.|[Aa]ve\.|[Rr]d\.|[Bb]lvd\.|[Dd]r\.|[Ll]n\.))?\s+(?:[A-Z][a-z]+)?(?:[\s-][A-Z][a-z]+)*)\b`:                                            "LOC",
	`\b(?:[Mm]ount|[Ll]ake)\s+[A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*\b`:                                                                                                                     "LOC",
	`\b(?:[A-Z][a-z]{3,}\s+[A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*)\b`:                                                                                                                       "PRODUCT",
	`\b(?:[0-9]{1,2}(?:st|nd|rd|th)?\s+(?:[Jj]anuary|[Ff]ebruary|[Mm]arch|[Aa]pril|[Mm]ay|[Jj]une|[Jj]uly|[Aa]ugust|[Ss]eptember|[Oo]ctober|[Nn]ovember|[Dd]ecember)\s+[0-9]{1,4})\b`: "DATE",
	`\b(?:[0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4})\b`:                                                                                                                                        "DATE",
	`\b(?:[0-9]{1,2}:\d{2}(?::\d{2})?(?:\s?(?:AM|PM))?)\b`:                                                                                                                            "TIME",
	`\bdata\s+structure\s+of\b`:          "DATA_STRUCTURE",
	`\b(?:[A-Z][a-z]+\s+)+[A-Z][a-z]+\b`: "OBJECT_NAME",
	`\b(?:create|delete|update)\s+(?:a\s+)?(webserver|database|handler)\s+named\s+([a-zA-Z]+)\b`: "COMMAND",
	`\bwith\s+(?:the\s+)?handlers?\s+named\s+([a-zA-Z]+(?:\s*,\s*[a-zA-Z]+)*)\b`:                 "HANDLERS",
	`\bset\s+the\s+port\s+to\s+(\d+)\b`:                                                          "PORT",
	`\bwith\s+(\d+)\s+fields\b`:                                                                  "FIELDS",
	`\b(?:[Tt]he\s+)?[A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*\s+(?:[Ii]nc\.|[Ll]td\.|[Cc]orp\.)\b`:       "ORG",
	`(the\s+)?[A-Z][a-zA-Z]+([\s-][A-Za-z0-9]+)*`:                                                "OBJECT_NAME",
	`\bgenerate\s+a\s+(webserver|database|handler)\b`:                                            "ACTION", // For "generate a webserver"
	`\bcreate\s+(?:a\s+)?(webserver|database|handler)\s+named\s+([a-zA-Z]+)\b`:                   "COMMAND",
	// First and Last Names (with optional middle names/initials)
	`[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}`: "PERSON",
	// Last Name, First Name (with optional middle names/initials)
	`[A-Z][a-z]+,\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?`: "PERSON",
	// Names with suffixes (Jr., Sr., etc.)
	`[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\s+(?:Jr\.|Sr\.|III|IV)`: "PERSON",
	// Organizations (simple pattern)
	`(?:[A-Z][a-z]+\s+)+Inc\.|Corp\.|Ltd\.`: "ORG",
	// More specific patterns
	`\bcreate\s+(?:a\s+)?webserver\b`:                  "COMMAND",
	`\b(?:with\s+(?:the\s+)?)?handler\s+([a-zA-Z]+)\b`: "HANDLER_NAME",
	`\b\d+\b`: "NUMBER", // Matches one or more digits
}

// take in text and return a map of tokens to NER tags
func Nertagger(t tag.Tag) tag.Tag {
	for i, token := range t.Tokens {
		// Force "Create" to be "COMMAND"
		if t.Tokens[0] == "Create" {
			t.NerTag[0] = "COMMAND"
		}
		for pattern, tag := range nerTags {
			// Check if the pattern is found *within* the token
			re := regexp.MustCompile(pattern)
			matches := re.FindAllStringSubmatch(token, -1)
			if len(matches) > 0 {
				t.NerTag[i] = tag
			}
			// 2. If no exact match, use context and POS tags:

		}
		if t.NerTag[i] == "" {
			switch t.Tokens[i] {
			case "Inventory":
				if i > 0 && t.NerTag[i-1] == "ACTION" {
					t.NerTag[i] = "OBJECT_NAME"
				}
				break

			case "Product":
				if i > 0 && t.NerTag[i-1] == "OBJECT_TYPE" {
					t.NerTag[i] = "OBJECT_NAME"
				} else {
					t.NerTag[i] = "PRODUCT"
				}
				break

			case "field", "fields", "Fields", "Field":
				if i > 0 && t.NerTag[i-1] == "DATA_TYPE" {
					t.NerTag[i] = "DATA_STRUCTURE_FIELD"
				} else if i > 0 && t.NerTag[i-1] == "OBJECT_TYPE" {
					t.NerTag[i] = "DATA_STRUCTURE_FIELD"
				}
				if i > 0 && t.NerTag[i-1] == "NUMBER" { // If preceded by a number
					t.NerTag[i] = "DATA_STRUCTURE_FIELD" // Tag as data structure field
				}
				break

			case "webserver":
				t.NerTag[i] = "OBJECT_TYPE" // Or "TECHNOLOGY" if more appropriate
				break

			case "named":
				// Check if it's part of a naming pattern (e.g., "named <ENTITY>")
				if i+1 < len(t.Tokens) {
					t.NerTag[i] = "NAME_PREFIX"
					t.NerTag[i+1] = "NAME"
				}
				break

			case "handler":
				t.NerTag[i+1] = "HANDLER_NAME"
				break

			}

			if i > 2 && i < len(t.Tokens)-2 {
				switch {

				case t.NerTag[i-2] == "OBJECT_TYPE" && t.NerTag[i-1] == "ACTION" || t.NerTag[i-1] == "NAME_PREFIX":
					t.NerTag[i] = "DATA_STRUCTURE_TYPE"
				}

			}

			// Utilize POS tags for better tagging
			switch t.PosTag[i] {

			case "DET": //
				t.NerTag[i] = "DETERMINER" // Or another custom tag
				break

			case "NNP", "NNPS": // Proper nouns (singular/plural)
				t.NerTag[i] = "NAME"
				break

			case "NN", "NNS": // Common nouns (singular/plural)
				// Check for specific nouns or context to assign NER tags
				if strings.Contains(token, "server") { // Example: Check for server-related nouns
					t.NerTag[i] = "OBJECT_TYPE"
				} else if strings.Contains(token, "handler") {
					t.NerTag[i] = "OBJECT_TYPE"
				}
				break

			case "JJ": // Adjectives
				// Check context or preceding nouns for NER tag assignment
				if i > 0 && t.NerTag[i-1] == "NAME" { // Example: Adjective modifying a name
					t.NerTag[i] = "NAME_MODIFIER" // Or another suitable tag
				} else if i > 0 && (t.NerTag[i-1] == "OBJECT_TYPE" || t.NerTag[i-1] == "LOCATION") {
					// Adjective modifying an object type or location
					t.NerTag[i] = "OBJECT_MODIFIER" // Or a more specific tag if needed
				} else if i > 0 && t.PosTag[i-1] == "NN" && strings.Contains(t.Tokens[i-1], "size") {
					// Adjective modifying a noun related to size (e.g., "large size")
					t.NerTag[i] = "SIZE_MODIFIER"
				} else if i > 0 && t.PosTag[i-1] == "NN" && strings.Contains(t.Tokens[i-1], "color") {
					// Adjective modifying a noun related to color (e.g., "red color")
					t.NerTag[i] = "COLOR_MODIFIER"
				} else if i > 0 && t.PosTag[i-1] == "NN" && strings.Contains(t.Tokens[i-1], "material") {
					// Adjective modifying a noun related to material (e.g., "wooden material")
					t.NerTag[i] = "MATERIAL_MODIFIER"
				}
				break

			}
		}
	}
	return t
}
func NerNounCheck(t tag.Tag) tag.Tag {

	for i := 0; i < len(t.Tokens); i++ {

		if i > 0 && i < len(t.Tokens)-1 && t.Tokens[i] != "" {

			var prevToken, nextToken string
			token := t.Tokens[i]
			tag := t.NerTag[i]
			if i < len(t.Tokens)-1 {
				nextToken = strings.ToLower(t.Tokens[i+1])
			}
			if i > 0 {
				prevToken = strings.ToLower(t.Tokens[i-1])
			}
			if nextToken == "structure" && t.NerTag[i] == "DATA" {
				t.NerTag[i] = "DATA_STRUCTURE"
				token += " " + nextToken
				nextToken = ""
			} else if nextToken == "name" && tag == "OBJECT" {
				tag = "OBJECT_NAME"
				token += " " + nextToken
				nextToken = ""
			} else if tag == "PERSON" && (nextToken == "said" || nextToken == "says") {
				tag = "PERSON_SPEAKER"
			} else if prevToken == "a" || prevToken == "an" || prevToken == "the" {
				if nextToken == "of" {
					tag = "OBJECT_TYPE"
				} else {
					tag = "OBJECT_NAME"
				}
			} else if prevToken == "named" {
				tag = "NAME" // Directly tag as "NAME" after "named"
			} else if prevToken == "with" {
				if nextToken == "handler" || nextToken == "handlers" {
					tag = "HANDLER_TYPE"
				} else if i < len(t.Tokens)-2 && t.Tokens[i+2] == "fields" {
					tag = "FIELDS"
				} else {
					tag = "OBJECT_PROPERTY"
				}
			} else if nextToken == "handler" || nextToken == "handlers" {
				tag = "HANDLER_TYPE"
			} else if i > 1 && t.Tokens[i-2] == "data" && prevToken == "structure" {
				tag = "DATA_STRUCTURE_TYPE"
			} else if strings.HasPrefix(token, "-") {
				tag = "FLAG"
			} else if _, err := strconv.Atoi(token); err == nil {
				tag = "NUMBER"
			} else if i > 1 && t.Tokens[i-2] == "the" && prevToken == "handler" {
				tag = "HANDLER_NAME"
			} else if i > 2 && t.Tokens[i-3] == "the" && t.Tokens[i-2] == "data" && prevToken == "structure" {
				tag = "DATA_STRUCTURE_NAME"
			}
		}
	}
	return t // Return the nounMap with updated tags
}
func NerVerbCheck(t tag.Tag) tag.Tag {
	for i := 0; i < len(t.Tokens); i++ {
		if i > 0 {
			prevToken := strings.ToLower(t.Tokens[i-1])
			if prevToken == "create" && t.NerTag[i] == "OBJECT" {
				t.NerTag[i] = "ACTION_CREATE"
			} else if (prevToken == "with" || prevToken == "using") && t.NerTag[i] == "TOOL" { // More context for tools
				t.NerTag[i] = "TOOL_WITH"
			}
		}
		// List of known action verbs
		var actionVerbs = map[string]bool{
			"create": true, "delete": true, "update": true, "build": true, "set": true, "add": true, "remove": true, "modify": true, "configure": true, "deploy": true, "run": true, "execute": true, "start": true, "stop": true, "generate": true, "write": true, "read": true, "send": true, "receive": true, "connect": true, "disconnect": true, "attach": true, "detach": true, "upload": true, "download": true, "install": true, "uninstall": true, "copy": true, "move": true, "rename": true, "open": true, "close": true, "find": true, "search": true, "get": true, "put": true, "post": true, "list": true, "show": true, "display": true, "view": true, "edit": true, "save": true, "load": true, "import": true, "export": true, "test": true, "debug": true, "validate": true, "compile": true, "link": true, "publish": true, "subscribe": true, "authorize": true, "authenticate": true, "encrypt": true, "decrypt": true, "compress": true, "decompress": true, "backup": true, "restore": true, "rollback": true, "commit": true, "push": true, "pull": true, "clone": true, "fork": true, "merge": true, "revert": true, "branch": true, "named": true,
		}
		if actionVerbs[t.Tokens[i]] {
			t.NerTag[i] = "ACTION"
		} else {
			if i > 0 && t.Tokens[i-1] == "a" {
				t.NerTag[i] = "ACTION"
			}
		}
	}
	return t
}
