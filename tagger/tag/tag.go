package tag

import (
	"strings"

	"gorgonia.org/gorgonia"
)

type Features struct {
	WebServerKeyword  float64
	PreviousWord      float64
	NextWord          float64
	PreviousArticle   float64
	NextPreposition   float64
	NextOfIn          float64
	SpecialCharacters float64
	NameSuffix        float64
	PreviousTag       float64
	NextTag           float64
	FollowedByNumber  float64
	IsNoun            float64
}
type Tag struct {
	PosTag    []string
	NerTag    []string
	PhraseTag []string
	Tokens    []string
	Features  []Features
	Epoch     int
	Cost      gorgonia.Value
	IsName    bool
	Token     string
	Tags      []Tag
}

// ExtractFeaturesForNoun extracts features for a noun at the given index.
func (t Tag) ExtractFeaturesForNoun() Tag {
	for i := range t.Tokens {
		// Extract features for nouns
		if t.PosTag[i] == "NNP" || t.PosTag[i] == "NN" {
			t = t.extractNounFeatures()

		}
	}
	return t // Return the tag.Tag with extracted features
}

// extractNounFeatures extracts features for a noun based on its context.
func (t Tag) extractNounFeatures() Tag {
	// Pad t.Features to match the length of t.Tokens
	if len(t.Features) < len(t.Tokens) {
		paddingSize := len(t.Tokens) - len(t.Features)
		padding := make([]Features, paddingSize)
		t.Features = append(t.Features, padding...)
	}
	if len(t.NerTag) < len(t.Tokens) {
		paddingSize := len(t.Tokens) - len(t.NerTag)
		padding := make([]string, paddingSize)
		t.NerTag = append(t.NerTag, padding...)
	}

	for i := range t.Tokens {
		// Extract features for nouns
		if t.PosTag[i] == "NNP" || t.PosTag[i] == "NN" { // Use t.PosTag instead of t.Tagger
			if i > 1 && i < len(t.Tokens)-1 {
				// Feature 1: Is token a web server keyword?
				webserverKeywords := []string{"named", "with", "has", "string", "int", "integer", "float", "boolean", "bool", "structure", "object", "class", "array", "list", "webserver", "server", "handler", "endpoint", "api", "request", "response"}
				t.Features[i].WebServerKeyword = boolToFloat(contains(webserverKeywords, strings.ToLower(t.Tokens[i])))
				// Feature 2: Is the previous word a web server keyword?
				prevword := []string{"named", "with", "has", "string", "int", "integer", "float", "boolean", "bool", "structure", "object", "class", "array", "list", "webserver", "server", "handler", "endpoint", "api", "request", "response"}
				t.Features[i].PreviousWord = boolToFloat(contains(prevword, strings.ToLower(t.Tokens[i-1])))
				// Feature 3: Is the next word a web server keyword?
				nextword := []string{"named", "with", "has", "string", "int", "integer", "float", "boolean", "bool", "structure", "object", "class", "array", "list", "webserver", "server", "handler", "endpoint", "api", "request", "response"}
				t.Features[i].NextWord = boolToFloat(contains(nextword, strings.ToLower(t.Tokens[i+1])))
				// Feature 4: Is the previous word an article ("a", "an", "the")?
				t.Features[i].PreviousArticle = boolToFloat(t.Tokens[i-2] == "a" || t.Tokens[i-2] == "an" || t.Tokens[i-2] == "the")
				// Feature 5: Is the next word a preposition?
				prepositions := []string{"of", "in", "to", "for", "with", "on", "at", "from", "by"}
				t.Features[i].NextPreposition = boolToFloat(contains(prepositions, t.Tokens[i+1]))
				// Feature 6: Is the next word "of" or "in"?
				t.Features[i].NextOfIn = boolToFloat(t.Tokens[i+1] == "of" || t.Tokens[i+1] == "in")
				// Feature 7: Contains special characters (limited set for now)
				t.Features[i].SpecialCharacters = boolToFloat(strings.ContainsAny(t.Tokens[i], "-'"))
				// Feature 8: Check for name suffixes
				nameSuffixes := []string{"son", "ton", "man", "er", "smith"}
				t.Features[i].NameSuffix = boolToFloat(hasSuffix(t.Tokens[i], nameSuffixes)) // Use t.Tokens[i] instead of t.Tagger
				// Feature 9: Contextual feature: POS tag of the previous word
				prevTag := t.PosTag[i-1]
				t.Features[i].PreviousTag = boolToFloat(prevTag == "JJ" || prevTag == "DT") // Adjective or determiner
				// Feature 10: Contextual feature: POS tag of the next word
				nextTag := t.PosTag[i+1]
				t.Features[i].NextTag = boolToFloat(nextTag == "VBZ" || nextTag == "VBP") // Verb (present tense)

			} else if i >= len(t.Tokens)-1 {
				// Feature 1: Is token a web server keyword?
				webserverKeywords := []string{"named", "with", "has", "string", "int", "integer", "float", "boolean", "bool", "structure", "object", "class", "array", "list", "webserver", "server", "handler", "endpoint", "api", "request", "response"}
				t.Features[i].WebServerKeyword = boolToFloat(contains(webserverKeywords, strings.ToLower(t.Tokens[i])))
				// Feature 2: Is the previous word a web server keyword?
				prevword := []string{"named", "with", "has", "string", "int", "integer", "float", "boolean", "bool", "structure", "object", "class", "array", "list", "webserver", "server", "handler", "endpoint", "api", "request", "response"}
				t.Features[i].PreviousWord = boolToFloat(contains(prevword, strings.ToLower(t.Tokens[i-1]))) // Assuming i > 0
				// Feature 3: Is the next word a web server keyword? (Not applicable here)
				t.Features[i].NextWord = 0.0 // Since i is at or beyond the last token
				// Feature 4: Is token followed by a number? (Could indicate number of fields)
				t.Features[i].FollowedByNumber = 0.0 // Since i is at or beyond the last token
				// Feature 5: Contains special characters (limited set for now)
				t.Features[i].SpecialCharacters = boolToFloat(strings.ContainsAny(t.Tokens[i], "-'"))
				// Feature 6: Check for name suffixes
				nameSuffixes := []string{"son", "ton", "man", "er", "smith"}
				t.Features[i].NameSuffix = boolToFloat(hasSuffix(t.Tokens[i], nameSuffixes)) // Use t.Tokens[i] instead of t.Tagger
				// Feature 7: Is the word a noun?
				t.Features[i].IsNoun = boolToFloat((t.NerTag[i] == "PERSON") || (t.NerTag[i] == "ORG") || (t.NerTag[i] == "GPE") || (t.PosTag[i] == "NNP") || (t.PosTag[i] == "NN"))

			} else if i <= 1 {
				// Feature 1: Is token a web server keyword?
				webserverKeywords := []string{"named", "with", "has", "string", "int", "integer", "float", "boolean", "bool", "structure", "object", "class", "array", "list", "webserver", "server", "handler", "endpoint", "api", "request", "response"}
				t.Features[i].WebServerKeyword = boolToFloat(contains(webserverKeywords, strings.ToLower(t.Tokens[i])))
				// Feature 2: Is the previous word a web server keyword? (Not applicable if i == 0)
				if i > 0 {
					prevword := []string{"named", "with", "has", "string", "int", "integer", "float", "boolean", "bool", "structure", "object", "class", "array", "list", "webserver", "server", "handler", "endpoint", "api", "request", "response"}
					t.Features[i].PreviousWord = boolToFloat(contains(prevword, strings.ToLower(t.Tokens[i-1])))
				} else {
					t.Features[i].PreviousWord = 0.0
				}
				// Feature 3: Is the next word a web server keyword?
				nextword := []string{"named", "with", "has", "string", "int", "integer", "float", "boolean", "bool", "structure", "object", "class", "array", "list", "webserver", "server", "handler", "endpoint", "api", "request", "response"}
				t.Features[i].NextWord = boolToFloat(contains(nextword, strings.ToLower(t.Tokens[i+1])))

				webserverKeywords = []string{"named", "with", "has", "string", "int", "integer", "float", "boolean", "bool", "structure", "object", "class", "array", "list", "webserver", "server", "handler", "endpoint", "api", "request", "response"}
				t.Features[i].WebServerKeyword = boolToFloat(contains(webserverKeywords, strings.ToLower(t.Tokens[i])))
				// Feature 2: Is the previous word a web server keyword? (Not applicable if i == 0)
				if i > 0 {
					prevword := []string{"named", "with", "has", "string", "int", "integer", "float", "boolean", "bool", "structure", "object", "class", "array", "list", "webserver", "server", "handler", "endpoint", "api", "request", "response"}
					t.Features[i].PreviousWord = boolToFloat(contains(prevword, strings.ToLower(t.Tokens[i-1])))
				} else {
					t.Features[i].PreviousWord = 0.0
				}
				// Feature 3: Is the next word a web server keyword?
				nextword = []string{"named", "with", "has", "string", "int", "integer", "float", "boolean", "bool", "structure", "object", "class", "array", "list", "webserver", "server", "handler", "endpoint", "api", "request", "response"}
				t.Features[i].NextWord = boolToFloat(contains(nextword, strings.ToLower(t.Tokens[i+1])))
				// Feature 4: Is token followed by a number? (Could indicate number of fields)
				if i < len(t.Tokens)-1 && containsDigit(t.Tokens[i+1]) {
					t.Features[i].FollowedByNumber = 1.0
				} else {
					t.Features[i].FollowedByNumber = 0.0
				}
				// Feature 5: Contains special characters (limited set for now)
				t.Features[i].SpecialCharacters = boolToFloat(strings.ContainsAny(t.Tokens[i], "-'"))
				// Feature 6: Check for name suffixes
				nameSuffixes := []string{"son", "ton", "man", "er", "smith"}
				t.Features[i].NameSuffix = boolToFloat(hasSuffix(t.Tokens[i], nameSuffixes)) // Use t.Tokens[i] instead of t.Tagger
				// Feature 7: Is the next word "of" or "in"?
				if i < len(t.Tokens)-1 {
					t.Features[i].NextOfIn = boolToFloat(t.Tokens[i+1] == "of" || t.Tokens[i+1] == "in")
				} else {
					t.Features[i].NextOfIn = 0.0
				}
				// Feature 8: Is the word a noun?
				t.Features[i].IsNoun = boolToFloat((t.NerTag[i] == "PERSON") || (t.NerTag[i] == "ORG") || (t.NerTag[i] == "GPE") || (t.PosTag[i] == "NNP") || (t.PosTag[i] == "NN"))
			}

		}

	}
	return t
}

func hasSuffix(text string, suffixes []string) bool {
	for _, suffix := range suffixes {
		if strings.HasSuffix(strings.ToLower(text), suffix) {
			return true
		}
	}
	return false
}

// Helper function to convert bool to float64
func boolToFloat(b bool) float64 {
	if b {
		return 1.0
	}
	return 0.0
}

// Helper function to check if a string is in a slice (case-insensitive)
func contains(s []string, e string) bool {
	for _, a := range s {
		if strings.ToLower(a) == strings.ToLower(e) {
			return true
		}
	}
	return false
}

// Helper function to check if a string contains a digit
func containsDigit(text string) bool {
	for _, char := range text {
		if char >= '0' && char <= '9' {
			return true
		}
	}
	return false
}
