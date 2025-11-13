package nertagger

import (
	"regexp"
)

type nerRule struct {
	pattern *regexp.Regexp
	tag     string
}

var rules []nerRule

func init() {
	// Order matters here. More specific rules should come first.
	rules = []nerRule{
		{regexp.MustCompile(`\b(create|add|make)\b`), "COMMAND"},
		{regexp.MustCompile(`\b(delete|remove)\b`), "COMMAND"},
		{regexp.MustCompile(`\b(move|mv)\b`), "COMMAND"},
		{regexp.MustCompile(`\b(webserver|database|handler|structure|field|data|file|folder)\b`), "OBJECT_TYPE"},
		{regexp.MustCompile(`[a-zA-Z0-9_.-]+\.[a-zA-Z0-9_.-]+`), "NAME"},
		{regexp.MustCompile(`\b(in|on|to|from|for)\b`), "PREPOSITION"},
		{regexp.MustCompile(`\b(the|a|an)\b`), "DETERMINER"},
		{regexp.MustCompile(`\bnamed\b`), "NAME_PREFIX"},
		{regexp.MustCompile(`[a-zA-Z0-9_./-]+`), "PATH"},
	}
}

func RefactoredTagTokens(tokens []string, posTags []string) []string {
	tags := make([]string, len(tokens))
	for i, token := range tokens {
		// Default to POS tag if no NER tag is found
		tags[i] = posTags[i]

		for _, rule := range rules {
			if rule.pattern.MatchString(token) {
				tags[i] = rule.tag
				break // First match wins
			}
		}

		// Handle proper nouns
		if posTags[i] == "NNP" || posTags[i] == "NNPS" {
			tags[i] = "NAME"
		}

		// If previous tag was OBJECT_TYPE and current tag is NN, tag as NAME
		if i > 0 && tags[i-1] == "OBJECT_TYPE" && posTags[i] == "NN" {
			tags[i] = "NAME"
		}
	}
	return tags
}
