package tokenize

import (
	"strconv"
	"strings"
	"tr/tagging"
	"tr/tagging/nertag"
	"tr/tagging/nouns"
	"tr/tagging/verbs"
)

// Tokenize splits text into words, handling punctuation and other complexities.
func Tokenize(text string) []Entity {
	tokens, contractions, posTags, nerTags, actionVerbs := tagging.Tagging(text)
	// Normalize tokens and add POS tags and NER tags
	var entities []Entity

	for i, token := range tokens {
		originalToken := token
		//filter
		token = BasicTokenChecking(token)
		//verbs
		posTag := verbs.VerbTag(token)
		//nouns
		posTag = nouns.NounTag(token)
		//nertags
		nerTag := nertag.NerTagger(token, nerTags, originalToken, posTag, i, tokens, actionVerbs)
		// Combine tags
		entity := Entity{
			Text:       token,
			Type:       nerTag,
			Tag:        posTag,
			StartIndex: i, // Assuming tokens are words
			EndIndex:   i + 1,
		}
		entities = append(entities, entity)
	}
	return entities

}

func BasicTokenChecking(token string) string {
	// Handle special characters and numbers
	if strings.ContainsAny(token, ".,;!?") {
		token = "."
	} else if _, err := strconv.Atoi(token); err == nil {
		token = "NUM"
	}
}

type Entity struct {
	Text       string
	Type       string
	Tag        string
	StartIndex int
	EndIndex   int
}
