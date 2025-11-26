package nertagger

import "github.com/zendrulat/nlptagger/tagger/tag"

func TagTokens(tokens []string, posTags []string) []string {
	return RefactoredTagTokens(tokens, posTags)
}

func Nertagger(t tag.Tag) tag.Tag {
	t.NerTag = RefactoredTagTokens(t.Tokens, t.PosTag)
	return t
}
