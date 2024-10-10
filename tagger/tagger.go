package tagger

import (
	"github.com/golangast/nlptagger/tagger/nertagger"
	"github.com/golangast/nlptagger/tagger/phrasetagger"
	"github.com/golangast/nlptagger/tagger/postagger"
	"github.com/golangast/nlptagger/tagger/tag"
)

// checking and returning the tags
func Tagging(text string) tag.Tag {

	//start tagging
	tag := postagger.Postagger(text)

	//Pos checking nouns/verbs
	tag = postagger.VerbCheck(tag)
	tag = postagger.NounCheck(tag)
	tag = nertagger.Nertagger(tag)

	//Ner checking nouns/verbs
	tag = nertagger.NerNounCheck(tag)
	tag = nertagger.NerVerbCheck(tag)

	tag = phrasetagger.CheckPhrase(text, tag)

	return tag // Return the slice of Tag structs
}
