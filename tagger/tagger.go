package tagger

import (
	"fmt"
	"strings"

	"github.com/zendrulat/nlptagger/tagger/dependencyrelation"
	"github.com/zendrulat/nlptagger/tagger/nertagger"
	"github.com/zendrulat/nlptagger/tagger/phrasetagger"
	"github.com/zendrulat/nlptagger/tagger/postagger"
	"github.com/zendrulat/nlptagger/tagger/tag"
)

const (
	BOS_ID int = 1
	EOS_ID int = 2
	PAD_ID int = 0
	UNK_ID int = 3
)

// checking and returning the tags
func Tagging(text string) tag.Tag {

	//start tagging
	tag := postagger.Postagger(text)

	//Pos checking nouns/verbs
	tag = postagger.VerbCheck(tag)
	tag = postagger.NounCheck(tag)
	tag = nertagger.Nertagger(tag)

	tag = phrasetagger.CheckPhrase(text, tag)

	tag, err := dependencyrelation.PredictDependencies(tag)
	if err != nil {
		fmt.Println(err)
	}

	return tag // Return the slice of Tag structs
}

func Filtertags(text string) string {
	tag := postagger.Postagger(text)
	filtertags := []string{"DT", "PRP", "IN", "CC", "RP", "ADP", "WDT", "DET", "WP"}
	newsentence := filterString(tag, filtertags)
	//fmt.Println(newsentence)
	return newsentence
}
func filterString(tag tag.Tag, filters []string) string {
	//	fmt.Println("begin with ", tag.Sentence)
	if len(tag.Tokens) == 0 {
		return ""
	}
	str := ""
	for i, token := range tag.Tokens {
		filtered := false
		for _, filter := range filters {
			if tag.PosTag[i] == filter {
				//fmt.Println("words that filtered", token, "tag: ", tag.PosTag[i])
				filtered = true
			}
		}
		if !filtered {
			str += token + " "
		}
	}
	//fmt.Println("end sentence ", str)
	return strings.TrimSpace(str)

}
