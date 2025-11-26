package nertagger

var nerTags = []string{
	"O", // Outside of a named entity
	"COMMAND",
	"OBJECT_TYPE",
	"NAME",
	"PREPOSITION",
	"DETERMINER",
	"NAME_PREFIX",
	"PATH",
	"WEBSERVER",
	"HANDLER",
	"OBJECT_NAME",
}

var nerTagToID map[string]int
var idToNerTag map[int]string

func init() {
	nerTagToID = make(map[string]int)
	idToNerTag = make(map[int]string)
	for i, tag := range nerTags {
		nerTagToID[tag] = i
		idToNerTag[i] = tag
	}
}

func NerTagToIDMap() map[string]int {
	return nerTagToID
}

func IDToNerTagMap() map[int]string {
	return idToNerTag
}
