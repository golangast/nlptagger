package verbs

import (
	"regexp"
	"strings"
)

func VerbTag(posTags map[string]string, token string, posTag string, i int, tokens []string) string {
	for pattern, tag := range posTags {
		re := regexp.MustCompile(pattern)
		if re.MatchString(token) {
			posTag = tag
			// Contextual rule for verbs ending in "ing"
			if posTag == "NN" && strings.HasSuffix(token, "ing") {
				if i > 0 && (tokens[i-1] == "is" || tokens[i-1] == "are" || tokens[i-1] == "was" || tokens[i-1] == "were") {
					posTag = "VBG" // Present participle
					return posTag
				}
			}
			break
		}
	}
	return posTag
}
