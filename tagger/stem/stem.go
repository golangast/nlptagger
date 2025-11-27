package stem

import (
	"regexp"
	"sort"
	"strings"
)

// Define a struct to hold suffix and frequency information
type SuffixFrequency struct {
	Suffix    string
	Frequency int
}

func Stem(word string) string {
	if word == "" {
		return ""
	}
	// Rule 1: Remove "-ing" suffix, handling double consonants
	re := regexp.MustCompile(`ing$`)
	if re.MatchString(word) && len(word) >= 3 {
		if len(word) >= 4 && word[len(word)-4] == word[len(word)-3] {
			return word[:len(word)-3]
		}
		return re.ReplaceAllString(word, "")
	}
	// Rule 2: Remove "-ed" suffix
	re = regexp.MustCompile(`ed$`)
	if re.MatchString(word) {
		return re.ReplaceAllString(word, "")
	}
	// Rule 3: Remove "-s" or "-es" suffix
	re = regexp.MustCompile(`(s|es)$`)
	if re.MatchString(word) {
		return re.ReplaceAllString(word, "")
	}
	// Rule 4: Remove "-ly" suffix
	re = regexp.MustCompile(`ly$`)
	if re.MatchString(word) {
		return re.ReplaceAllString(word, "")
	}
	// Rule 5: Replace "-y" with "-i" if preceded by a consonant
	re = regexp.MustCompile(`([^aeiou])y$`)
	if re.MatchString(word) {
		return re.ReplaceAllString(word, "$1i")
	}
	// Statistical stemming (with fallback)
	suffixes := getSuffixesByFrequency(word)
	for _, suffixFreq := range suffixes {
		suffix := suffixFreq.Suffix
		stem := word[:len(word)-len(suffix)]
		if _, ok := stemFrequencies[stem]; ok && suffixFreq.Frequency >= minimumSuffixFrequency {
			return stem
		}
	}
	// Add more rules and exceptions as needed
	return word
}

// ... (other functions) ...
// Get suffixes sorted by frequency, considering only those present in commonSuffixes
func getSuffixesByFrequency(word string) []SuffixFrequency {
	var suffixes []SuffixFrequency
	for suffix, freq := range suffixFrequencies {
		if strings.HasSuffix(word, suffix) && commonSuffixes[suffix] {
			suffixes = append(suffixes, SuffixFrequency{suffix, freq})
		}
	}
	// Sort suffixes by frequency in descending order
	sort.Slice(suffixes, func(i, j int) bool {
		return suffixes[i].Frequency > suffixes[j].Frequency
	})
	return suffixes
}

// Example: Simulate commonSuffixes, stemFrequencies, and suffixFrequencies based on analysis of a corpus
var commonSuffixes = map[string]bool{
	"ing": true,
	"ies": true,
	"ed":  true,
	"es":  true,
	"y":   true, // Add "y" to commonSuffixes
	"ly":  true,
}
var stemFrequencies = map[string]int{
	"run":     100,
	"studi":   50,
	"jump":    75,
	"box":     60,
	"see":     90,
	"happi":   40,
	"happili": 40,
}
var suffixFrequencies = map[string]int{
	"ing": 80,
	"ies": 30,
	"ed":  60,
	"es":  40,
	"y":   50, // Add frequency for "y"
	"ly":  70,
}
var minimumSuffixFrequency = 20 // Define a minimum frequency threshold for suffixes