package vocab

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"regexp"
	"strings"

	mainvocab "github.com/zendrulat/nlptagger/neural/nnu/vocab"
)

// SetupVocabulary attempts to load a vocabulary from vocabPath, and if it fails,
// builds a new one from the provided dataPaths.
func SetupVocabulary(vocabPath string, dataPaths []string) (*mainvocab.Vocabulary, error) {
	vocab, err := mainvocab.LoadVocabulary(vocabPath)
	if err == nil {
		log.Printf("Loaded vocabulary from %s", vocabPath)
		return vocab, nil
	}

	log.Printf("Could not load vocabulary from %s, building new one.", vocabPath)
	if len(dataPaths) < 2 {
		return nil, fmt.Errorf("at least two data paths are required to build vocabulary")
	}
	vocab = BuildVocabulary(dataPaths[0], dataPaths[1])
	if vocab == nil {
		return nil, fmt.Errorf("failed to build vocabulary")
	}

	if err := vocab.Save(vocabPath); err != nil {
		return nil, fmt.Errorf("failed to save vocabulary: %w", err)
	}
	log.Printf("Saved new vocabulary to %s", vocabPath)

	return vocab, nil
}

// BuildVocabulary builds a vocabulary from the provided training data file and additional software commands data.
func BuildVocabulary(trainingDataPath, softwareCommandsPath string) *mainvocab.Vocabulary {
	wordSet := make(map[string]bool)

	// Read training data and extract words
	file, err := os.Open(trainingDataPath)
	if err != nil {
		log.Printf("Error opening training data file: %v", err)
		return nil
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	queryRe := regexp.MustCompile(`^Query: "(.*)"$`)
	intentRe := regexp.MustCompile(`^Intent: (.*)$`)

	for scanner.Scan() {
		line := scanner.Text()
		if matches := queryRe.FindStringSubmatch(line); matches != nil {
			tokenizeAndAddWords(matches[1], wordSet)
		} else if matches := intentRe.FindStringSubmatch(line); matches != nil {
			tokenizeAndAddWords(matches[1], wordSet)
		}
	}

	if err := scanner.Err(); err != nil {
		log.Printf("Error reading training data file: %v", err)
		return nil
	}

	// Expand vocabulary with software commands data
	addWordsFromSoftwareCommands(softwareCommandsPath, wordSet)

	// Convert set to slice and create vocabulary
	words := make([]string, 0, len(wordSet))
	for word := range wordSet {
		words = append(words, word)
	}

	vocab := mainvocab.NewVocabulary()
	for _, word := range words {
		vocab.AddToken(word)
	}

	log.Printf("Built vocabulary with %d words", len(vocab.TokenToWord))
	return vocab
}

// tokenizeAndAddWords tokenizes the input text and adds the tokens to the word set.
func tokenizeAndAddWords(text string, wordSet map[string]bool) {
	// Simple whitespace tokenizer; can be replaced with a more sophisticated one if needed
	tokens := strings.Fields(text)
	for _, token := range tokens {
		cleanedToken := strings.Trim(token, ".,!?\"'()[]{}:;`")
		if cleanedToken != "" {
			wordSet[cleanedToken] = true
		}
	}
}

// parseTrainingData reads and parses the training data file into query-intent pairs.
func parseTrainingData(filePath string) ([]struct {
	Query  string
	Intent string
}, error) {
	log.Printf("Parsing training data from %s", filePath)
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("could not open training data file: %w", err)
	}
	defer file.Close()

	var data []struct {
		Query  string
		Intent string
	}

	scanner := bufio.NewScanner(file)
	queryRe := regexp.MustCompile(`^Query: "(.*)"$`)
	intentRe := regexp.MustCompile(`^Intent: (.*)$`)

	currentQuery := ""
	currentIntent := ""

	log.Println("Starting to parse training data...")
	for scanner.Scan() {
		line := scanner.Text()
		log.Printf("Read line: %s", line)

		if matches := queryRe.FindStringSubmatch(line); len(matches) > 1 {
			// If we find a new query, and we have a pending query/intent pair, append it first
			if currentQuery != "" && currentIntent != "" {
				data = append(data, struct {
					Query  string
					Intent string
				}{Query: currentQuery, Intent: currentIntent})
				log.Printf("Appended training example: Query=\"%s\", Intent=\"%s\"", currentQuery, currentIntent)
			}
			currentQuery = matches[1]
			currentIntent = "" // Reset intent for the new query
			log.Printf("Matched Query: %s", currentQuery)
		} else if matches := intentRe.FindStringSubmatch(line); len(matches) > 1 {
			currentIntent = matches[1]
			log.Printf("Matched Intent: %s", currentIntent)
			// If we have both query and intent, append the example
			if currentQuery != "" && currentIntent != "" {
				data = append(data, struct {
					Query  string
					Intent string
				}{Query: currentQuery, Intent: currentIntent})
				log.Printf("Appended training example: Query=\"%s\", Intent=\"%s\"", currentQuery, currentIntent)
				currentQuery = ""  // Reset for next entry
				currentIntent = "" // Reset for next entry
			}
		} else if line == "" {
			log.Println("Empty line encountered.")
			// Do not reset currentQuery or currentIntent on blank lines.
			// They should only be reset after a full example is appended.
		}
	}

	// Append any remaining query/intent pair after the loop finishes
	if currentQuery != "" && currentIntent != "" {
		data = append(data, struct {
			Query  string
			Intent string
		}{Query: currentQuery, Intent: currentIntent})
		log.Printf("Appended final training example: Query=\"%%s\", Intent=\"%%s\"", currentQuery, currentIntent)
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading training data file: %w", err)
	}

	log.Printf("Finished parsing training data. Total examples: %d", len(data))
	return data, nil
}

// addWordsFromSoftwareCommands extracts words from the software commands JSON data.
func addWordsFromSoftwareCommands(filePath string, wordSet map[string]bool) {
	log.Printf("Expanding vocabulary with words from %s", filePath)
	file, err := os.Open(filePath)
	if err != nil {
		log.Printf("Warning: could not open software commands data to expand vocabulary: %v", err)
		return
	}
	defer file.Close()

	var data []struct {
		Query   string            
		Intent  string            
		Entities map[string]string 
	}
	if err := json.NewDecoder(file).Decode(&data); err != nil {
		log.Printf("Warning: could not decode software commands data from %s: %v", filePath, err)
		return
	}

	for _, entry := range data {
		tokenizeAndAddWords(entry.Query, wordSet)
		tokenizeAndAddWords(entry.Intent, wordSet)
		for _, value := range entry.Entities {
			tokenizeAndAddWords(value, wordSet)
		}
	}
}
