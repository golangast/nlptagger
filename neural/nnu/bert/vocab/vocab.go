package vocabbert

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"

	"golang.org/x/net/html"

	"github.com/golangast/nlptagger/neural/nnu/bartsimple"
	"github.com/golangast/nlptagger/neural/nnu/vocab"
)

// setupVocabulary loads a vocabulary from vocabPath or builds a new one if loading fails.
func SetupVocabulary(vocabPath, trainingDataPath string) (*bartsimple.Vocabulary, error) {
	// Attempt to load the vocabulary first
	vocabulary, err := bartsimple.LoadVocabulary(vocabPath)
	if err == nil && vocabulary != nil && vocabulary.WordToToken != nil {
		// Validate that the loaded vocabulary contains the essential special tokens.
		// If not, we'll treat it as an invalid vocabulary and rebuild it.
		_, unkExists := vocabulary.WordToToken["[UNK]"]
		_, padExists := vocabulary.WordToToken["[PAD]"]
		_, bosExists := vocabulary.WordToToken["[BOS]"]
		_, eosExists := vocabulary.WordToToken["[EOS]"]

		if unkExists && padExists && bosExists && eosExists {
			// All essential tokens exist, set the IDs and return.
			vocabulary.UnknownTokenID = vocabulary.WordToToken["[UNK]"]
			vocabulary.PaddingTokenID = vocabulary.WordToToken["[PAD]"]
			vocabulary.BeginningOfSentenceID = vocabulary.WordToToken["[BOS]"]
			vocabulary.EndOfSentenceID = vocabulary.WordToToken["[EOS]"]
			return vocabulary, nil
		}
		fmt.Println("Loaded vocabulary is missing one or more special tokens. Rebuilding.")
	} else if err != nil {
		// If there was an error loading (e.g., file not found), print it and proceed to build.
		fmt.Printf("Error loading vocabulary: %v. Building a new one from training data.\n", err)
	}

	// If loading fails, build a new one

	trainingData, loadErr := vocab.LoadTrainingDataJSON(trainingDataPath)
	if loadErr != nil {
		return nil, fmt.Errorf("error loading training data to build new vocabulary: %w", loadErr)
	}

	fmt.Println("Building new vocabulary...")
	allWords := make(map[string]bool)

	// Gather words from all sources.
	addWordsFromJSONTrainingData(trainingData, allWords)
	addWordsFromRAGDocs("trainingdata/ragdata/ragdocs.txt", allWords)
	addWordsFromRAGJSON("trainingdata/ragdata/rag_data.json", allWords)
	addWordsFromHTML("docs/index.html", allWords)

	// 5. Expand vocabulary with words from WikiQA data
	addWordsFromWikiQA("trainingdata/WikiQA-train.txt", allWords)


	newVocab := bartsimple.NewVocabulary()

	// Add special tokens first to ensure they have consistent IDs
	newVocab.AddToken("[PAD]", 0)
	newVocab.AddToken("[UNK]", 1)
	newVocab.AddToken("[BOS]", 2)
	newVocab.AddToken("[EOS]", 3)

	for word := range allWords {
		// Avoid re-adding special tokens
		if _, exists := newVocab.WordToToken[word]; !exists {
			newVocab.AddToken(word, len(newVocab.TokenToWord))
		}
	}

	newVocab.PaddingTokenID = newVocab.WordToToken["[PAD]"]
	newVocab.UnknownTokenID = newVocab.WordToToken["[UNK]"]
	newVocab.BeginningOfSentenceID = newVocab.WordToToken["[BOS]"]
	newVocab.EndOfSentenceID = newVocab.WordToToken["[EOS]"]

	// Save the new vocabulary
	if err := newVocab.Save(vocabPath); err != nil {
		// Log the error but continue, as we have a functional vocabulary in memory
		fmt.Printf("Warning: Error saving newly built vocabulary: %v\n", err)
	} else {
		fmt.Printf("Saved new vocabulary to %s\n", vocabPath)
	}

	return newVocab, nil
}

// addWordsFromJSONTrainingData extracts words from the primary annotated training data.
func addWordsFromJSONTrainingData(trainingData *vocab.TrainingDataJSON, wordSet map[string]bool) {
	fmt.Println("Expanding vocabulary with words from JSON training data...")
	tokenVocab := vocab.CreateTokenVocab(trainingData.Sentences)
	for word := range tokenVocab {
		wordSet[word] = true
	}
}

// addWordsFromRAGDocs extracts words from the plain text RAG documents
func addWordsFromRAGDocs(path string, wordSet map[string]bool) {
	fmt.Printf("Expanding vocabulary with words from %s\n", path)
	file, err := os.Open(path)
	if err != nil {
		fmt.Printf("Warning: could not open RAG docs to expand vocabulary: %v\n", err)
		return
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		words := strings.Fields(scanner.Text())
		for _, word := range words {
			cleanedWord := strings.ToLower(strings.Trim(word, ".,!?;:\"'()[]{}-_"))
			if cleanedWord != "" {
				wordSet[cleanedWord] = true
			}
		}
	}
}

// addWordsFromWikiQA extracts words from the WikiQA training data.
func addWordsFromWikiQA(path string, wordSet map[string]bool) {
	fmt.Printf("Expanding vocabulary with words from %s\n", path)
	file, err := os.Open(path)
	if err != nil {
		fmt.Printf("Warning: could not open WikiQA data to expand vocabulary: %v\n", err)
		return
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.SplitN(line, "\t", 4) // Split into max 4 parts
		if len(parts) > 1 {                   // Ensure there's at least a question and answer
			question := parts[0]             // The question
			words := strings.Fields(question) // Then process the question
			for _, word := range words {
				cleanedWord := strings.ToLower(strings.Trim(word, ".,!?;:\"'()[]{}-_"))
				if cleanedWord != "" {
					wordSet[cleanedWord] = true
				}
			}
		}
	}
}

// addWordsFromRAGJSON extracts words from the structured RAG JSON data.
func addWordsFromRAGJSON(path string, wordSet map[string]bool) {
	fmt.Printf("Expanding vocabulary with words from %s\n", path)
	file, err := os.Open(path)
	if err != nil {
		fmt.Printf("Warning: could not open RAG JSON data to expand vocabulary: %v\n", err)
		return
	}
	defer file.Close()

	var ragData []struct {
		Content string `json:"Content"`
	}
	if err := json.NewDecoder(file).Decode(&ragData); err != nil {
		fmt.Printf("Warning: could not decode RAG JSON data from %s: %v\n", path, err)
		return
	}

	for _, entry := range ragData {
		words := strings.Fields(entry.Content)
		for _, word := range words {
			cleanedWord := strings.ToLower(strings.Trim(word, ".,!?;:\"'()[]{}-_"))
			if cleanedWord != "" {
				wordSet[cleanedWord] = true
			}
		}
	}
}

// addWordsFromHTML extracts text content from an HTML file to expand the vocabulary.
func addWordsFromHTML(path string, wordSet map[string]bool) {
	fmt.Printf("Expanding vocabulary with words from %s\n", path)
	file, err := os.Open(path)
	if err != nil {
		fmt.Printf("Warning: could not open docs file to expand vocabulary: %v\n", err)
		return
	}
	defer file.Close()

	tokenizer := html.NewTokenizer(file)
htmlLoop:
	for {
		tt := tokenizer.Next()
		switch tt {
		case html.ErrorToken:
			if tokenizer.Err() != io.EOF {
				fmt.Printf("Warning: error parsing HTML from %s: %v\n", path, tokenizer.Err())
			}
			break htmlLoop // End of the document
		case html.TextToken:
			words := strings.Fields(string(tokenizer.Text()))
			for _, word := range words {
				cleanedWord := strings.ToLower(strings.Trim(word, ".,!?;:\"'()[]{}-_"))
				if cleanedWord != "" {
					wordSet[cleanedWord] = true
				}
			}
		}
	}
}

