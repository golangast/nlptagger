package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/zendrulat/nlptagger/tagger"
	"github.com/zendrulat/nlptagger/tagger/tag"
)

// IntentExample represents a single intent training example
type IntentExample struct {
	Query        string `json:"query"`
	ParentIntent string `json:"parent_intent"`
	ChildIntent  string `json:"child_intent"`
}

// Function to determine intent based on linguistic tags
func determineIntent(taggedSentence tag.Tag) (string, string) {
	parentIntent := "general"
	childIntent := "query"

	// Look for verbs to determine child_intent (action)
	for i, pos := range taggedSentence.PosTag {
		// Declare token outside the switch to ensure it's always in scope
		token := strings.ToLower(taggedSentence.Tokens[i])
		switch pos {
		case "VB", "VBD", "VBG", "VBN", "VBP", "VBZ": // All verb forms
			if strings.Contains(token, "how many") || strings.Contains(token, "how much") {
				childIntent = "get"
				parentIntent = "quantity"
			} else if strings.Contains(token, "is") || strings.Contains(token, "are") || strings.Contains(token, "what") {
				childIntent = "get"
				parentIntent = "definition"
			} else if strings.Contains(token, "form") || strings.Contains(token, "make") || strings.Contains(token, "build") {
				childIntent = "create"
				parentIntent = "process"
			} else if strings.Contains(token, "die") || strings.Contains(token, "kill") {
				childIntent = "event"
				parentIntent = "history"
			} else if strings.Contains(token, "work") || strings.Contains(token, "function") {
				childIntent = "explain"
				parentIntent = "process"
			} else if strings.Contains(token, "get") || strings.Contains(token, "find") || strings.Contains(token, "retrieve") {
				childIntent = "get"
			} else if strings.Contains(token, "do") || strings.Contains(token, "did") {
				childIntent = "explain"
			}
			// More specific verb-based rules can be added here
			break // Found a verb, prioritize it
		}
	}

	// Look for specific nouns/entities to refine parent_intent (domain)
	for _, ner := range taggedSentence.NerTag {
		switch ner {
		case "PERSON":
			parentIntent = "person"
			childIntent = "identify"
		case "LOCATION":
			parentIntent = "location"
			childIntent = "get"
		case "ORGANIZATION":
			parentIntent = "organization"
			childIntent = "get"
		case "DATE", "TIME":
			parentIntent = "time"
			childIntent = "get"
		}
	}

	// Fallback to general if no specific intent found
	if parentIntent == "general" && childIntent == "query" {
		lowerQuery := strings.ToLower(taggedSentence.Sentence)
		if strings.HasPrefix(lowerQuery, "how many") || strings.HasPrefix(lowerQuery, "how much") {
			parentIntent = "quantity"
			childIntent = "get"
		} else if strings.HasPrefix(lowerQuery, "what is") || strings.HasPrefix(lowerQuery, "what does") {
			parentIntent = "definition"
			childIntent = "get"
		} else if strings.HasPrefix(lowerQuery, "how are") || strings.HasPrefix(lowerQuery, "how do") {
			parentIntent = "process"
			childIntent = "explain"
		} else if strings.HasPrefix(lowerQuery, "where is") {
			parentIntent = "location"
			childIntent = "get"
		} else if strings.HasPrefix(lowerQuery, "who is") {
			parentIntent = "person"
			childIntent = "identify"
		} else if strings.HasPrefix(lowerQuery, "when was") {
			parentIntent = "time"
			childIntent = "get"
		}
	}

	return parentIntent, childIntent
}

func main() {
	inputFilePath := "trainingdata/WikiQA-train.txt"
	outputFilePath := "trainingdata/generated_wikiqa_intents.json"

	inputFile, err := os.Open(inputFilePath)
	if err != nil {
		fmt.Printf("Error opening input file: %v\n", err)


		return
	}
	defer inputFile.Close()

	var generatedData []IntentExample
	scanner := bufio.NewScanner(inputFile)

	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Split(line, "\t")
		if len(parts) >= 1 {
			query := strings.TrimSpace(parts[0])
			if query == "" {
				continue
			}

			// Use the tagger to get linguistic information
			var taggedSentence tag.Tag
			func() {
				defer func() {
					if r := recover(); r != nil {
						fmt.Printf("Warning: Panic during tagging for query '%s': %v\n", query, r)
						// Initialize taggedSentence with basic info to avoid nil pointer dereference
					taggedSentence = tag.Tag{Sentence: query, Tokens: strings.Fields(query), PosTag: make([]string, len(strings.Fields(query))), NerTag: make([]string, len(strings.Fields(query)))}
					}
				}()
			taggedSentence = tagger.Tagging(query)
			}()

			parentIntent, childIntent := determineIntent(taggedSentence)

			generatedData = append(generatedData, IntentExample{
				Query:        query,
				ParentIntent: parentIntent,
				ChildIntent:  childIntent,
			})
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Printf("Error reading input file: %v\n", err)
		return
	}

	outputFile, err := os.Create(outputFilePath)
	if err != nil {
		fmt.Printf("Error creating output file: %v\n", err)

		return
	}
	defer outputFile.Close()

	encoder := json.NewEncoder(outputFile)
	encoder.SetIndent("", "  ") // Pretty print JSON
	if err := encoder.Encode(generatedData); err != nil {
		fmt.Printf("Error encoding JSON to file: %v\n", err)
		return
	}

	fmt.Printf("Successfully generated intent data to %s\n", outputFilePath)
}
