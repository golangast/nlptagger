package main

import (
	"encoding/json"
	"fmt"
	"strings"

	"nlptagger/neural/nn/ner"
	"nlptagger/neural/semantic"
)

func main() {
	// Example query demonstrating structured command parsing
	query := "create folder jime with file Jill.go"

	fmt.Println("=== Structured Command Parser Demo ===")
	fmt.Printf("Query: %s\n\n", query)

	// Initialize parser
	parser := semantic.NewCommandParser()

	// Initialize NER with query and empty semantic output (uses pattern matching)
	nerSystem, err := ner.NewRuleBasedNER(query, "")
	if err != nil {
		fmt.Printf("Error initializing NER: %v\n", err)
		return
	}

	// Tokenize and get NER tags
	words := strings.Fields(query)
	entityMap := nerSystem.GetEntityMap()

	// Parse into structured command
	cmd := parser.Parse(query, words, entityMap)

	// Display structured command elements
	fmt.Println("=== Structured Command Elements ===")
	fmt.Printf("Action:        %s\n", cmd.Action)
	fmt.Printf("Object Type:   %s\n", cmd.ObjectType)
	fmt.Printf("Name:          %s\n", cmd.Name)
	fmt.Printf("Keyword:       %s\n", cmd.Keyword)
	fmt.Printf("Argument Type: %s\n", cmd.ArgumentType)
	fmt.Printf("Argument Name: %s\n", cmd.ArgumentName)
	fmt.Printf("Path:          %s\n", cmd.Path)
	fmt.Printf("\nCommand String: %s\n", cmd.String())
	fmt.Printf("Is Valid: %t\n", cmd.IsValid())
	fmt.Printf("Has Secondary: %t\n\n", cmd.HasSecondaryOperation())

	// Generate semantic output
	output := semantic.FillFromStructuredCommand(cmd)

	// Display JSON output
	jsonBytes, _ := json.MarshalIndent(output, "", "  ")
	fmt.Println("=== Semantic Output (JSON) ===")
	fmt.Println(string(jsonBytes))

	// Test more examples
	fmt.Println("\n\n=== Additional Examples ===")
	testCases := []string{
		"create folder src",
		"delete file temp.txt",
		"move file main.go to src",
		"rename folder old to new",
	}

	for _, tc := range testCases {
		ner, _ := ner.NewRuleBasedNER(tc, "")
		words := strings.Fields(tc)
		entityMap := ner.GetEntityMap()
		cmd := parser.Parse(tc, words, entityMap)
		fmt.Printf("\nQuery: %s\n", tc)
		fmt.Printf("  → %s\n", cmd.String())
	}
}
