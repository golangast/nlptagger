package main

import (
	"encoding/json"
	"fmt"
	"strings"

	"nlptagger/neural/nn/ner"
	"nlptagger/neural/semantic"
)

func main() {
	fmt.Println("=== Hierarchical Project Scaffolding Demo ===\n")

	// Test cases for hierarchical commands
	testCases := []string{
		"create folder myproject with webserver",
		"create folder myapp with api",
		"create folder mytool with cli",
	}

	parser := semantic.NewHierarchicalParser()

	for i, query := range testCases {
		fmt.Printf("--- Example %d ---\n", i+1)
		fmt.Printf("Query: %s\n\n", query)

		// Parse with NER
		nerSystem, _ := ner.NewRuleBasedNER(query, "")
		words := strings.Fields(query)
		entityMap := nerSystem.GetEntityMap()

		// Parse to hierarchical command
		cmd := parser.Parse(query, words, entityMap)

		// Display hierarchical structure
		fmt.Println("Hierarchical Command Tree:")
		fmt.Println(cmd.String())
		fmt.Println()

		// Generate semantic output
		output := semantic.FillFromHierarchicalCommand(cmd)

		// Display JSON
		jsonBytes, _ := json.MarshalIndent(output, "", "  ")
		fmt.Println("Generated Semantic Output:")
		fmt.Println(string(jsonBytes))
		fmt.Println("\n" + strings.Repeat("=", 60) + "\n")
	}

	// Advanced example with explicit files
	fmt.Println("--- Advanced Example ---")
	advancedQuery := "create folder myproject with webserver in the file main.go with handler.go and a folder for templates"
	fmt.Printf("Query: %s\n\n", advancedQuery)

	nerSystem, _ := ner.NewRuleBasedNER(advancedQuery, "")
	words := strings.Fields(advancedQuery)
	entityMap := nerSystem.GetEntityMap()

	cmd := parser.Parse(advancedQuery, words, entityMap)

	fmt.Println("Hierarchical Command Tree:")
	fmt.Println(cmd.String())
	fmt.Println()

	output := semantic.FillFromHierarchicalCommand(cmd)
	jsonBytes, _ := json.MarshalIndent(output, "", "  ")
	fmt.Println("Generated Semantic Output:")
	fmt.Println(string(jsonBytes))
}
