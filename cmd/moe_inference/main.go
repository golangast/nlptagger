package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"strings" // Added for string manipulation

	"nlptagger/neural/nn/ner"
	"nlptagger/neural/semantic"
)

var (
	query             = flag.String("query", "", "Query for MoE inference")
	maxSeqLength      = flag.Int("maxlen", 32, "Maximum sequence length")
	temperature       = flag.Float64("temperature", 0.8, "Sampling temperature (0.0 = deterministic, 1.0 = normal, >1.0 = more random)")
	samplingMethod    = flag.String("sampling-method", "temperature", "Sampling method: greedy, temperature, top-k, top-p")
	topK              = flag.Int("top-k", 0, "Top-k sampling: only sample from top K tokens (0 = disabled)")
	topP              = flag.Float64("top-p", 0.0, "Top-p (nucleus) sampling: sample from tokens with cumulative probability <= p (0.0 = disabled)")
	repetitionPenalty = flag.Float64("repetition-penalty", 1.0, "Repetition penalty (1.0 = no penalty, > 1.0 = penalize repetition)")
)

func main() {
	rand.Seed(1) // Seed the random number generator for deterministic behavior
	flag.Parse()

	if *query == "" {
		log.Fatal("Please provide a query using the -query flag.")
	}

	log.Printf("Running template-based inference for query: \"%s\"", *query)

	// === TEMPLATE-BASED APPROACH ===
	// We don't need the model, vocabularies, or tokenizers anymore
	// Template-based approach uses:
	// 1. Intent classification (keyword-based)
	// 2. Entity extraction (NER)
	// 3. Template filling (deterministic)

	log.Println("Using template-based JSON generation")

	// Step 1: Classify intent from query
	classifier := semantic.NewIntentClassifier()
	intent := classifier.Classify(*query)
	log.Printf("Classified intent: %s", intent)

	// Step 2: Extract entities using NER
	ruleNER, err := ner.NewRuleBasedNER(*query, "")
	if err != nil {
		log.Fatalf("Failed to create NER: %v", err)
	}

	entityMap := ruleNER.GetEntityMap()
	extractor := semantic.NewEntityExtractor()
	entities := extractor.ExtractFromQuery(*query, entityMap)

	log.Printf("Extracted entities: %v", entities)

	// Check if query contains template keywords
	words := strings.Fields(*query)
	templateRegistry := semantic.NewTemplateRegistry()
	hasTemplate := false
	for _, word := range words {
		lowerWord := strings.ToLower(word)
		for _, tmpl := range templateRegistry.ListTemplates() {
			if lowerWord == strings.ToLower(tmpl) {
				hasTemplate = true
				break
			}
		}
		if hasTemplate {
			break
		}
	}

	var semanticOutput semantic.SemanticOutput
	var structuredCmd *semantic.StructuredCommand
	var hierarchicalCmd *semantic.HierarchicalCommand

	if hasTemplate {
		// Use hierarchical parser for template-based scaffolding
		hierarchicalParser := semantic.NewHierarchicalParser()
		hierarchicalCmd = hierarchicalParser.Parse(*query, words, entityMap)
		semanticOutput = semantic.FillFromHierarchicalCommand(hierarchicalCmd)
	} else {
		// Use standard parser for simple commands
		parser := semantic.NewCommandParser()
		structuredCmd = parser.Parse(*query, words, entityMap)

		// Step 3: Fill template with entities
		filler := semantic.NewTemplateFiller()
		var err error
		semanticOutput, err = filler.Fill(intent, entities)
		if err != nil {
			log.Fatalf("Failed to fill template: %v", err)
		}
	}

	// Step 4: Marshal to JSON
	jsonBytes, err := json.MarshalIndent(semanticOutput, "", "  ")
	if err != nil {
		log.Fatalf("Failed to marshal JSON: %v", err)
	}

	// Display command pattern
	if hasTemplate && hierarchicalCmd != nil {
		fmt.Println("\n=== Hierarchical Command Tree ===")
		fmt.Println(hierarchicalCmd.String())
		fmt.Println("==================================")
	} else if structuredCmd != nil {
		fmt.Println("\n=== Structured Command Pattern ===")
		fmt.Printf("Action:        %s\n", structuredCmd.Action)
		fmt.Printf("Object Type:   %s\n", structuredCmd.ObjectType)
		fmt.Printf("Name:          %s\n", structuredCmd.Name)
		if structuredCmd.Keyword != "" {
			fmt.Printf("Keyword:       %s\n", structuredCmd.Keyword)
		}
		if structuredCmd.ArgumentType != "" {
			fmt.Printf("Argument Type: %s\n", structuredCmd.ArgumentType)
		}
		if structuredCmd.ArgumentName != "" {
			fmt.Printf("Argument Name: %s\n", structuredCmd.ArgumentName)
		}
		fmt.Printf("\nPattern: %s\n", structuredCmd.String())
		fmt.Println("===================================")
	}

	fmt.Println("\n=== Generated Semantic Output ===")
	fmt.Println(string(jsonBytes))
	fmt.Println("=================================")

	// --- Named Entity Recognition (Rule-Based) ---
	fmt.Println("\n--- Named Entity Recognition (Rule-Based) ---")

	// Display entities (reuse words from earlier)
	for i, word := range words {

		entityType := entityMap[i]
		fmt.Printf("Word: %s, Type: %s\n", word, entityType)
	}
	fmt.Println("--------------------------------------------")
}
