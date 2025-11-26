package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"

	"github.com/zendrulat/nlptagger/neural/semantic"
	"github.com/zendrulat/nlptagger/neural/tokenizer"
)

// IntentTrainingExample represents a single training example from the old format.
type IntentTrainingExample struct {
	Query          string                  `json:"query"`
	SemanticOutput semantic.SemanticOutput `json:"semantic_output"`
}

// IntentTrainingData represents the structure of the old intent training data JSON.
type IntentTrainingData []IntentTrainingExample

// TaggedTrainingExample represents the new format for training the NER/tagging model.
type TaggedTrainingExample struct {
	Query  string   `json:"query"`
	Intent string   `json:"intent"`
	Tokens []string `json:"tokens"`
	Tags   []string `json:"tags"` // IOB format (Inside, Outside, Beginning)
}

// LoadIntentTrainingData loads the intent training data from a JSON file.
func LoadIntentTrainingData(filePath string) (*IntentTrainingData, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open training data file %s: %w", filePath, err)
	}
	defer file.Close()

	bytes, err := io.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("failed to read training data file %s: %w", filePath, err)
	}

	var data IntentTrainingData
	err = json.Unmarshal(bytes, &data)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal training data JSON from %s: %w", filePath, err)
	}

	return &data, nil
}

func main() {
	const intentDataPath = "trainingdata/semantic_output_data.json"
	const taggedDataPath = "trainingdata/tagged_training_data.json"

	// Load the original training data
	trainingData, err := LoadIntentTrainingData(intentDataPath)
	if err != nil {
		log.Fatalf("Failed to load training data: %v", err)
	}

	var taggedTrainingData []TaggedTrainingExample

	for _, example := range *trainingData {
		tokens := tokenizer.Tokenize(example.Query)
		tags := make([]string, len(tokens))
		for i := range tags {
			tags[i] = "O" // Default to Outside
		}

		// Extract intent
		intent := example.SemanticOutput.Operation

		// Create a map of entities to tag
		entitiesToTag := make(map[string]string)
		if example.SemanticOutput.TargetResource.Name != "" {
			entitiesToTag[example.SemanticOutput.TargetResource.Name] = "resource_name"
		}
		if example.SemanticOutput.TargetResource.Type != "" {
			// This is tricky because the type is not always in the query
			// For now, we will only tag entities that are directly in the query.
		}
		// Add other entities from properties if they exist in the query

		// Tag entities in the query
		for entityValue, entityType := range entitiesToTag {
			entityTokens := tokenizer.Tokenize(entityValue)
			for i := 0; i <= len(tokens)-len(entityTokens); i++ {
				match := true
				for j := 0; j < len(entityTokens); j++ {
					if tokens[i+j] != entityTokens[j] {
						match = false
						break
					}
				}

				if match {
					tags[i] = "B-" + entityType // Beginning of entity
					for j := 1; j < len(entityTokens); j++ {
						tags[i+j] = "I-" + entityType // Inside of entity
					}
					// To avoid tagging overlaps, we could break here
					// but for now, we allow multiple tags if entities overlap.
				}
			}
		}

		taggedExample := TaggedTrainingExample{
			Query:  example.Query,
			Intent: intent,
			Tokens: tokens,
			Tags:   tags,
		}
		taggedTrainingData = append(taggedTrainingData, taggedExample)
	}

	// Save the new tagged data
	file, err := os.Create(taggedDataPath)
	if err != nil {
		log.Fatalf("Failed to create tagged training data file: %v", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(taggedTrainingData); err != nil {
		log.Fatalf("Failed to encode tagged training data: %v", err)
	}

	log.Printf("Successfully converted %d examples to tagged format.", len(taggedTrainingData))
}
