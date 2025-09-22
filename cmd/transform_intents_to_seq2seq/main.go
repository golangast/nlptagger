package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"strings"
)

// IntentExample represents the input structure from generated_wikiqa_intents.json
type IntentExample struct {
	Query        string `json:"query"`
	ParentIntent string `json:"parent_intent"`
	ChildIntent  string `json:"child_intent"`
}

// Seq2SeqTrainingExample represents the output structure for train_moe
type Seq2SeqTrainingExample struct {
	Query       string `json:"Query"`
	Description string `json:"description"`
}

func main() {
	inputFilePath := "trainingdata/generated_wikiqa_intents.json"
	outputFilePath := "trainingdata/wikiqa_seq2seq_training.json"

	// Read the input JSON file
	inputBytes, err := ioutil.ReadFile(inputFilePath)
	if err != nil {
		fmt.Printf("Error reading input file %s: %v\n", inputFilePath, err)
		return
	}

	var inputData []IntentExample
	err = json.Unmarshal(inputBytes, &inputData)
	if err != nil {
		fmt.Printf("Error unmarshaling input JSON from %s: %v\n", inputFilePath, err)
		return
	}

	var outputData []Seq2SeqTrainingExample
	for _, item := range inputData {
		// Concatenate parent_intent and child_intent for the Description field
		description := strings.Join([]string{item.ParentIntent, item.ChildIntent}, "_")
		outputData = append(outputData, Seq2SeqTrainingExample{
			Query:       item.Query,
			Description: description,
		})
	}

	// Marshal the output data to JSON
	outputBytes, err := json.MarshalIndent(outputData, "", "  ")
	if err != nil {
		fmt.Printf("Error marshaling output JSON: %v\n", err)
		return
	}

	// Write the output JSON to file
	err = ioutil.WriteFile(outputFilePath, outputBytes, 0644)
	if err != nil {
		fmt.Printf("Error writing output file %s: %v\n", outputFilePath, err)
		return
	}

	fmt.Printf("Successfully transformed data from %s to %s\n", inputFilePath, outputFilePath)
}