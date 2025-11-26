package parser

import (
	"fmt"
	"log"

	"github.com/zendrulat/nlptagger/neural/semantic"
	"github.com/zendrulat/nlptagger/neural/tokenizer"
	"github.com/zendrulat/nlptagger/neural/workflow"
	"github.com/zendrulat/nlptagger/tagger/nertagger"
	"github.com/zendrulat/nlptagger/tagger/postagger"
)

// Parser struct holds the necessary components for parsing.
type Parser struct {
	// Add necessary components like tokenizer, POS tagger, NER, etc.
	ruleEngine *ParsingRuleEngine
}

// NewParser creates a new Parser.
func NewParser() *Parser {
	re := NewParsingRuleEngine()
	re.RegisterDefaultParsingRules()
	return &Parser{
		ruleEngine: re,
	}
}

// Parse takes a natural language query and returns a structured workflow.
func (p *Parser) Parse(query string) (*workflow.Workflow, error) { // Changed return type
	// 1. Tokenize the input string.
	tokens := p.tokenize(query)

	// 2. Perform Part-of-Speech (POS) tagging.
	posTags := p.posTag(tokens)

	// 3. Perform Named Entity Recognition (NER).
	nerTags := p.nerTag(tokens, posTags)

	log.Printf("Tokens: %v", tokens)
	log.Printf("POS Tags: %v", posTags)
	log.Printf("NER Tags: %v", nerTags)

	// 4. Map tokens, POS tags, and NER tags to semantic output.
	semanticOutput, err := p.mapToSemanticOutput(tokens, posTags, nerTags)
	if err != nil {
		return nil, fmt.Errorf("failed to map to semantic output: %w", err)
	}

	// Validate and infer properties for the semantic output
	if err := semantic.ValidateAndInferProperties(semanticOutput); err != nil {
		return nil, fmt.Errorf("semantic validation and inference failed: %w", err)
	}

	// Generate workflow from semantic output
	wf, err := workflow.GenerateWorkflow(semanticOutput)
	if err != nil {
		return nil, fmt.Errorf("failed to generate workflow: %w", err)
	}

	return wf, nil // Changed return value
}

func (p *Parser) tokenize(query string) []string {
	return tokenizer.Tokenize(query)
}

func (p *Parser) posTag(tokens []string) []string {
	return postagger.TagTokens(tokens)
}

func (p *Parser) nerTag(tokens []string, posTags []string) []string {
	return nertagger.TagTokens(tokens, posTags)
}

func (p *Parser) mapToSemanticOutput(tokens, posTags, nerTags []string) (*semantic.SemanticOutput, error) {
	output := &semantic.SemanticOutput{}
	output.Context.UserRole = "" // Default to guest for policy testing

	var folderResource *semantic.Resource
	var webserverResource *semantic.Resource
	var fileResource *semantic.Resource
	var lastFolderResource *semantic.Resource

	// lastObjectType is no longer needed as rules will directly set resource types
	// var lastObjectType string
	var lastCreatedResourceName string
	log.Print(lastCreatedResourceName)
	var lastProcessedResource *semantic.Resource // Track the actual resource object

	expectingDependencyTarget := false
	expectingDependencySource := false
	dependencyType := "" // "THEN", "AFTER", "BEFORE"

	// Helper to get resource name from token and type - this is now in rules.go
	// getResourceName := func(idx int, objType string) string { ... }

	for i := 0; i < len(tokens); i++ {
		applied, tokensConsumed, err := p.ruleEngine.ApplyRules(tokens, posTags, nerTags, i, output,
			&fileResource, &folderResource, &webserverResource, &lastFolderResource,
			&lastProcessedResource, &expectingDependencyTarget, &expectingDependencySource, &dependencyType)
		if err != nil {
			return nil, err
		}
		if applied {
			i += tokensConsumed - 1 // Adjust index by tokens consumed by the rule
			continue
		}

		// The remaining heuristic-based logic will be gradually moved into rules.
		// For now, it's removed from here as the rule engine is the primary mechanism.
		// If no rule applies, we simply move to the next token.
	}

	log.Printf("Before final target assignment, output.TargetResource: %+v", output.TargetResource)

	var target *semantic.Resource
	if fileResource != nil {
		target = fileResource
	} else if folderResource != nil {
		target = folderResource
	} else if webserverResource != nil {
		target = webserverResource
	}
	output.TargetResource = target

	log.Printf("SemanticOutput TargetResource Type before returning: %s", output.TargetResource.Type)
	return output, nil
}
