package parser

import (
	"fmt"
	"strings"

	"nlptagger/neural/semantic"
	"nlptagger/neural/tokenizer"
	"nlptagger/neural/workflow"
	"nlptagger/tagger/nertagger"
	"nlptagger/tagger/postagger"
)

// Parser struct holds the necessary components for parsing.
type Parser struct {
	// Add necessary components like tokenizer, POS tagger, NER, etc.
}

// NewParser creates a new Parser.
func NewParser() *Parser {
	return &Parser{}
}

// Parse takes a natural language query and returns a structured workflow.
func (p *Parser) Parse(query string) (*workflow.Workflow, error) { // Changed return type
	// 1. Tokenize the input string.
	tokens := p.tokenize(query)

	// 2. Perform Part-of-Speech (POS) tagging.
	posTags := p.posTag(tokens)

	// 3. Perform Named Entity Recognition (NER).
	nerTags := p.nerTag(tokens, posTags)

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

	// For demonstration, set a default UserRole. In a real app, this would come from auth.
	output.Context.UserRole = "" // Default to guest for policy testing

	var folderResource *semantic.Resource
	var webserverResource *semantic.Resource
	var fileResource *semantic.Resource

	lastObjectType := ""

	for i, token := range tokens {
		nerTag := nerTags[i]

		switch nerTag {
		case "COMMAND":
			output.Operation = strings.ToUpper(token)
		case "OBJECT_TYPE":
			lastObjectType = token
			switch token {
			case "webserver":
				webserverResource = &semantic.Resource{Type: "Deployment::GoWebserver", Properties: map[string]interface{}{"port": 8080, "runtime_image": "golang:latest"}}
			case "folder":
				folderResource = &semantic.Resource{Type: "Filesystem::Folder", Properties: map[string]interface{}{"permissions": 0755}}
			case "file":
				fileResource = &semantic.Resource{Type: "Filesystem::File"}
			}
		case "NAME":
			if lastObjectType == "folder" && folderResource != nil {
				folderResource.Name = token
			} else if lastObjectType == "webserver" && webserverResource != nil {
				webserverResource.Name = token
			} else if lastObjectType == "file" && fileResource != nil {
				fileResource.Name = token
			}
		}
	}

	// Heuristic to find folder name if NER failed
	if folderResource != nil && folderResource.Name == "" {
		for i, token := range tokens {
			if token == "folder" {
				// Heuristic 1: folder called <name> or folder named <name>
				if i+2 < len(tokens) && (tokens[i+1] == "called" || tokens[i+1] == "named") {
					folderResource.Name = tokens[i+2]
					break
				}
				// Heuristic 2: <name> folder
				if i > 0 {
					prevToken := tokens[i-1]
					// Avoid grabbing keywords
					if prevToken != "a" && prevToken != "the" && prevToken != "create" {
						folderResource.Name = prevToken
						break
					}
				}
				// Heuristic 3: folder <name>
				if i+1 < len(tokens) {
					nextToken := tokens[i+1]
					if nextToken != "with" && nextToken != "and" && nextToken != "called" && nextToken != "named" && nextToken != "in" {
						folderResource.Name = nextToken
						break
					}
				}
			}
		}
	}

	// Heuristic to find webserver name if NER failed
	if webserverResource != nil && webserverResource.Name == "" {
		for i, token := range tokens {
			if token == "webserver" {
				// Heuristic 1: webserver called <name> or webserver named <name>
				if i+2 < len(tokens) && (tokens[i+1] == "called" || tokens[i+1] == "named") {
					webserverResource.Name = tokens[i+2]
					break
				}
				// Heuristic 2: webserver <name>
				if i+1 < len(tokens) {
					nextToken := tokens[i+1]
					if nextToken != "with" && nextToken != "and" {
						webserverResource.Name = nextToken
						break
					}
				}
			}
		}
	}

	// Heuristic for file name if NER failed
	if fileResource != nil && fileResource.Name == "" {
		for i, token := range tokens {
			if token == "file" {
				if i+1 < len(tokens) {
					fileResource.Name = tokens[i+1]
					break
				}
			}
		}
	}

	// Handle "go file" case
	if fileResource != nil && fileResource.Name != "" {
		isGoFile := false
		for i, token := range tokens {
			if token == "go" && i+1 < len(tokens) && tokens[i+1] == "file" {
				isGoFile = true
				break
			}
		}

		if isGoFile && !strings.HasSuffix(fileResource.Name, ".go") {
			fileResource.Name += ".go"
		}
	}

	// Heuristic for parent folder
	var parentFolder *semantic.Resource
	for i, token := range tokens {
		if token == "in" {
			if i+1 < len(tokens) {
				parentName := tokens[i+1]
				if parentName != "with" && parentName != "and" {
					parentFolder = &semantic.Resource{Type: "Filesystem::Folder", Name: parentName, Properties: map[string]interface{}{"permissions": 0755}}
					break
				}
			}
		}
	}

	// Establish parent-child relationship
	if parentFolder != nil {
		if folderResource != nil {
			if webserverResource != nil {
				folderResource.Children = append(folderResource.Children, *webserverResource)
			}
			if fileResource != nil {
				folderResource.Children = append(folderResource.Children, *fileResource)
			}
			parentFolder.Children = append(parentFolder.Children, *folderResource)
			output.TargetResource = *parentFolder
		} else {
			output.TargetResource = *parentFolder
			if webserverResource != nil {
				output.TargetResource.Children = append(output.TargetResource.Children, *webserverResource)
			}
			if fileResource != nil {
				output.TargetResource.Children = append(output.TargetResource.Children, *fileResource)
			}
		}
	} else if folderResource != nil {
		output.TargetResource = *folderResource
		if webserverResource != nil {
			output.TargetResource.Children = append(output.TargetResource.Children, *webserverResource)
		}
		if fileResource != nil {
			output.TargetResource.Children = append(output.TargetResource.Children, *fileResource)
		}
	} else if webserverResource != nil {
		output.TargetResource = *webserverResource
	} else if fileResource != nil {
		output.TargetResource = *fileResource
	}

	return output, nil
}
