package context

import (
	"fmt"
	"slices"
	"strings"

	"github.com/zendrulat/nlptagger/tagger"
)

// Entity represents a recognized entity with its type and value.
type Entity struct {
	Type  string
	Value string
}

// ConversationContext stores the history of a conversation for context and co-reference resolution.
type ConversationContext struct {
	MaxTurns        int
	TurnHistory     []string   // Raw text of past user inputs
	LastIntents     []string   // Last N predicted intents
	LastEntities    [][]Entity // Last N sets of recognized entities
	CurrentIntent   string     // The most recently predicted intent
	CurrentEntities []Entity   // The most recently recognized entities
}

// NewConversationContext initializes a new ConversationContext with a specified history size.
func NewConversationContext(maxTurns int) *ConversationContext {
	return &ConversationContext{
		MaxTurns:     maxTurns,
		TurnHistory:  make([]string, 0, maxTurns),
		LastIntents:  make([]string, 0, maxTurns),
		LastEntities: make([][]Entity, 0, maxTurns),
	}
}

// AddTurn adds the current turn's information to the conversation context.
func (cc *ConversationContext) AddTurn(intent string, entities []Entity, rawText string) {
	// Update current intent and entities
	cc.CurrentIntent = intent
	cc.CurrentEntities = entities

	// Add to history and manage size
	cc.TurnHistory = append(cc.TurnHistory, rawText)
	cc.LastIntents = append(cc.LastIntents, intent)
	cc.LastEntities = append(cc.LastEntities, entities)

	if len(cc.TurnHistory) > cc.MaxTurns {
		cc.TurnHistory = cc.TurnHistory[1:]
		cc.LastIntents = cc.LastIntents[1:]
		cc.LastEntities = cc.LastEntities[1:]
	}
}

// GetLastIntent retrieves the most recent intent from the history.
func (cc *ConversationContext) GetLastIntent() string {
	if len(cc.LastIntents) > 0 {
		return cc.LastIntents[len(cc.LastIntents)-1]
	}
	return ""
}

// GetLastEntities retrieves the most recent set of entities from the history.
func (cc *ConversationContext) GetLastEntities() []Entity {
	if len(cc.LastEntities) > 0 {
		return cc.LastEntities[len(cc.LastEntities)-1]
	}
	return nil
}

// ResolveCoReference attempts to resolve co-references in the input text based on the conversation context.
func (cc *ConversationContext) ResolveCoReference(inputText string) string {
	resolvedText := inputText
	lowerInput := strings.ToLower(inputText)

	taggedEntities := tagger.Tagging(lowerInput)

	// Iterate through each pronoun to see if it's in the input
	for _, pronoun := range taggedEntities.Tokens {
		if slices.Contains(taggedEntities.PosTag, "PRP") || taggedEntities.IsName {
			// if strings.Contains(lowerInput, pronoun) {
			// If a pronoun is found, search backwards through the conversation history for an entity to replace it with.
			for i := len(cc.LastEntities) - 1; i >= 0; i-- {
				entities := cc.LastEntities[i]
				if len(entities) > 0 {
					// A simple strategy: replace the pronoun with the value of the first entity found.
					// This can be improved with more sophisticated entity-type matching.
					replacementEntity := entities[0]
					resolvedText = strings.Replace(resolvedText, pronoun, replacementEntity.Value, 1)
					fmt.Printf("Resolved '%s' to '%s'. New text: '%s'\n", pronoun, replacementEntity.Value, resolvedText)
					// After a successful replacement, we update lowerInput to avoid multiple replacements of the same pronoun.
					// lowerInput = strings.ToLower(resolvedText)
					break // Move to the next pronoun
				}
			}
		}
	}

	return resolvedText
}
