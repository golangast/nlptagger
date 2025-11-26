package semantic

import "fmt"

// TemplateFiller fills JSON templates with extracted entities
type TemplateFiller struct {
	templates map[IntentType]Template
}

// NewTemplateFiller creates a new template filler
func NewTemplateFiller() *TemplateFiller {
	return &TemplateFiller{
		templates: GetTemplates(),
	}
}

// Fill generates SemanticOutput from intent and entities
func (tf *TemplateFiller) Fill(intent IntentType, entities map[string]string) (SemanticOutput, error) {
	template, exists := tf.templates[intent]
	if !exists {
		return SemanticOutput{}, fmt.Errorf("no template for intent: %s", intent)
	}

	return template.Fill(entities), nil
}

// GetIntent returns the template for a given intent type
func (tf *TemplateFiller) GetIntent(intent IntentType) (Template, bool) {
	template, exists := tf.templates[intent]
	return template, exists
}
