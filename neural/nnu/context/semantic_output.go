package context

// SemanticOutput represents the top-level structure of the semantic output.
type SemanticOutput struct {
	Operation     string        `json:"operation"`
	TargetResource TargetResource `json:"target_resource"`
	Context       Context       `json:"context"`
}

// TargetResource represents the target resource in the semantic output.
type TargetResource struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Properties  map[string]interface{} `json:"properties"`
	Directory   string          `json:"directory,omitempty"`
	Destination string          `json:"destination,omitempty"`
	Children    []TargetResource `json:"children,omitempty"`
}

// Context represents the context in the semantic output.
type Context struct {
	UserRole string `json:"user_role"`
}
