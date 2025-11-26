package semantic

// SemanticOutput represents the structured output from the NLP layer.
type SemanticOutput struct {
	Operation          string               `json:"operation"`
	Command            string               `json:"command,omitempty"` // Added Command field
	TargetResource     *Resource            `json:"target_resource"`
	Context            Context              `json:"context"`
	ExplicitDependencies []TemporalDependency `json:"explicit_dependencies,omitempty"`
}

// TemporalDependency represents an explicit temporal dependency between two resources.
type TemporalDependency struct {
	SourceResourceName string
	TargetResourceName string
	Type               string // e.g., "AFTER", "BEFORE", "THEN
}

// Resource represents a single entity or component in the system.
type Resource struct {
	Type       string                 `json:"type"`
	Name       string                 `json:"name"`
	Content    string                 `json:"content,omitempty"` // Add this line
	Properties map[string]interface{} `json:"properties"`
	Children   []Resource             `json:"children,omitempty"`
	DependsOn  []string               `json:"depends_on,omitempty"`
	Directory  string                 `json:"directory,omitempty"` // Add this line
	Destination string              `json:"destination,omitempty"`
}

// Context provides additional information about the request environment.
type Context struct {
	UserRole string `json:"user_role"`
}
