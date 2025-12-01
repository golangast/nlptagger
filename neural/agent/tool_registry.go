package agent

import (
	"fmt"
)

// ToolSchema defines the structure of a tool's parameters
type ToolSchema struct {
	Parameters map[string]ParameterDef `json:"parameters"`
	Required   []string                `json:"required"`
}

// ParameterDef describes a single parameter
type ParameterDef struct {
	Type        string      `json:"type"` // "string", "number", "boolean", "array", "object"
	Description string      `json:"description"`
	Enum        []string    `json:"enum,omitempty"`
	Default     interface{} `json:"default,omitempty"`
}

// ToolResult represents the outcome of tool execution
type ToolResult struct {
	Success  bool                   `json:"success"`
	Output   string                 `json:"output"`
	Error    error                  `json:"error,omitempty"`
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// Tool interface for all executable tools
type Tool interface {
	Name() string
	Description() string
	Schema() ToolSchema
	Execute(args map[string]interface{}) (ToolResult, error)
}

// ToolRegistry manages available tools
type ToolRegistry struct {
	tools map[string]Tool
}

// NewToolRegistry creates a new tool registry
func NewToolRegistry() *ToolRegistry {
	return &ToolRegistry{
		tools: make(map[string]Tool),
	}
}

// RegisterTool adds a tool to the registry
func (tr *ToolRegistry) RegisterTool(tool Tool) error {
	if tool == nil {
		return fmt.Errorf("cannot register nil tool")
	}

	name := tool.Name()
	if name == "" {
		return fmt.Errorf("tool name cannot be empty")
	}

	if _, exists := tr.tools[name]; exists {
		return fmt.Errorf("tool already registered: %s", name)
	}

	tr.tools[name] = tool
	return nil
}

// GetTool retrieves a tool by name
func (tr *ToolRegistry) GetTool(name string) (Tool, error) {
	tool, exists := tr.tools[name]
	if !exists {
		return nil, fmt.Errorf("tool not found: %s", name)
	}
	return tool, nil
}

// ListTools returns all registered tool names
func (tr *ToolRegistry) ListTools() []string {
	names := make([]string, 0, len(tr.tools))
	for name := range tr.tools {
		names = append(names, name)
	}
	return names
}

// HasTool checks if a tool is registered
func (tr *ToolRegistry) HasTool(name string) bool {
	_, exists := tr.tools[name]
	return exists
}

// Execute runs a tool by name with given arguments
func (tr *ToolRegistry) Execute(toolName string, args map[string]interface{}) (ToolResult, error) {
	tool, err := tr.GetTool(toolName)
	if err != nil {
		return ToolResult{
			Success: false,
			Error:   err,
		}, err
	}

	return tool.Execute(args)
}
