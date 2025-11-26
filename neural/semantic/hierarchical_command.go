package semantic

// HierarchicalCommand represents a command that can have multiple nested children
// This extends StructuredCommand to support complex project scaffolding
type HierarchicalCommand struct {
	// Primary operation
	Action     CommandAction // e.g., "create"
	ObjectType ObjectType    // e.g., "folder"
	Name       string        // e.g., "myproject"

	// Template/boilerplate to apply
	Template string // e.g., "webserver", "react-app", "go-cli"

	// Nested children (files, folders, etc.)
	Children []*HierarchicalCommand

	// Additional context
	Path       string            // Path/location for the operation
	Properties map[string]string // Additional properties
}

// NewHierarchicalCommand creates a new hierarchical command
func NewHierarchicalCommand() *HierarchicalCommand {
	return &HierarchicalCommand{
		Children:   make([]*HierarchicalCommand, 0),
		Properties: make(map[string]string),
	}
}

// AddChild adds a child command to this command
func (hc *HierarchicalCommand) AddChild(child *HierarchicalCommand) {
	hc.Children = append(hc.Children, child)
}

// HasChildren checks if the command has nested children
func (hc *HierarchicalCommand) HasChildren() bool {
	return len(hc.Children) > 0
}

// IsValid checks if the command has the minimum required elements
func (hc *HierarchicalCommand) IsValid() bool {
	return hc.Action != "" && hc.ObjectType != ""
}

// String returns a human-readable representation of the command tree
func (hc *HierarchicalCommand) String() string {
	return hc.stringWithIndent(0)
}

func (hc *HierarchicalCommand) stringWithIndent(indent int) string {
	prefix := ""
	for i := 0; i < indent; i++ {
		prefix += "  "
	}

	result := prefix + string(hc.Action) + " " + string(hc.ObjectType)
	if hc.Name != "" {
		result += " " + hc.Name
	}
	if hc.Template != "" {
		result += " (template: " + hc.Template + ")"
	}

	if hc.HasChildren() {
		for _, child := range hc.Children {
			result += "\n" + child.stringWithIndent(indent+1)
		}
	}

	return result
}

// ToStructuredCommand converts to legacy StructuredCommand (for backward compatibility)
// Only preserves the first child
func (hc *HierarchicalCommand) ToStructuredCommand() *StructuredCommand {
	cmd := NewStructuredCommand()
	cmd.Action = hc.Action
	cmd.ObjectType = hc.ObjectType
	cmd.Name = hc.Name
	cmd.Path = hc.Path
	cmd.Properties = hc.Properties

	if len(hc.Children) > 0 {
		firstChild := hc.Children[0]
		cmd.ArgumentType = firstChild.ObjectType
		cmd.ArgumentName = firstChild.Name
		// Infer keyword based on context
		if hc.ObjectType == ObjectFolder && firstChild.ObjectType == ObjectFile {
			cmd.Keyword = KeywordWith
		}
	}

	return cmd
}

// FromStructuredCommand converts from legacy StructuredCommand
func FromStructuredCommand(sc *StructuredCommand) *HierarchicalCommand {
	hc := NewHierarchicalCommand()
	hc.Action = sc.Action
	hc.ObjectType = sc.ObjectType
	hc.Name = sc.Name
	hc.Path = sc.Path
	hc.Properties = sc.Properties

	// Convert single argument to child
	if sc.HasSecondaryOperation() {
		child := NewHierarchicalCommand()
		child.Action = sc.Action // Inherit action
		child.ObjectType = sc.ArgumentType
		child.Name = sc.ArgumentName
		hc.AddChild(child)
	}

	return hc
}
