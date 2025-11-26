package semantic

// CommandElement represents the type of element in a structured command
type CommandElement string

const (
	ElementAction       CommandElement = "action"        // The function/method to execute
	ElementObjectType   CommandElement = "object_type"   // Primary data structure/entity
	ElementName         CommandElement = "name"          // Name/identifier for the entity
	ElementKeyword      CommandElement = "keyword"       // Separator for secondary operations
	ElementArgumentType CommandElement = "argument_type" // Secondary object type
	ElementArgumentName CommandElement = "argument_name" // Name/identifier for secondary object
)

// CommandAction represents the action to perform
type CommandAction string

const (
	ActionCreate CommandAction = "create"
	ActionDelete CommandAction = "delete"
	ActionMove   CommandAction = "move"
	ActionRename CommandAction = "rename"
	ActionRead   CommandAction = "read"
	ActionModify CommandAction = "modify"
	ActionAdd    CommandAction = "add"
)

// ObjectType represents the type of object being operated on
type ObjectType string

const (
	ObjectFile      ObjectType = "file"
	ObjectFolder    ObjectType = "folder"
	ObjectComponent ObjectType = "component"
	ObjectFeature   ObjectType = "feature"
	ObjectCode      ObjectType = "code"
)

// CommandKeyword represents keywords that separate or relate operations
type CommandKeyword string

const (
	KeywordWith   CommandKeyword = "with"
	KeywordAnd    CommandKeyword = "and"
	KeywordIn     CommandKeyword = "in"
	KeywordInto   CommandKeyword = "into"
	KeywordTo     CommandKeyword = "to"
	KeywordInside CommandKeyword = "inside"
)

// StructuredCommand represents a parsed command with all its elements
type StructuredCommand struct {
	// Primary operation
	Action     CommandAction // e.g., "create"
	ObjectType ObjectType    // e.g., "folder"
	Name       string        // e.g., "jime"

	// Secondary/nested operation (optional)
	Keyword      CommandKeyword // e.g., "with"
	ArgumentType ObjectType     // e.g., "file"
	ArgumentName string         // e.g., "Jill.go"

	// Additional context
	Path       string            // Path/location for the operation
	Properties map[string]string // Additional properties
}

// NewStructuredCommand creates a new structured command
func NewStructuredCommand() *StructuredCommand {
	return &StructuredCommand{
		Properties: make(map[string]string),
	}
}

// HasSecondaryOperation checks if the command has a nested/secondary operation
func (sc *StructuredCommand) HasSecondaryOperation() bool {
	return sc.Keyword != "" && sc.ArgumentType != ""
}

// IsValid checks if the command has the minimum required elements
func (sc *StructuredCommand) IsValid() bool {
	return sc.Action != "" && sc.ObjectType != ""
}

// String returns a human-readable representation of the command
func (sc *StructuredCommand) String() string {
	result := string(sc.Action) + " " + string(sc.ObjectType)
	if sc.Name != "" {
		result += " " + sc.Name
	}
	if sc.HasSecondaryOperation() {
		result += " " + string(sc.Keyword) + " " + string(sc.ArgumentType)
		if sc.ArgumentName != "" {
			result += " " + sc.ArgumentName
		}
	}
	return result
}

// IntentToAction converts an IntentType to CommandAction
func IntentToAction(intent IntentType) CommandAction {
	switch intent {
	case IntentCreateFolder, IntentCreateFile, IntentCreateFolderWithFile:
		return ActionCreate
	case IntentDeleteFile, IntentDeleteFolder:
		return ActionDelete
	case IntentMoveFile, IntentMoveFolder:
		return ActionMove
	case IntentRenameFile, IntentRenameFolder:
		return ActionRename
	case IntentReadFile:
		return ActionRead
	case IntentModifyCode:
		return ActionModify
	case IntentAddFeature:
		return ActionAdd
	default:
		return ""
	}
}

// IntentToObjectType extracts the primary object type from an intent
func IntentToObjectType(intent IntentType) ObjectType {
	switch intent {
	case IntentCreateFile, IntentDeleteFile, IntentMoveFile, IntentRenameFile, IntentReadFile:
		return ObjectFile
	case IntentCreateFolder, IntentDeleteFolder, IntentMoveFolder, IntentRenameFolder, IntentCreateFolderWithFile:
		return ObjectFolder
	case IntentModifyCode:
		return ObjectCode
	case IntentAddFeature:
		return ObjectComponent
	default:
		return ""
	}
}

// IntentToSecondaryObjectType extracts the secondary object type from an intent
func IntentToSecondaryObjectType(intent IntentType) ObjectType {
	switch intent {
	case IntentCreateFolderWithFile:
		return ObjectFile
	default:
		return ""
	}
}
