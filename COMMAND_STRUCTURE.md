# Command Structure Documentation

## Overview

The NLP tagger now supports explicit **structured command parsing**, which organizes natural language queries into clearly defined semantic elements.

## Command Elements

Each parsed command consists of the following elements:

| Element | Role | Example |
|---------|------|---------|
| **Action** | The function or method to execute | `create`, `delete`, `move`, `rename` |
| **Object Type** | The primary data structure or entity to operate on | `folder`, `file`, `component` |
| **Name** | The name or identifier for the new entity | `jime`, `main.go`, `src` |
| **Keyword** | A separator to introduce a secondary operation | `with`, `and`, `in`, `to` |
| **Argument Type** | The secondary object to create/operate on | `file`, `folder` |
| **Argument Name** | The name/identifier for the secondary object | `Jill.go`, `destination` |

## Example

### Query
```
create folder jime with file Jill.go
```

### Parsed Structure
```
Action:        create
Object Type:   folder
Name:          jime
Keyword:       with
Argument Type: file
Argument Name: Jill.go
```

### Generated Semantic Output
```json
{
  "Operation": "create",
  "TargetResource": {
    "Type": "Filesystem::Folder",
    "Name": "jime",
    "Properties": {
      "path": "./"
    },
    "Children": [
      {
        "Type": "Filesystem::File",
        "Name": "Jill.go",
        "Properties": {}
      }
    ]
  },
  "Context": {
    "UserRole": "admin"
  }
}
```

## Supported Commands

### Creation Commands
- `create folder <name>`
- `create file <name>`
- `create folder <name> with file <name>`
- `add folder <name> and file <name>`

### Deletion Commands
- `delete file <name>`
- `delete folder <name>`
- `remove file <name>`

### Move Commands
- `move file <name> to <destination>`
- `move folder <name> into <destination>`

### Rename Commands
- `rename file <old> to <new>`
- `rename folder <old> to <new>`

### Code Modification Commands
- `add <feature> to <component>`
- `modify <component>`

## Architecture

### Core Components

1. **`command_structure.go`** - Defines the structured command types
   - `StructuredCommand` struct
   - `CommandAction`, `ObjectType`, `CommandKeyword` enums
   - Helper methods for validation

2. **`command_parser.go`** - Parses queries into structured commands
   - `CommandParser` - Main parser
   - `BuildCommand()` - Constructs structured commands from intents and entities
   - `detectKeyword()` - Identifies separating keywords

3. **`intent_templates.go`** - Generates semantic output
   - `FillFromStructuredCommand()` - Converts structured commands to JSON
   - Backward-compatible entity map approach still available

### Data Flow

```
Natural Language Query
        ↓
Intent Classifier → Intent Type
        ↓
NER System → Entity Map
        ↓
Command Parser → Structured Command
        ↓
Template Filler → Semantic Output (JSON)
```

## Using the Structured Parser

### Basic Usage

```go
import (
    "nlptagger/neural/semantic"
    "nlptagger/neural/nn/ner"
)

// Initialize components
parser := semantic.NewCommandParser()
nerSystem := ner.NewRuleBasedNER()

// Parse query
query := "create folder jime with file Jill.go"
words := strings.Fields(query)
entityMap := nerSystem.TagSequence(words)

// Get structured command
cmd := parser.Parse(query, words, entityMap)

// Generate semantic output
output := semantic.FillFromStructuredCommand(cmd)
```

### Accessing Command Elements

```go
// Check command validity
if cmd.IsValid() {
    fmt.Printf("Action: %s\n", cmd.Action)
    fmt.Printf("Operating on %s: %s\n", cmd.ObjectType, cmd.Name)
}

// Check for secondary operations
if cmd.HasSecondaryOperation() {
    fmt.Printf("With %s: %s\n", cmd.ArgumentType, cmd.ArgumentName)
}

// Get human-readable representation
fmt.Println(cmd.String())
// Output: "create folder jime with file Jill.go"
```

## Running the Demo

A demonstration program is available at `cmd/command_structure_demo/main.go`:

```bash
go run cmd/command_structure_demo/main.go
```

This will show:
- Structured command elements for "create folder jime with file Jill.go"
- Generated semantic JSON output
- Additional examples with different command types

## Extending the System

### Adding New Actions

1. Add action constant to `command_structure.go`:
```go
const (
    ActionCopy CommandAction = "copy"
    // ... other actions
)
```

2. Update `IntentToAction()` mapping
3. Create corresponding intent type in `intent_templates.go`

### Adding New Object Types

1. Add type constant to `command_structure.go`:
```go
const (
    ObjectModule ObjectType = "module"
    // ... other types
)
```

2. Update `objectTypeToResourceType()` mapping

### Adding New Keywords

1. Add keyword constant:
```go
const (
    KeywordContaining CommandKeyword = "containing"
    // ... other keywords
)
```

2. Update `detectKeyword()` in `command_parser.go`

## Benefits

1. **Clarity** - Each command element has a clear, documented role
2. **Maintainability** - Easy to understand and modify parsing logic
3. **Extensibility** - Simple to add new actions, types, and keywords
4. **Debugging** - Structured output makes it easy to see what was parsed
5. **Type Safety** - Strong typing prevents errors
6. **Backward Compatible** - Existing entity map approach still works
