# Multi-Orchestrator with NLP Understanding

## Overview

The Multi-Orchestrator has been enhanced with **semantic understanding** using the MoE (Mixture of Experts) model and NER (Named Entity Recognition) tagging. Instead of relying on simple keyword matching, it now understands what you're writing in natural language.

## What's New?

### Before (Keyword-based)
The orchestrator would only recognize specific keywords like:
- "create webserver"
- "create handler"
- "create database"

### After (Semantic Understanding)
Now you can write naturally, and the orchestrator will understand your intent:
- "I want to build a webserver with an authentication handler"
- "Make me a Go server that has JWT support"
- "Create an API with a database for storing user data"

## How It Works

The enhanced orchestrator uses a multi-stage NLP pipeline:

1. **Intent Classification** - Determines what you want to do (create, modify, delete, etc.)
2. **Named Entity Recognition (NER)** - Extracts key entities like handler names, database names, component names
3. **Semantic Parsing** - Builds a structured representation of your command
4. **Template Filling** - Generates the appropriate code based on the semantic understanding

## Example Usage

```bash
# Run the orchestrator
./cmd/multi_orchestrator/multi_orchestrator

# Try natural language commands:
> create a webserver with authentication handler
> build me a Go server with a users database
> I need an API with JWT middleware
```

## Semantic Analysis Output

When you enter a command, you'll see:

```
🧠 Semantic Analysis:
  Intent: create_handler
  Entities: {handler_name: "authentication", component_name: "webserver"}
  Pattern: CREATE webserver WITH authentication handler
  Semantic Output:
  {
    "action": "create",
    "object_type": "handler",
    "name": "authentication",
    ...
  }
```

## Features

- ✅ **Natural Language Understanding** - Write commands as you would speak them
- ✅ **Intent Detection** - Automatically determines what you want to do
- ✅ **Entity Extraction** - Intelligently extracts names, types, and parameters
- ✅ **Fallback Support** - Falls back to keyword matching if semantic parsing fails
- ✅ **Semantic Output** - Generates structured JSON representation of commands
- ✅ **Git Integration** - Tracks changes with automatic commits
- ✅ **History & Revert** - View command history and revert to previous states

## Architecture

```
User Input (Natural Language)
    ↓
Intent Classifier (MoE-based)
    ↓
Named Entity Recognition (Rule-based + Neural)
    ↓
Semantic Parser (Template-based or Hierarchical)
    ↓
Template Filler
    ↓
Code Generation
    ↓
Git Commit
```

## Technical Details

### Components Used

- **Intent Classifier** (`neural/semantic/intent_classifier.go`) - Classifies user intent
- **Entity Extractor** (`neural/semantic/entity_extractor.go`) - Extracts entities from queries
- **NER Tagger** (`neural/nn/ner/`) - Rule-based named entity recognition
- **Semantic Parser** (`neural/semantic/command_parser.go`) - Parses commands into structured format
- **Template Registry** (`neural/semantic/template_registry.go`) - Manages code templates

### Supported Intents

- `create_file` - Create a new file
- `create_folder` - Create a new folder
- `create_handler` - Create a handler/endpoint
- `create_database` - Create a database integration
- `add_feature` - Add a feature to a component
- `modify_code` - Modify existing code
- `delete_file` - Delete a file
- `delete_folder` - Delete a folder
- `rename_file` - Rename a file
- `rename_folder` - Rename a folder
- `move_file` - Move a file
- `move_folder` - Move a folder

## Future Enhancements

- [ ] Integration with actual MoE neural model for inference
- [ ] Support for more complex multi-step commands
- [ ] Context-aware suggestions
- [ ] Learning from user corrections
- [ ] Support for custom templates

## Comparison: Before vs After

### Before (Keyword Matching)
```go
if strings.Contains(goal, "handler") {
    // Extract handler name with string manipulation
    handlerKeywordIndex := strings.LastIndex(goal, "handler")
    // ... complex string parsing logic
}
```

### After (Semantic Understanding)
```go
parsedGoal, err := parseGoalWithSemantics(goal)
if handlerName, ok := parsedGoal.Entities["handler_name"]; ok {
    customHandlerName = handlerName
    fmt.Printf("✅ Extracted handler name: %s\n", customHandlerName)
}
```

The semantic approach is:
- **More robust** - Handles variations in phrasing
- **More accurate** - Uses NER to identify entities correctly
- **More extensible** - Easy to add new intents and entities
- **More user-friendly** - Users can write naturally

## Try It Out!

```bash
# Build
go build -o cmd/multi_orchestrator/multi_orchestrator ./cmd/multi_orchestrator

# Run
./cmd/multi_orchestrator/multi_orchestrator

# Try these natural language commands:
> create a webserver with authentication handler
> I want to build an API with a database called users.db
> make me a Go server with JWT support
```
