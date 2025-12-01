# NLP Tagger - Multi-Orchestrator

An intelligent code generation and project management tool powered by natural language processing. The Multi-Orchestrator understands your intent from natural language commands and generates complete project structures, handlers, databases, and more.

## 🚀 Features

### 🧠 Natural Language Understanding
- **Intent Classification**: Automatically classifies user commands using MoE (Mixture of Experts) based semantic analysis
- **Named Entity Recognition (NER)**: Extracts entities like handler names, database names, file paths from natural language
- **Semantic Parsing**: Converts natural language into structured semantic output for precise code generation
- **Interactive Clarification**: When intent is unclear, the system asks for clarification with helpful options

### 🎨 Custom Intent Creation
Create your own custom intents with reusable code templates:

```bash
# Create a new custom intent
> create intent
# or
> new intent

# Follow the interactive prompts:
# 1. Intent name (e.g., 'create_jwt_middleware')
# 2. Description
# 3. Keywords (comma-separated)
# 4. Patterns (comma-separated phrases that trigger this intent)
# 5. Output file path
# 6. Code template (inline or from learning directory)
```

**Template Variables**: Use `{{.VariableName}}` placeholders in your templates for dynamic code generation.

**Example Custom Intent**:
- **Name**: `create_jwt_middleware`
- **Keywords**: `jwt`, `auth`, `middleware`
- **Patterns**: `add jwt`, `create jwt middleware`
- **Template**: Uses Go template syntax with variables like `{{.SecretKey}}`, `{{.TokenExpiry}}`

### 📚 Custom Intent Management
```bash
# List all custom intents
> list intents
> show intents

# Remove a custom intent
> remove intent <name>
> delete intent <name>
```

### ⏪ Git-Based Revert System
Every command is automatically committed to Git, enabling powerful revert capabilities:

```bash
# Show command history with commit hashes
> show history

# Revert by command ID
> revert 5

# Revert by commit hash
> revert a1b2c3d

# Revert by command text (finds most recent match)
> revert create authentication handler
```

**How it works**:
- Each successful command creates a Git commit
- Command history is stored in SQLite database with commit hashes
- Revert uses `git reset --hard` to restore project state
- Git history is preserved even when deleting projects

### 🏗️ Intelligent Code Generation
- **Go Web Servers**: Generate complete HTTP servers with routing
- **Custom Handlers**: Create specific handlers (authentication, user management, etc.)
- **Database Integration**: Generate servers with SQLite database support
- **Dockerfile**: Automatic Docker containerization
- **README**: Auto-generated project documentation

### 📖 Knowledge Base Learning
The orchestrator learns from your code:
- Store code templates in the `learning/` directory
- When generating files, the system searches for relevant learned content
- Interactive selection of learned templates during generation
- Reuse patterns across projects

### 🔄 Multi-Agent Architecture
Parallel task execution with specialized agents:
- **Coder Agent**: Generates Go server code and handlers
- **DevOps Agent**: Creates Dockerfile and deployment configs
- **Documentation Agent**: Writes README files
- **QA Agent**: Runs tests and builds to validate generated code

### 🎯 Semantic File Generation
Create complex project structures from natural language:
```bash
> create a webserver with authentication handler
> add a database with users table
> create middleware folder with jwt auth
```

The system understands:
- File and folder hierarchies
- Component relationships
- Code dependencies
- Project structure conventions

### 🗄️ Persistent State Management
- **SQLite Database**: Stores all commands, generated code, and metadata
- **Git Repository**: Tracks all changes with full history
- **Session Continuity**: Resume work across sessions
- **Command History**: Full audit trail of all operations

## 🛠️ Usage

### Running the Multi-Orchestrator

```bash
go run ./cmd/multi_orchestrator
```

### Example Commands

```bash
# Create a basic web server
> create a webserver

# Create a server with specific handler
> create a webserver with authentication handler

# Create a server with database
> create a webserver with database users.db

# Delete project (preserves Git history)
> delete project

# View command history
> show history

# Revert to previous state
> revert 3
> revert create authentication

# Create custom intent
> create intent

# List custom intents
> list intents

# Exit
> exit
```

## 📁 Project Structure

```
generated_projects/
├── project/                    # Generated project files
│   ├── .git/                  # Git repository (preserved on delete)
│   ├── server.go              # Generated Go server
│   ├── handlers/              # Generated handlers
│   ├── database/              # SQLite databases
│   ├── Dockerfile             # Docker configuration
│   └── README.md              # Project documentation
├── orchestrator.db            # Command history and metadata
└── custom_intents.json        # User-defined custom intents

learning/                       # Knowledge base directory
└── *.go                       # Learned code templates

cmd/multi_orchestrator/
├── main.go                    # Main orchestrator logic
├── custom_intents.go          # Custom intent system
└── knowledge.go               # Knowledge base implementation
```

## 🧪 QA Phase

The orchestrator automatically validates generated code:
1. **Syntax Check**: Ensures Go code compiles
2. **Build Test**: Attempts to build the project
3. **Error Reporting**: Provides detailed error messages
4. **Auto-retry**: Prompts for retry on failure

## 🔧 Advanced Features

### Template Variables
Custom intents support Go template syntax:
- `{{.Name}}` - Simple variable substitution
- `{{.HandlerName}}` - Dynamic handler names
- `{{.DatabasePath}}` - File paths
- Any custom variables you define

### Typo Tolerance
Command matching is flexible:
- Case-insensitive matching
- Partial phrase matching
- Keyword-based intent detection

### Interactive Workflows
When the system is uncertain:
1. Shows detected entities
2. Presents likely intent options
3. Allows manual intent specification
4. Supports custom intent creation on-the-fly

## 🎓 Learning System

Place code templates in the `learning/` directory:
```bash
learning/
├── auth_handler.go
├── jwt_middleware.go
└── database_models.go
```

When generating similar files, the orchestrator:
1. Searches for relevant templates
2. Presents matches to the user
3. Allows selection of learned content
4. Applies templates to new files

## 🔐 Database Schema

The SQLite database stores:
- **messages**: Command history with timestamps
- **commit_hash**: Git commit for each command
- **role**: User commands vs. generated code
- **content**: Full command text or generated code

## 🚦 Getting Started

1. **Run the orchestrator**:
   ```bash
   go run ./cmd/multi_orchestrator
   ```

2. **Create your first project**:
   ```bash
   > create a webserver with user handler
   ```

3. **View what was created**:
   ```bash
   ls generated_projects/project/
   ```

4. **Create a custom intent**:
   ```bash
   > create intent
   ```

5. **Experiment with revert**:
   ```bash
   > show history
   > revert 1
   ```

## 🎯 Use Cases

- **Rapid Prototyping**: Generate project scaffolding in seconds
- **Learning Tool**: Understand project structure patterns
- **Template Management**: Reuse code patterns across projects
- **Experimentation**: Try different approaches with easy revert
- **Custom Workflows**: Define your own intents for repeated tasks

## 🔮 Future Enhancements

- [ ] CSV to handler code generation
- [ ] better learning and remembering intent

GPL 3

---

**Built with**: Go, SQLite, Git, Neural Networks (MoE), NLP
