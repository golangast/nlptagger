package semantic

import "fmt"

// ProjectTemplate defines a project scaffolding template
type ProjectTemplate struct {
	Name        string            // Template name (e.g., "webserver", "api")
	Description string            // Human-readable description
	Files       []TemplateFile    // Files to create
	Folders     []string          // Folders to create
	Properties  map[string]string // Additional properties
}

// TemplateFile defines a file to create with optional boilerplate
type TemplateFile struct {
	Name    string // Filename (e.g., "main.go")
	Content string // Boilerplate content
}

// TemplateRegistry manages project templates
type TemplateRegistry struct {
	templates map[string]*ProjectTemplate
}

// NewTemplateRegistry creates a new template registry with built-in templates
func NewTemplateRegistry() *TemplateRegistry {
	registry := &TemplateRegistry{
		templates: make(map[string]*ProjectTemplate),
	}

	// Register built-in templates
	registry.registerBuiltInTemplates()

	return registry
}

// registerBuiltInTemplates registers common project templates
func (tr *TemplateRegistry) registerBuiltInTemplates() {
	// Go Web Server template
	tr.RegisterTemplate(&ProjectTemplate{
		Name:        "webserver",
		Description: "Go HTTP web server with handler",
		Files: []TemplateFile{
			{
				Name: "main.go",
				Content: `package main

import (
	"fmt"
	"log"
	"net/http"
)

func main() {
	http.HandleFunc("/", Handler)
	
	fmt.Println("Server starting on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
`,
			},
			{
				Name: "handler.go",
				Content: `package main

import (
	"fmt"
	"net/http"
)

// Handler handles HTTP requests
func Handler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Hello, World!")
}
`,
			},
		},
		Folders: []string{"templates", "static"},
	})

	// REST API template
	tr.RegisterTemplate(&ProjectTemplate{
		Name:        "api",
		Description: "REST API server with routes",
		Files: []TemplateFile{
			{
				Name: "main.go",
				Content: `package main

import (
	"encoding/json"
	"log"
	"net/http"
)

func main() {
	http.HandleFunc("/api/health", healthHandler)
	http.HandleFunc("/api/data", dataHandler)
	
	log.Println("API server starting on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func dataHandler(w http.ResponseWriter, r *http.Request) {
	json.NewEncoder(w).Encode(map[string]interface{}{
		"message": "Data endpoint",
		"version": "1.0",
	})
}
`,
			},
		},
		Folders: []string{"models", "routes", "middleware"},
	})

	// CLI Tool template
	tr.RegisterTemplate(&ProjectTemplate{
		Name:        "cli",
		Description: "Command-line tool",
		Files: []TemplateFile{
			{
				Name: "main.go",
				Content: `package main

import (
	"flag"
	"fmt"
	"os"
)

func main() {
	var name string
	flag.StringVar(&name, "name", "World", "Name to greet")
	flag.Parse()
	
	fmt.Printf("Hello, %s!\n", name)
	os.Exit(0)
}
`,
			},
		},
		Folders: []string{"cmd", "internal"},
	})
}

// RegisterTemplate adds a template to the registry
func (tr *TemplateRegistry) RegisterTemplate(template *ProjectTemplate) {
	tr.templates[template.Name] = template
}

// GetTemplate retrieves a template by name
func (tr *TemplateRegistry) GetTemplate(name string) (*ProjectTemplate, error) {
	template, exists := tr.templates[name]
	if !exists {
		return nil, fmt.Errorf("template '%s' not found", name)
	}
	return template, nil
}

// HasTemplate checks if a template exists
func (tr *TemplateRegistry) HasTemplate(name string) bool {
	_, exists := tr.templates[name]
	return exists
}

// ListTemplates returns all available template names
func (tr *TemplateRegistry) ListTemplates() []string {
	names := make([]string, 0, len(tr.templates))
	for name := range tr.templates {
		names = append(names, name)
	}
	return names
}

// ApplyTemplate applies a template to a hierarchical command
func (tr *TemplateRegistry) ApplyTemplate(cmd *HierarchicalCommand, templateName string) error {
	template, err := tr.GetTemplate(templateName)
	if err != nil {
		return err
	}

	// Add files from template
	for _, file := range template.Files {
		fileCmd := NewHierarchicalCommand()
		fileCmd.Action = ActionCreate
		fileCmd.ObjectType = ObjectFile
		fileCmd.Name = file.Name
		fileCmd.Properties = map[string]string{
			"content": file.Content,
		}
		cmd.AddChild(fileCmd)
	}

	// Add folders from template
	for _, folder := range template.Folders {
		folderCmd := NewHierarchicalCommand()
		folderCmd.Action = ActionCreate
		folderCmd.ObjectType = ObjectFolder
		folderCmd.Name = folder
		cmd.AddChild(folderCmd)
	}

	// Store template name
	cmd.Template = templateName

	return nil
}
