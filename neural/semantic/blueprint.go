package semantic

import (
	"bytes"
	"fmt"
	"strings"
	"text/template"
)

// Blueprint represents a parameterized code template
type Blueprint struct {
	Name         string           // Blueprint name (e.g., "webserver", "handler")
	Description  string           // Human-readable description
	TargetRole   SemanticRole     // What role this blueprint creates
	Parameters   []BlueprintParam // Template parameters
	CodeTemplate string           // Template with {{.VarName}} placeholders
	Dependencies []string         // Required imports/dependencies
	Files        []BlueprintFile  // Additional files this blueprint creates
}

// BlueprintParam represents a template parameter
type BlueprintParam struct {
	Name         string      // Parameter name
	Type         string      // "string", "int", "bool"
	DefaultValue interface{} // Default value if not provided
	Description  string      // Parameter description
}

// BlueprintFile represents an additional file created by a blueprint
type BlueprintFile struct {
	Name     string // Filename
	Role     string // Semantic role
	Template string // Code template
}

// BlueprintEngine executes blueprints with parameters
type BlueprintEngine struct {
	registry map[string]*Blueprint
}

// NewBlueprintEngine creates a new blueprint engine
func NewBlueprintEngine() *BlueprintEngine {
	engine := &BlueprintEngine{
		registry: make(map[string]*Blueprint),
	}

	engine.registerBuiltInBlueprints()
	return engine
}

// registerBuiltInBlueprints registers default blueprints
func (be *BlueprintEngine) registerBuiltInBlueprints() {
	// Webserver blueprint
	be.RegisterBlueprint(&Blueprint{
		Name:        "webserver",
		Description: "HTTP web server",
		TargetRole:  RoleEntrypoint,
		Parameters: []BlueprintParam{
			{Name: "ServerName", Type: "string", DefaultValue: "MyServer", Description: "Server name"},
			{Name: "Port", Type: "int", DefaultValue: 8080, Description: "Server port"},
		},
		CodeTemplate: `package main

import (
	"fmt"
	"log"
	"net/http"
)

func main() {
	http.HandleFunc("/", Handler)
	
	fmt.Println("{{.ServerName}} starting on :{{.Port}}")
	log.Fatal(http.ListenAndServe(":{{.Port}}", nil))
}
`,
		Files: []BlueprintFile{
			{
				Name: "handler.go",
				Role: string(RoleHandler),
				Template: `package main

import (
	"fmt"
	"net/http"
)

// Handler handles HTTP requests
func Handler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "Hello from {{.ServerName}}!")
}
`,
			},
		},
	})

	// API blueprint
	be.RegisterBlueprint(&Blueprint{
		Name:        "api",
		Description: "REST API server",
		TargetRole:  RoleEntrypoint,
		Parameters: []BlueprintParam{
			{Name: "APIName", Type: "string", DefaultValue: "MyAPI", Description: "API name"},
			{Name: "Version", Type: "string", DefaultValue: "1.0", Description: "API version"},
		},
		CodeTemplate: `package main

import (
	"encoding/json"
	"log"
	"net/http"
)

func main() {
	http.HandleFunc("/api/health", healthHandler)
	http.HandleFunc("/api/info", infoHandler)
	
	log.Println("{{.APIName}} v{{.Version}} starting")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func healthHandler(w http.ResponseWriter, r *http.Request) {
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func infoHandler(w http.ResponseWriter, r *http.Request) {
	json.NewEncoder(w).Encode(map[string]string{
		"name": "{{.APIName}}",
		"version": "{{.Version}}",
	})
}
`,
	})

	// Handler blueprint
	be.RegisterBlueprint(&Blueprint{
		Name:        "handler",
		Description: "HTTP request handler",
		TargetRole:  RoleHandler,
		Parameters: []BlueprintParam{
			{Name: "HandlerName", Type: "string", DefaultValue: "Handler", Description: "Handler function name"},
		},
		CodeTemplate: `package main

import (
	"fmt"
	"net/http"
)

// {{.HandlerName}} handles HTTP requests
func {{.HandlerName}}(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "{{.HandlerName}} response")
}
`,
	})
}

// RegisterBlueprint adds a blueprint to the registry
func (be *BlueprintEngine) RegisterBlueprint(bp *Blueprint) {
	be.registry[bp.Name] = bp
}

// GetBlueprint retrieves a blueprint by name
func (be *BlueprintEngine) GetBlueprint(name string) (*Blueprint, bool) {
	bp, exists := be.registry[name]
	return bp, exists
}

// Execute renders a blueprint with parameters
func (be *BlueprintEngine) Execute(blueprintName string, params map[string]interface{}) (string, error) {
	bp, exists := be.GetBlueprint(blueprintName)
	if !exists {
		return "", fmt.Errorf("blueprint '%s' not found", blueprintName)
	}

	// Merge with defaults
	finalParams := be.mergeWithDefaults(bp, params)

	// Execute template
	tmpl, err := template.New(bp.Name).Parse(bp.CodeTemplate)
	if err != nil {
		return "", fmt.Errorf("template parse error: %v", err)
	}

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, finalParams); err != nil {
		return "", fmt.Errorf("template execution error: %v", err)
	}

	return buf.String(), nil
}

// ExecuteFile renders a blueprint file with parameters
func (be *BlueprintEngine) ExecuteFile(bp *Blueprint, file BlueprintFile, params map[string]interface{}) (string, error) {
	// Merge with defaults
	finalParams := be.mergeWithDefaults(bp, params)

	// Execute template
	tmpl, err := template.New(file.Name).Parse(file.Template)
	if err != nil {
		return "", fmt.Errorf("template parse error: %v", err)
	}

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, finalParams); err != nil {
		return "", fmt.Errorf("template execution error: %v", err)
	}

	return buf.String(), nil
}

// mergeWithDefaults merges provided params with blueprint defaults
func (be *BlueprintEngine) mergeWithDefaults(bp *Blueprint, params map[string]interface{}) map[string]interface{} {
	result := make(map[string]interface{})

	// Start with defaults
	for _, param := range bp.Parameters {
		result[param.Name] = param.DefaultValue
	}

	// Override with provided values
	for key, value := range params {
		result[key] = value
	}

	return result
}

// ExtractParameters parses parameters from a query string
// Example: "create server named MyAPI on port 3000" → {ServerName: "MyAPI", Port: 3000}
func (be *BlueprintEngine) ExtractParameters(query string, bp *Blueprint) map[string]interface{} {
	params := make(map[string]interface{})

	words := splitWords(query)

	for i, word := range words {
		lowerWord := toLower(word)

		// Pattern: "named <value>"
		if lowerWord == "named" && i+1 < len(words) {
			params["ServerName"] = words[i+1]
			params["APIName"] = words[i+1]
			params["HandlerName"] = words[i+1]
		}

		// Pattern: "port <number>"
		if (lowerWord == "port" || lowerWord == "on") && i+1 < len(words) {
			if i+2 < len(words) && toLower(words[i+1]) == "port" {
				// "on port 3000"
				params["Port"] = parseInt(words[i+2])
			} else {
				// "port 3000"
				params["Port"] = parseInt(words[i+1])
			}
		}

		// Pattern: "version <value>"
		if lowerWord == "version" && i+1 < len(words) {
			params["Version"] = words[i+1]
		}
	}

	return params
}

// Helper functions
func splitWords(s string) []string {
	return strings.Fields(s)
}

func toLower(s string) string {
	return strings.ToLower(s)
}

func parseInt(s string) int {
	var i int
	fmt.Sscanf(s, "%d", &i)
	return i
}
