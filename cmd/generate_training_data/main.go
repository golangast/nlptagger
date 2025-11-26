package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
)

// SemanticOutput represents the structured output for a command
type SemanticOutput struct {
	Operation      string         `json:"operation"`
	TargetResource TargetResource `json:"target_resource"`
	Context        Context        `json:"context"`
}

// TargetResource represents the resource being operated on
type TargetResource struct {
	Type       string                 `json:"type"`
	Name       string                 `json:"name"`
	Properties map[string]interface{} `json:"properties"`
	Children   []ChildResource        `json:"children,omitempty"`
}

// ChildResource represents a child resource
type ChildResource struct {
	Type       string                 `json:"type"`
	Name       string                 `json:"name"`
	Properties map[string]interface{} `json:"properties"`
}

// Context represents the execution context
type Context struct {
	UserRole string `json:"user_role"`
}

// TrainingExample represents a single training example
type TrainingExample struct {
	Query          string         `json:"query"`
	SemanticOutput SemanticOutput `json:"semantic_output"`
}

func main() {
	examples := []TrainingExample{}

	// Generate examples
	examples = append(examples, generateCreateFolderExamples()...)
	examples = append(examples, generateCreateFileExamples()...)
	examples = append(examples, generateDeleteExamples()...)
	examples = append(examples, generateReadExamples()...)
	examples = append(examples, generateUpdateExamples()...)
	examples = append(examples, generateNestedExamples()...)

	log.Printf("Generated %d training examples", len(examples))

	// Write to file
	outputPath := "trainingdata/semantic_output_data.json"
	file, err := os.Create(outputPath)
	if err != nil {
		log.Fatalf("Failed to create output file: %v", err)
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(examples); err != nil {
		log.Fatalf("Failed to encode JSON: %v", err)
	}

	log.Printf("Successfully wrote %d examples to %s", len(examples), outputPath)
}

func generateCreateFolderExamples() []TrainingExample {
	examples := []TrainingExample{}

	folderNames := []string{"test", "data", "src", "bin", "config", "assets", "utils", "models", "lib", "docs",
		"tmp", "cache", "logs", "output", "input", "backup", "archive", "scripts", "tools", "resources",
		"build", "dist", "public", "private", "static", "media", "uploads", "downloads", "temp", "work",
		"projects", "app", "server", "client", "api", "web", "mobile", "desktop", "shared", "common",
		"core", "base", "vendor", "node_modules", "pkg", "internal", "external", "third_party", "plugins"}
	paths := []string{"./", "./src", "./data", "./config", "/home/user", "./projects", "./app", "./server",
		"./client", "./api", "./web", "./backend", "./frontend", "./services", "./components"}

	queryTemplates := []string{
		"create folder %s",
		"create a folder named %s",
		"make a new folder %s",
		"create directory %s",
		"make directory %s",
		"add folder %s",
		"new folder %s",
		"create a new directory called %s",
		"make a folder called %s",
		"add a new folder %s",
	}

	for _, name := range folderNames {
		for i, template := range queryTemplates {
			if i >= 5 { // More variations per name
				break
			}
			path := paths[i%len(paths)]
			examples = append(examples, TrainingExample{
				Query: fmt.Sprintf(template, name),
				SemanticOutput: SemanticOutput{
					Operation: "Create",
					TargetResource: TargetResource{
						Type: "Filesystem::Folder",
						Name: name,
						Properties: map[string]interface{}{
							"path": path,
						},
					},
					Context: Context{UserRole: "admin"},
				},
			})
		}
	}

	// Add examples with path specifications
	for i, name := range folderNames[:30] {
		path := paths[i%len(paths)]
		examples = append(examples, TrainingExample{
			Query: fmt.Sprintf("create folder %s in %s", name, path),
			SemanticOutput: SemanticOutput{
				Operation: "Create",
				TargetResource: TargetResource{
					Type: "Filesystem::Folder",
					Name: name,
					Properties: map[string]interface{}{
						"path": path,
					},
				},
				Context: Context{UserRole: "admin"},
			},
		})
	}

	return examples
}

func generateCreateFileExamples() []TrainingExample {
	examples := []TrainingExample{}

	fileNames := []string{"main.go", "config.json", "README.md", "test.txt", "data.csv", "app.py", "index.html",
		"style.css", "script.js", "package.json", "Dockerfile", "Makefile", "setup.py", "requirements.txt",
		"server.go", "client.go", "utils.go", "models.go", "handler.go", "middleware.go",
		"router.go", "controller.go", "service.go", "repository.go", "database.go", "auth.go",
		"api.go", "types.go", "errors.go", "constants.go", "config.yaml", "docker-compose.yml",
		"LICENSE", ".gitignore", ".env", "tsconfig.json", "webpack.config.js", "babel.config.js"}
	paths := []string{"./", "./src", "./config", "./docs", "./scripts", "./tests", "./app", "./server", "./client"}

	queryTemplates := []string{
		"create file %s",
		"create a file named %s",
		"make a new file %s",
		"add file %s",
		"new file %s",
		"create a file called %s",
		"make file %s",
	}

	for _, name := range fileNames {
		for i, template := range queryTemplates {
			if i >= 4 { // More variations
				break
			}
			path := paths[i%len(paths)]
			examples = append(examples, TrainingExample{
				Query: fmt.Sprintf(template, name),
				SemanticOutput: SemanticOutput{
					Operation: "Create",
					TargetResource: TargetResource{
						Type: "Filesystem::File",
						Name: name,
						Properties: map[string]interface{}{
							"path": path,
						},
					},
					Context: Context{UserRole: "admin"},
				},
			})
		}
	}

	return examples
}

func generateDeleteExamples() []TrainingExample {
	examples := []TrainingExample{}

	resources := []struct {
		name     string
		resType  string
		template string
	}{
		{"test.txt", "Filesystem::File", "delete file %s"},
		{"config.json", "Filesystem::File", "delete the file %s"},
		{"old_data", "Filesystem::Folder", "delete folder %s"},
		{"temp", "Filesystem::Folder", "remove folder %s"},
		{"cache", "Filesystem::Folder", "remove directory %s"},
		{"logs.txt", "Filesystem::File", "remove file %s"},
		{"backup.tar", "Filesystem::File", "delete %s"},
		{"archive", "Filesystem::Folder", "delete directory %s"},
	}

	for _, res := range resources {
		examples = append(examples, TrainingExample{
			Query: fmt.Sprintf(res.template, res.name),
			SemanticOutput: SemanticOutput{
				Operation: "Delete",
				TargetResource: TargetResource{
					Type: res.resType,
					Name: res.name,
					Properties: map[string]interface{}{
						"path": "./",
					},
				},
				Context: Context{UserRole: "admin"},
			},
		})
	}

	return examples
}

func generateReadExamples() []TrainingExample {
	examples := []TrainingExample{}

	queries := []struct {
		query string
		name  string
		path  string
	}{
		{"list all files", ".", "./"},
		{"list files in current directory", ".", "./"},
		{"show files", ".", "./"},
		{"list all files in src", "src", "./src"},
		{"show contents of data folder", "data", "./data"},
		{"list directory contents", ".", "./"},
	}

	for _, q := range queries {
		examples = append(examples, TrainingExample{
			Query: q.query,
			SemanticOutput: SemanticOutput{
				Operation: "Read",
				TargetResource: TargetResource{
					Type: "Filesystem::Folder",
					Name: q.name,
					Properties: map[string]interface{}{
						"path": q.path,
					},
				},
				Context: Context{UserRole: "admin"},
			},
		})
	}

	return examples
}

func generateUpdateExamples() []TrainingExample {
	examples := []TrainingExample{}

	updates := []struct {
		query   string
		name    string
		resType string
	}{
		{"update file config.json", "config.json", "Filesystem::File"},
		{"modify test.txt", "test.txt", "Filesystem::File"},
		{"edit main.go", "main.go", "Filesystem::File"},
		{"update README.md", "README.md", "Filesystem::File"},
		{"change config.json", "config.json", "Filesystem::File"},
	}

	for _, u := range updates {
		examples = append(examples, TrainingExample{
			Query: u.query,
			SemanticOutput: SemanticOutput{
				Operation: "Update",
				TargetResource: TargetResource{
					Type: u.resType,
					Name: u.name,
					Properties: map[string]interface{}{
						"path": "./",
					},
				},
				Context: Context{UserRole: "admin"},
			},
		})
	}

	return examples
}

func generateNestedExamples() []TrainingExample {
	examples := []TrainingExample{}

	// Folder with file children
	examples = append(examples, TrainingExample{
		Query: "create folder project with file main.go",
		SemanticOutput: SemanticOutput{
			Operation: "Create",
			TargetResource: TargetResource{
				Type: "Filesystem::Folder",
				Name: "project",
				Properties: map[string]interface{}{
					"path": "./",
				},
				Children: []ChildResource{
					{
						Type: "Filesystem::File",
						Name: "main.go",
						Properties: map[string]interface{}{
							"path": "./project",
						},
					},
				},
			},
			Context: Context{UserRole: "admin"},
		},
	})

	// Folder with webserver
	examples = append(examples, TrainingExample{
		Query: "in a new folder myapp, create a go webserver api",
		SemanticOutput: SemanticOutput{
			Operation: "Create",
			TargetResource: TargetResource{
				Type: "Filesystem::Folder",
				Name: "myapp",
				Properties: map[string]interface{}{
					"path": "./",
				},
				Children: []ChildResource{
					{
						Type: "Deployment::GoWebserver",
						Name: "api",
						Properties: map[string]interface{}{
							"port":    8080,
							"runtime": "go",
							"source":  "boilerplate_v1",
						},
					},
				},
			},
			Context: Context{UserRole: "admin"},
		},
	})

	return examples
}
