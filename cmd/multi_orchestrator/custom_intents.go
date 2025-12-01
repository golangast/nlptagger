package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// CustomIntent represents a user-defined intent with its patterns and code template
type CustomIntent struct {
	Name        string   `json:"name"`
	Description string   `json:"description"`
	Keywords    []string `json:"keywords"`
	Patterns    []string `json:"patterns"`
	Template    string   `json:"template"`
	FilePath    string   `json:"file_path"`
}

// CustomIntentRegistry manages custom user-defined intents
type CustomIntentRegistry struct {
	Intents    map[string]*CustomIntent
	ConfigPath string
}

// NewCustomIntentRegistry creates a new registry
func NewCustomIntentRegistry(configPath string) *CustomIntentRegistry {
	registry := &CustomIntentRegistry{
		Intents:    make(map[string]*CustomIntent),
		ConfigPath: configPath,
	}
	registry.Load()
	return registry
}

// Load loads custom intents from the config file
func (r *CustomIntentRegistry) Load() error {
	if _, err := os.Stat(r.ConfigPath); os.IsNotExist(err) {
		// Create empty config if it doesn't exist
		return r.Save()
	}

	data, err := os.ReadFile(r.ConfigPath)
	if err != nil {
		return err
	}

	var intents []*CustomIntent
	if err := json.Unmarshal(data, &intents); err != nil {
		return err
	}

	for _, intent := range intents {
		r.Intents[intent.Name] = intent
	}

	return nil
}

// Save saves custom intents to the config file
func (r *CustomIntentRegistry) Save() error {
	// Ensure directory exists
	dir := filepath.Dir(r.ConfigPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	intents := make([]*CustomIntent, 0, len(r.Intents))
	for _, intent := range r.Intents {
		intents = append(intents, intent)
	}

	data, err := json.MarshalIndent(intents, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(r.ConfigPath, data, 0644)
}

// Add adds a new custom intent
func (r *CustomIntentRegistry) Add(intent *CustomIntent) error {
	r.Intents[intent.Name] = intent
	return r.Save()
}

// Remove removes a custom intent
func (r *CustomIntentRegistry) Remove(name string) error {
	delete(r.Intents, name)
	return r.Save()
}

// Match checks if a query matches any custom intent
func (r *CustomIntentRegistry) Match(query string) *CustomIntent {
	queryLower := strings.ToLower(query)

	for _, intent := range r.Intents {
		// Check if all keywords are present
		keywordMatch := true
		for _, keyword := range intent.Keywords {
			if !strings.Contains(queryLower, strings.ToLower(keyword)) {
				keywordMatch = false
				break
			}
		}

		if keywordMatch {
			return intent
		}

		// Check if any pattern matches
		for _, pattern := range intent.Patterns {
			if strings.Contains(queryLower, strings.ToLower(pattern)) {
				return intent
			}
		}
	}

	return nil
}

// List returns all custom intents
func (r *CustomIntentRegistry) List() []*CustomIntent {
	intents := make([]*CustomIntent, 0, len(r.Intents))
	for _, intent := range r.Intents {
		intents = append(intents, intent)
	}
	return intents
}

// InteractiveCreate guides the user through creating a custom intent
func (r *CustomIntentRegistry) InteractiveCreate(reader interface{ ReadString(byte) (string, error) }) error {
	fmt.Println("\n🎨 Create Custom Intent")
	fmt.Println("========================")

	// Get intent name
	fmt.Print("Intent name (e.g., 'create_jwt_middleware'): ")
	name, _ := reader.ReadString('\n')
	name = strings.TrimSpace(name)

	if name == "" {
		return fmt.Errorf("intent name cannot be empty")
	}

	// Get description
	fmt.Print("Description: ")
	description, _ := reader.ReadString('\n')
	description = strings.TrimSpace(description)

	// Get keywords
	fmt.Print("Keywords (comma-separated, e.g., 'jwt,auth,middleware'): ")
	keywordsStr, _ := reader.ReadString('\n')
	keywordsStr = strings.TrimSpace(keywordsStr)
	keywords := strings.Split(keywordsStr, ",")
	for i := range keywords {
		keywords[i] = strings.TrimSpace(keywords[i])
	}

	// Get patterns
	fmt.Print("Patterns (comma-separated, e.g., 'add jwt,create jwt middleware'): ")
	patternsStr, _ := reader.ReadString('\n')
	patternsStr = strings.TrimSpace(patternsStr)
	patterns := strings.Split(patternsStr, ",")
	for i := range patterns {
		patterns[i] = strings.TrimSpace(patterns[i])
	}

	// Get file path
	fmt.Print("Output file path (e.g., 'middleware/jwt.go'): ")
	filePath, _ := reader.ReadString('\n')
	filePath = strings.TrimSpace(filePath)

	// Get template - offer choice between inline or file
	fmt.Println("\nHow would you like to provide the code template?")
	fmt.Println("  1. Type code inline")
	fmt.Println("  2. Use a template file from 'learning' directory")
	fmt.Print("Choice (1 or 2): ")
	choice, _ := reader.ReadString('\n')
	choice = strings.TrimSpace(choice)

	var template string

	if choice == "2" {
		// List available template files
		learningDir := "learning"
		files, err := os.ReadDir(learningDir)
		if err != nil || len(files) == 0 {
			fmt.Println("⚠️  No template files found in 'learning' directory.")
			fmt.Println("Falling back to inline code entry...")
			choice = "1"
		} else {
			fmt.Println("\n📁 Available template files:")
			for i, file := range files {
				if !file.IsDir() {
					fmt.Printf("  %d. %s\n", i+1, file.Name())
				}
			}
			fmt.Print("\nEnter file number or name: ")
			fileChoice, _ := reader.ReadString('\n')
			fileChoice = strings.TrimSpace(fileChoice)

			var selectedFile string
			// Try to parse as number
			if fileNum, err := fmt.Sscanf(fileChoice, "%d", new(int)); err == nil && fileNum == 1 {
				var idx int
				fmt.Sscanf(fileChoice, "%d", &idx)
				if idx > 0 && idx <= len(files) {
					selectedFile = files[idx-1].Name()
				}
			} else {
				// Use as filename
				selectedFile = fileChoice
			}

			if selectedFile != "" {
				templatePath := filepath.Join(learningDir, selectedFile)
				content, err := os.ReadFile(templatePath)
				if err != nil {
					fmt.Printf("❌ Failed to read template file: %v\n", err)
					fmt.Println("Falling back to inline code entry...")
					choice = "1"
				} else {
					template = string(content)
					fmt.Printf("✅ Loaded template from %s\n", selectedFile)
				}
			} else {
				fmt.Println("Invalid selection. Falling back to inline code entry...")
				choice = "1"
			}
		}
	}

	if choice == "1" {
		// Get template inline
		fmt.Println("\nEnter code template (type 'END' on a new line to finish):")
		fmt.Println("You can use {{.VariableName}} for placeholders")
		var templateLines []string
		for {
			line, _ := reader.ReadString('\n')
			line = strings.TrimRight(line, "\n")
			if line == "END" {
				break
			}
			templateLines = append(templateLines, line)
		}
		template = strings.Join(templateLines, "\n")
	}

	intent := &CustomIntent{
		Name:        name,
		Description: description,
		Keywords:    keywords,
		Patterns:    patterns,
		Template:    template,
		FilePath:    filePath,
	}

	if err := r.Add(intent); err != nil {
		return fmt.Errorf("failed to save custom intent: %w", err)
	}

	fmt.Printf("\n✅ Custom intent '%s' created successfully!\n", name)
	return nil
}

// Execute executes a custom intent by generating code from its template
func (r *CustomIntentRegistry) Execute(intent *CustomIntent, projectRoot string, variables map[string]string) error {
	// Simple template replacement
	code := intent.Template
	for key, value := range variables {
		placeholder := "{{." + key + "}}"
		code = strings.ReplaceAll(code, placeholder, value)
	}

	// Determine output path
	outputPath := filepath.Join(projectRoot, intent.FilePath)

	// Ensure directory exists
	if err := os.MkdirAll(filepath.Dir(outputPath), 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	// Write file
	if err := os.WriteFile(outputPath, []byte(code), 0644); err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}

	fmt.Printf("✅ Generated %s\n", outputPath)
	return nil
}
