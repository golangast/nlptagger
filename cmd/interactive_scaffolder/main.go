package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"

	"nlptagger/neural/nn/ner"
	"nlptagger/neural/semantic"
)

func main() {
	fmt.Println("=== Advanced NLP Project Scaffolder ===")
	fmt.Println("Intelligent project manager with memory and context")
	fmt.Println("Commands: 'tree', 'show <file>', 'deps', 'exit'\n")

	// Initialize advanced system
	vfs := semantic.NewVFSTree()
	blueprintEngine := semantic.NewBlueprintEngine()
	depGraph := semantic.NewDependencyGraph()
	roleRegistry := semantic.NewRoleRegistry()

	hierarchicalParser := semantic.NewHierarchicalParser()
	standardParser := semantic.NewCommandParser()
	templateRegistry := semantic.NewTemplateRegistry()

	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Print("> ")
		if !scanner.Scan() {
			break
		}

		query := strings.TrimSpace(scanner.Text())
		if query == "" {
			continue
		}

		if query == "exit" || query == "quit" {
			fmt.Println("Goodbye!")
			break
		}

		// Special commands
		if query == "tree" {
			fmt.Println(vfs.Tree())
			continue
		}

		if query == "deps" {
			fmt.Println(depGraph.Summary())
			continue
		}

		// Handle modification requests: "update X to Y", "change X to Y", "replace X with Y"
		if containsModifyKeyword(query) {
			handleModification(vfs, query)
			continue
		}

		// Handle "show <file>" or "what is in <file>"
		if strings.HasPrefix(query, "show ") || strings.Contains(query, "what is in") {
			filename := extractFilename(query)
			if filename != "" {
				showFile(vfs, filename)
				continue
			}
		}

		// Process create command
		processCommand(query, vfs, blueprintEngine, depGraph, roleRegistry,
			hierarchicalParser, standardParser, templateRegistry)
	}
}

func extractFilename(query string) string {
	words := strings.Fields(query)
	for i, word := range words {
		if strings.Contains(word, ".") {
			return word
		}
		if word == "in" && i+1 < len(words) {
			return words[i+1]
		}
	}
	return ""
}

func showFile(vfs *semantic.VFSTree, filename string) {
	node, found := vfs.ResolvePath(filename)
	if !found {
		fmt.Printf("File '%s' not found\n", filename)
		return
	}

	if node.Type != "file" {
		fmt.Printf("'%s' is a folder, not a file\n", filename)
		return
	}

	fmt.Printf("\n=== %s [%s] ===\n", node.Path, node.Role)
	if node.Content != "" {
		fmt.Println(node.Content)
	} else {
		fmt.Println("(empty file)")
	}
	fmt.Println()
}

func containsModifyKeyword(query string) bool {
	lower := strings.ToLower(query)
	keywords := []string{"update", "change", "replace", "modify", "rename in", "fix"}
	for _, keyword := range keywords {
		if strings.Contains(lower, keyword) && (strings.Contains(lower, " to ") || strings.Contains(lower, " with ")) {
			return true
		}
	}
	return false
}

func handleModification(vfs *semantic.VFSTree, query string) {
	lower := strings.ToLower(query)

	// Extract: old value, new value, optional filename
	var oldValue, newValue, filename string

	// Pattern: "update X to Y in <file>"
	if strings.Contains(lower, " to ") {
		parts := strings.Split(query, " to ")
		if len(parts) >= 2 {
			// Get old value (everything after update/change/replace)
			firstPart := parts[0]
			for _, keyword := range []string{"update", "change", "replace", "modify"} {
				if idx := strings.Index(strings.ToLower(firstPart), keyword); idx >= 0 {
					oldValue = strings.TrimSpace(firstPart[idx+len(keyword):])
					break
				}
			}

			// Get new value and optional filename
			secondPart := parts[1]
			if strings.Contains(secondPart, " in ") {
				inParts := strings.Split(secondPart, " in ")
				newValue = strings.TrimSpace(inParts[0])
				if len(inParts) > 1 {
					filename = strings.TrimSpace(inParts[1])
				}
			} else {
				newValue = strings.TrimSpace(secondPart)
			}
		}
	}

	// Pattern: "replace X with Y"
	if oldValue == "" && strings.Contains(lower, " with ") {
		parts := strings.Split(query, " with ")
		if len(parts) >= 2 {
			firstPart := parts[0]
			if idx := strings.Index(strings.ToLower(firstPart), "replace"); idx >= 0 {
				oldValue = strings.TrimSpace(firstPart[idx+7:])
			}
			newValue = strings.TrimSpace(parts[1])
		}
	}

	if oldValue == "" || newValue == "" {
		fmt.Println("Could not parse modification. Use: 'update X to Y in file.go'")
		return
	}

	// If no filename specified, try to find the most recent file or use context
	if filename == "" {
		// Look for entrypoint
		nodes := vfs.GetByRole(string(semantic.RoleEntrypoint))
		if len(nodes) > 0 {
			filename = nodes[0].Name
		}
	}

	if filename == "" {
		fmt.Println("Please specify a filename: 'update X to Y in file.go'")
		return
	}

	// Find the file
	node, found := vfs.ResolvePath(filename)
	if !found {
		fmt.Printf("File '%s' not found\n", filename)
		return
	}

	if node.Type != "file" {
		fmt.Printf("'%s' is not a file\n", filename)
		return
	}

	// Perform replacement
	if !strings.Contains(node.Content, oldValue) {
		fmt.Printf("'%s' not found in %s\n", oldValue, filename)
		return
	}

	node.Content = strings.ReplaceAll(node.Content, oldValue, newValue)
	fmt.Printf("✓ Updated '%s' to '%s' in %s\n", oldValue, newValue, node.Path)
}

func processCommand(
	query string,
	vfs *semantic.VFSTree,
	blueprintEngine *semantic.BlueprintEngine,
	depGraph *semantic.DependencyGraph,
	roleRegistry *semantic.RoleRegistry,
	hierarchicalParser *semantic.HierarchicalParser,
	standardParser *semantic.CommandParser,
	templateRegistry *semantic.TemplateRegistry,
) {
	// Parse with NER
	nerSystem, err := ner.NewRuleBasedNER(query, "")
	if err != nil {
		log.Printf("NER error: %v\n", err)
		return
	}

	words := strings.Fields(query)
	entityMap := nerSystem.GetEntityMap()

	// Detect template/blueprint
	hasTemplate := false
	blueprintName := ""
	for _, word := range words {
		lowerWord := strings.ToLower(word)
		for _, tmpl := range templateRegistry.ListTemplates() {
			if lowerWord == strings.ToLower(tmpl) {
				hasTemplate = true
				blueprintName = tmpl
				break
			}
		}
		if hasTemplate {
			break
		}
	}

	// Parse command
	var semanticOutput semantic.SemanticOutput

	if hasTemplate {
		// Use hierarchical parser
		hierarchicalCmd := hierarchicalParser.Parse(query, words, entityMap)
		fmt.Println("\n" + hierarchicalCmd.String())

		// Extract parameters for blueprint
		bp, _ := blueprintEngine.GetBlueprint(blueprintName)
		params := blueprintEngine.ExtractParameters(query, bp)

		// Generate code with blueprint
		mainContent, _ := blueprintEngine.Execute(blueprintName, params)

		// Create in VFS
		folderName := hierarchicalCmd.Name
		if folderName == "" {
			folderName = "myproject"
		}

		folderPath := "/" + folderName
		vfs.CreateFolder(folderPath, string(semantic.RoleProjectRoot))

		// Create main file with blueprint content
		mainPath := folderPath + "/main.go"
		vfs.CreateFile(mainPath, string(semantic.RoleEntrypoint), mainContent)

		// Create additional blueprint files
		for _, bpFile := range bp.Files {
			content, _ := blueprintEngine.ExecuteFile(bp, bpFile, params)
			filePath := folderPath + "/" + bpFile.Name
			vfs.CreateFile(filePath, bpFile.Role, content)

			// Add dependency
			depGraph.AddDependency(mainPath, filePath, semantic.DepImport)
		}

		// Create folders from template
		for _, child := range hierarchicalCmd.Children {
			if child.ObjectType == semantic.ObjectFolder {
				childPath := folderPath + "/" + child.Name
				vfs.CreateFolder(childPath, string(semantic.RoleAssetDirectory))
			}
		}

		semanticOutput = semantic.FillFromHierarchicalCommand(hierarchicalCmd)

	} else {
		// Use standard parser
		structuredCmd := standardParser.Parse(query, words, entityMap)
		fmt.Printf("\n%s\n", structuredCmd.String())

		// Create in VFS
		if structuredCmd.Action == semantic.ActionCreate {
			if structuredCmd.ObjectType == semantic.ObjectFolder {
				path := "/" + structuredCmd.Name
				role := roleRegistry.InferRole(structuredCmd.Name, "folder", "")
				vfs.CreateFolder(path, role)
			} else if structuredCmd.ObjectType == semantic.ObjectFile {
				path := "/" + structuredCmd.Name
				role := roleRegistry.InferRole(structuredCmd.Name, "file", "")
				vfs.CreateFile(path, role, "")
			}
		}

		// Generate semantic output
		classifier := semantic.NewIntentClassifier()
		intent := classifier.Classify(query)
		extractor := semantic.NewEntityExtractor()
		entities := extractor.Extract(words, entityMap)

		filler := semantic.NewTemplateFiller()
		semanticOutput, err = filler.Fill(intent, entities)
		if err != nil {
			log.Printf("Template fill error: %v\n", err)
			return
		}
	}

	// Display JSON
	jsonBytes, _ := json.MarshalIndent(semanticOutput, "", "  ")
	fmt.Println("\nGenerated Output:")
	fmt.Println(string(jsonBytes))
	fmt.Println()
}
