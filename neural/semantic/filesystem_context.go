package semantic

import (
	"path/filepath"
	"strings"
)

// FilesystemContext maintains state of the virtual filesystem
type FilesystemContext struct {
	// Current working directory in virtual filesystem
	CurrentDirectory string

	// Map of all known files: filepath -> properties
	Files map[string]map[string]interface{}

	// Map of all known folders: folderpath -> children
	Folders map[string][]string

	// Last created project root
	LastProjectRoot string
}

// NewFilesystemContext creates a new filesystem context
func NewFilesystemContext() *FilesystemContext {
	return &FilesystemContext{
		CurrentDirectory: "./",
		Files:            make(map[string]map[string]interface{}),
		Folders:          make(map[string][]string),
	}
}

// RecordCommand updates context based on a semantic output
func (fc *FilesystemContext) RecordCommand(output SemanticOutput) {
	if output.Operation == "create" || output.Operation == "Create" {
		fc.recordCreation(output.TargetResource)
	}
}

// recordCreation records a created resource and its children
func (fc *FilesystemContext) recordCreation(resource *Resource) {
	if resource == nil {
		return
	}
	resourceType := strings.ToLower(resource.Type)

	// Determine full path
	basePath := "./"
	if path, ok := resource.Properties["path"].(string); ok && path != "" {
		basePath = path
	}

	fullPath := filepath.Join(basePath, resource.Name)

	if strings.Contains(resourceType, "folder") {
		// Record folder
		fc.Folders[fullPath] = make([]string, 0)

		// Set as current directory if it's a project root
		if len(fc.Folders) == 1 || basePath == "./" {
			fc.CurrentDirectory = fullPath
			fc.LastProjectRoot = fullPath
		}

		// Record children
		for i := range resource.Children {
			child := &resource.Children[i]
			childPath := filepath.Join(fullPath, child.Name)

			if strings.Contains(strings.ToLower(child.Type), "folder") {
				fc.Folders[childPath] = make([]string, 0)
				fc.Folders[fullPath] = append(fc.Folders[fullPath], child.Name)
			} else if strings.Contains(strings.ToLower(child.Type), "file") {
				fc.Files[childPath] = child.Properties
				fc.Folders[fullPath] = append(fc.Folders[fullPath], child.Name)
			}

			// Recursively record nested children
			fc.recordCreation(child)
		}
	} else if strings.Contains(resourceType, "file") {
		// Record file
		fc.Files[fullPath] = resource.Properties

		// Add to parent folder's children
		parentDir := filepath.Dir(fullPath)
		if children, ok := fc.Folders[parentDir]; ok {
			fc.Folders[parentDir] = append(children, resource.Name)
		}
	}
}

// ResolvePath attempts to resolve a name to a full path
func (fc *FilesystemContext) ResolvePath(name string) string {
	// Check if it's already a full path
	if strings.HasPrefix(name, "./") || strings.HasPrefix(name, "/") {
		return name
	}

	// Try exact match in current directory
	currentPath := filepath.Join(fc.CurrentDirectory, name)
	if fc.FileExists(currentPath) || fc.FolderExists(currentPath) {
		return currentPath
	}

	// Search in project root
	if fc.LastProjectRoot != "" {
		projectPath := filepath.Join(fc.LastProjectRoot, name)
		if fc.FileExists(projectPath) || fc.FolderExists(projectPath) {
			return projectPath
		}
	}

	// Search all folders for a match
	for folderPath := range fc.Folders {
		testPath := filepath.Join(folderPath, name)
		if fc.FileExists(testPath) || fc.FolderExists(testPath) {
			return testPath
		}
	}

	// Default to current directory
	return currentPath
}

// FileExists checks if a file exists in context
func (fc *FilesystemContext) FileExists(path string) bool {
	_, exists := fc.Files[path]
	return exists
}

// FolderExists checks if a folder exists in context
func (fc *FilesystemContext) FolderExists(path string) bool {
	_, exists := fc.Folders[path]
	return exists
}

// GetFolderContents returns the contents of a folder
func (fc *FilesystemContext) GetFolderContents(path string) []string {
	if children, ok := fc.Folders[path]; ok {
		return children
	}
	return []string{}
}

// FindFolder searches for a folder by name
func (fc *FilesystemContext) FindFolder(name string) (string, bool) {
	// Check current directory first
	currentPath := filepath.Join(fc.CurrentDirectory, name)
	if fc.FolderExists(currentPath) {
		return currentPath, true
	}

	// Search all folders
	for folderPath := range fc.Folders {
		if filepath.Base(folderPath) == name {
			return folderPath, true
		}
		// Also check subfolders
		subPath := filepath.Join(folderPath, name)
		if fc.FolderExists(subPath) {
			return subPath, true
		}
	}

	return "", false
}

// FindFile searches for a file by name
func (fc *FilesystemContext) FindFile(name string) (string, bool) {
	// Check current directory first
	currentPath := filepath.Join(fc.CurrentDirectory, name)
	if fc.FileExists(currentPath) {
		return currentPath, true
	}

	// Search all files
	for filePath := range fc.Files {
		if filepath.Base(filePath) == name {
			return filePath, true
		}
	}

	return "", false
}

// Summary returns a summary of the current context
func (fc *FilesystemContext) Summary() string {
	var sb strings.Builder
	sb.WriteString("Current Directory: ")
	sb.WriteString(fc.CurrentDirectory)
	sb.WriteString("\n")

	if fc.LastProjectRoot != "" {
		sb.WriteString("Project Root: ")
		sb.WriteString(fc.LastProjectRoot)
		sb.WriteString("\n")
	}

	if len(fc.Folders) > 0 {
		sb.WriteString("\nFolders:\n")
		for folder, children := range fc.Folders {
			sb.WriteString("  ")
			sb.WriteString(folder)
			if len(children) > 0 {
				sb.WriteString(" (")
				sb.WriteString(strings.Join(children, ", "))
				sb.WriteString(")")
			}
			sb.WriteString("\n")
		}
	}

	if len(fc.Files) > 0 {
		sb.WriteString("\nFiles:\n")
		for file := range fc.Files {
			sb.WriteString("  ")
			sb.WriteString(file)
			sb.WriteString("\n")
		}
	}

	return sb.String()
}
