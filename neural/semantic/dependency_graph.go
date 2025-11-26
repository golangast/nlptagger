package semantic

import "fmt"

// DependencyType represents the type of dependency relationship
type DependencyType string

const (
	DepImport DependencyType = "import" // Go import statement
	DepConfig DependencyType = "config" // Configuration reference
	DepAsset  DependencyType = "asset"  // Asset/resource reference
)

// Dependency represents a relationship between files
type Dependency struct {
	SourceFile string         // File that has the dependency
	TargetFile string         // File being depended on
	Type       DependencyType // Type of dependency
	LineNumber int            // Where the dependency is declared (0 if unknown)
}

// DependencyGraph tracks relationships between files
type DependencyGraph struct {
	// Outgoing dependencies: file -> list of files it depends on
	outgoing map[string][]Dependency

	// Incoming dependencies: file -> list of files that depend on it
	incoming map[string][]Dependency
}

// NewDependencyGraph creates a new dependency graph
func NewDependencyGraph() *DependencyGraph {
	return &DependencyGraph{
		outgoing: make(map[string][]Dependency),
		incoming: make(map[string][]Dependency),
	}
}

// AddDependency records a dependency relationship
func (dg *DependencyGraph) AddDependency(sourceFile, targetFile string, depType DependencyType) {
	dep := Dependency{
		SourceFile: sourceFile,
		TargetFile: targetFile,
		Type:       depType,
	}

	// Add to outgoing
	dg.outgoing[sourceFile] = append(dg.outgoing[sourceFile], dep)

	// Add to incoming
	dg.incoming[targetFile] = append(dg.incoming[targetFile], dep)
}

// GetDependencies returns all files that sourceFile depends on
func (dg *DependencyGraph) GetDependencies(sourceFile string) []Dependency {
	return dg.outgoing[sourceFile]
}

// GetDependents returns all files that depend on targetFile
func (dg *DependencyGraph) GetDependents(targetFile string) []Dependency {
	return dg.incoming[targetFile]
}

// UpdateReferences updates all references when a file is renamed
func (dg *DependencyGraph) UpdateReferences(oldPath, newPath string) []string {
	affected := make([]string, 0)

	// Find all files that depend on the old path
	dependents := dg.incoming[oldPath]

	for _, dep := range dependents {
		affected = append(affected, dep.SourceFile)
	}

	// Update the graph
	if deps, exists := dg.outgoing[oldPath]; exists {
		dg.outgoing[newPath] = deps
		delete(dg.outgoing, oldPath)

		// Update targets in outgoing dependencies
		for i := range dg.outgoing[newPath] {
			dg.outgoing[newPath][i].SourceFile = newPath
		}
	}

	if deps, exists := dg.incoming[oldPath]; exists {
		dg.incoming[newPath] = deps
		delete(dg.incoming, oldPath)

		// Update sources that point to this file
		for _, dep := range deps {
			// Update in source file's outgoing list
			if outList, ok := dg.outgoing[dep.SourceFile]; ok {
				for i := range outList {
					if outList[i].TargetFile == oldPath {
						outList[i].TargetFile = newPath
					}
				}
			}
		}
	}

	return affected
}

// HasCycle detects circular dependencies
func (dg *DependencyGraph) HasCycle() bool {
	visited := make(map[string]bool)
	recStack := make(map[string]bool)

	for file := range dg.outgoing {
		if dg.hasCycleUtil(file, visited, recStack) {
			return true
		}
	}

	return false
}

func (dg *DependencyGraph) hasCycleUtil(file string, visited, recStack map[string]bool) bool {
	if recStack[file] {
		return true // Cycle detected
	}

	if visited[file] {
		return false // Already processed
	}

	visited[file] = true
	recStack[file] = true

	// Check all dependencies
	for _, dep := range dg.outgoing[file] {
		if dg.hasCycleUtil(dep.TargetFile, visited, recStack) {
			return true
		}
	}

	recStack[file] = false
	return false
}

// Summary returns a string representation of the dependency graph
func (dg *DependencyGraph) Summary() string {
	var result string

	result += "Dependency Graph:\n"
	for file, deps := range dg.outgoing {
		if len(deps) > 0 {
			result += fmt.Sprintf("  %s depends on:\n", file)
			for _, dep := range deps {
				result += fmt.Sprintf("    - %s (%s)\n", dep.TargetFile, dep.Type)
			}
		}
	}

	return result
}
