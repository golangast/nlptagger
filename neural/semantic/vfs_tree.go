package semantic

import (
	"fmt"
	"path/filepath"
	"strings"
	"time"
)

// VFSNode represents a node in the virtual file system
type VFSNode struct {
	Name         string                 // Node name (e.g., "main.go", "src")
	Path         string                 // Full path from root
	Type         string                 // "file" or "folder"
	Role         string                 // Semantic role: "entrypoint", "handler", "config", etc.
	Content      string                 // File content (empty for folders)
	Properties   map[string]interface{} // Additional properties
	Parent       *VFSNode               // Parent node
	Children     map[string]*VFSNode    // Child nodes (for folders)
	Dependencies []string               // Files this node depends on
	CreatedAt    time.Time              // Creation timestamp
	ModifiedAt   time.Time              // Last modification timestamp
}

// VFSTree represents the virtual file system
type VFSTree struct {
	Root       *VFSNode              // Root node
	CurrentDir *VFSNode              // Current working directory
	PathIndex  map[string]*VFSNode   // Path -> Node index for fast lookup
	RoleIndex  map[string][]*VFSNode // Role -> Nodes for role-based lookup
	NameIndex  map[string][]*VFSNode // Name -> Nodes for name-based lookup
}

// NewVFSTree creates a new virtual file system
func NewVFSTree() *VFSTree {
	root := &VFSNode{
		Name:       "",
		Path:       "/",
		Type:       "folder",
		Role:       "root",
		Children:   make(map[string]*VFSNode),
		Properties: make(map[string]interface{}),
		CreatedAt:  time.Now(),
		ModifiedAt: time.Now(),
	}

	tree := &VFSTree{
		Root:       root,
		CurrentDir: root,
		PathIndex:  make(map[string]*VFSNode),
		RoleIndex:  make(map[string][]*VFSNode),
		NameIndex:  make(map[string][]*VFSNode),
	}

	tree.PathIndex["/"] = root
	tree.indexRole(root)

	return tree
}

// CreateFolder creates a new folder in the VFS
func (vfs *VFSTree) CreateFolder(path string, role string) (*VFSNode, error) {
	// Normalize path
	path = vfs.normalizePath(path)

	// Check for conflicts
	if existing, exists := vfs.PathIndex[path]; exists {
		return nil, fmt.Errorf("path '%s' already exists as %s", path, existing.Type)
	}

	// Get parent
	parentPath := filepath.Dir(path)
	parent, exists := vfs.PathIndex[parentPath]
	if !exists {
		return nil, fmt.Errorf("parent directory '%s' does not exist", parentPath)
	}

	if parent.Type != "folder" {
		return nil, fmt.Errorf("parent '%s' is not a folder", parentPath)
	}

	// Create node
	name := filepath.Base(path)
	node := &VFSNode{
		Name:       name,
		Path:       path,
		Type:       "folder",
		Role:       role,
		Children:   make(map[string]*VFSNode),
		Properties: make(map[string]interface{}),
		Parent:     parent,
		CreatedAt:  time.Now(),
		ModifiedAt: time.Now(),
	}

	// Add to parent
	parent.Children[name] = node
	parent.ModifiedAt = time.Now()

	// Update indices
	vfs.PathIndex[path] = node
	vfs.indexRole(node)
	vfs.indexName(node)

	return node, nil
}

// CreateFile creates a new file in the VFS
func (vfs *VFSTree) CreateFile(path string, role string, content string) (*VFSNode, error) {
	// Normalize path
	path = vfs.normalizePath(path)

	// Check for conflicts
	if existing, exists := vfs.PathIndex[path]; exists {
		return nil, fmt.Errorf("path '%s' already exists as %s", path, existing.Type)
	}

	// Get parent
	parentPath := filepath.Dir(path)
	parent, exists := vfs.PathIndex[parentPath]
	if !exists {
		return nil, fmt.Errorf("parent directory '%s' does not exist", parentPath)
	}

	if parent.Type != "folder" {
		return nil, fmt.Errorf("parent '%s' is not a folder", parentPath)
	}

	// Create node
	name := filepath.Base(path)
	node := &VFSNode{
		Name:       name,
		Path:       path,
		Type:       "file",
		Role:       role,
		Content:    content,
		Properties: make(map[string]interface{}),
		Parent:     parent,
		Children:   nil,
		CreatedAt:  time.Now(),
		ModifiedAt: time.Now(),
	}

	// Add to parent
	parent.Children[name] = node
	parent.ModifiedAt = time.Now()

	// Update indices
	vfs.PathIndex[path] = node
	vfs.indexRole(node)
	vfs.indexName(node)

	return node, nil
}

// ResolvePath finds a node by name (searches from current dir, then globally)
func (vfs *VFSTree) ResolvePath(name string) (*VFSNode, bool) {
	// Try exact path first
	if node, exists := vfs.PathIndex[name]; exists {
		return node, true
	}

	// Try in current directory
	currentPath := filepath.Join(vfs.CurrentDir.Path, name)
	if node, exists := vfs.PathIndex[currentPath]; exists {
		return node, true
	}

	// Search by name
	if nodes, exists := vfs.NameIndex[name]; exists && len(nodes) > 0 {
		return nodes[0], true // Return first match
	}

	return nil, false
}

// GetByRole returns all nodes with a specific role
func (vfs *VFSTree) GetByRole(role string) []*VFSNode {
	if nodes, exists := vfs.RoleIndex[role]; exists {
		return nodes
	}
	return []*VFSNode{}
}

// Rename renames a node and updates all indices
func (vfs *VFSTree) Rename(oldPath string, newName string) error {
	node, exists := vfs.PathIndex[oldPath]
	if !exists {
		return fmt.Errorf("path '%s' not found", oldPath)
	}

	// Calculate new path
	parentPath := filepath.Dir(oldPath)
	newPath := filepath.Join(parentPath, newName)

	// Check for conflicts
	if _, exists := vfs.PathIndex[newPath]; exists {
		return fmt.Errorf("path '%s' already exists", newPath)
	}

	// Update indices (remove old)
	delete(vfs.PathIndex, oldPath)
	vfs.removeFromNameIndex(node)

	// Update node
	oldName := node.Name
	node.Name = newName
	node.Path = newPath
	node.ModifiedAt = time.Now()

	// Update parent's children map
	if node.Parent != nil {
		delete(node.Parent.Children, oldName)
		node.Parent.Children[newName] = node
	}

	// Update indices (add new)
	vfs.PathIndex[newPath] = node
	vfs.indexName(node)

	// Recursively update children paths
	if node.Type == "folder" {
		vfs.updateChildrenPaths(node)
	}

	return nil
}

// ChangeDirectory changes the current working directory
func (vfs *VFSTree) ChangeDirectory(path string) error {
	node, exists := vfs.PathIndex[path]
	if !exists {
		return fmt.Errorf("path '%s' not found", path)
	}

	if node.Type != "folder" {
		return fmt.Errorf("'%s' is not a folder", path)
	}

	vfs.CurrentDir = node
	return nil
}

// Exists checks if a path exists
func (vfs *VFSTree) Exists(path string) bool {
	_, exists := vfs.PathIndex[path]
	return exists
}

// Get retrieves a node by path
func (vfs *VFSTree) Get(path string) (*VFSNode, bool) {
	node, exists := vfs.PathIndex[path]
	return node, exists
}

// normalizePath normalizes a path (removes ./, handles relative paths)
func (vfs *VFSTree) normalizePath(path string) string {
	// Remove leading ./
	path = strings.TrimPrefix(path, "./")

	// If path doesn't start with /, make it relative to current dir
	if !strings.HasPrefix(path, "/") {
		path = filepath.Join(vfs.CurrentDir.Path, path)
	}

	return filepath.Clean(path)
}

// indexRole adds a node to the role index
func (vfs *VFSTree) indexRole(node *VFSNode) {
	if node.Role == "" {
		return
	}

	if _, exists := vfs.RoleIndex[node.Role]; !exists {
		vfs.RoleIndex[node.Role] = make([]*VFSNode, 0)
	}

	vfs.RoleIndex[node.Role] = append(vfs.RoleIndex[node.Role], node)
}

// indexName adds a node to the name index
func (vfs *VFSTree) indexName(node *VFSNode) {
	if _, exists := vfs.NameIndex[node.Name]; !exists {
		vfs.NameIndex[node.Name] = make([]*VFSNode, 0)
	}

	vfs.NameIndex[node.Name] = append(vfs.NameIndex[node.Name], node)
}

// removeFromNameIndex removes a node from the name index
func (vfs *VFSTree) removeFromNameIndex(node *VFSNode) {
	if nodes, exists := vfs.NameIndex[node.Name]; exists {
		// Remove this specific node
		filtered := make([]*VFSNode, 0)
		for _, n := range nodes {
			if n != node {
				filtered = append(filtered, n)
			}
		}
		if len(filtered) > 0 {
			vfs.NameIndex[node.Name] = filtered
		} else {
			delete(vfs.NameIndex, node.Name)
		}
	}
}

// updateChildrenPaths recursively updates paths for all children
func (vfs *VFSTree) updateChildrenPaths(node *VFSNode) {
	if node.Type != "folder" {
		return
	}

	for _, child := range node.Children {
		oldPath := child.Path
		newPath := filepath.Join(node.Path, child.Name)

		// Update indices
		delete(vfs.PathIndex, oldPath)
		child.Path = newPath
		vfs.PathIndex[newPath] = child

		// Recurse
		if child.Type == "folder" {
			vfs.updateChildrenPaths(child)
		}
	}
}

// Tree returns a string representation of the tree
func (vfs *VFSTree) Tree() string {
	var sb strings.Builder
	vfs.printNode(&sb, vfs.Root, "", true)
	return sb.String()
}

func (vfs *VFSTree) printNode(sb *strings.Builder, node *VFSNode, prefix string, isLast bool) {
	if node == vfs.Root {
		sb.WriteString("/\n")
	} else {
		connector := "├── "
		if isLast {
			connector = "└── "
		}

		sb.WriteString(prefix)
		sb.WriteString(connector)
		sb.WriteString(node.Name)

		if node.Role != "" {
			sb.WriteString(" [")
			sb.WriteString(node.Role)
			sb.WriteString("]")
		}
		sb.WriteString("\n")
	}

	if node.Type == "folder" && len(node.Children) > 0 {
		childPrefix := prefix
		if node != vfs.Root {
			if isLast {
				childPrefix += "    "
			} else {
				childPrefix += "│   "
			}
		}

		i := 0
		total := len(node.Children)
		for _, child := range node.Children {
			i++
			vfs.printNode(sb, child, childPrefix, i == total)
		}
	}
}

// Delete removes a node from the tree
func (vfs *VFSTree) Delete(path string) error {
	path = vfs.normalizePath(path)
	node, exists := vfs.ResolvePath(path)
	if !exists {
		return fmt.Errorf("path not found: %s", path)
	}

	if node.Parent == nil {
		return fmt.Errorf("cannot delete root node")
	}

	// Remove from parent's children
	delete(node.Parent.Children, node.Name)

	// Remove from path index
	delete(vfs.PathIndex, path)

	// Remove from role index
	if node.Role != "" {
		if roles, ok := vfs.RoleIndex[node.Role]; ok {
			for i, n := range roles {
				if n == node {
					vfs.RoleIndex[node.Role] = append(roles[:i], roles[i+1:]...)
					break
				}
			}
		}
	}

	// Remove from name index
	vfs.removeFromNameIndex(node)

	return nil
}
