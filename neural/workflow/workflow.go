package workflow

import (
	"fmt"

	"github.com/zendrulat/nlptagger/neural/semantic"
)

// OperationType defines the type of operation a node represents.
type OperationType string

const (
	OperationCreate    OperationType = "CREATE"
	OperationExecute   OperationType = "EXECUTE"
	OperationStart     OperationType = "START"
	OperationDelete    OperationType = "DELETE"
	OperationWriteFile OperationType = "WRITE_FILE"
	// Add other operation types as needed
)

// Node represents a single step in the workflow (a node in the DAG).
type Node struct {
	ID           string                   `json:"id"`
	Operation    OperationType            `json:"operation"`
	Resource     semantic.Resource        `json:"resource"`
	Command      string                   `json:"command,omitempty"`      // For EXECUTE operations
	FilePath     string                   `json:"filepath,omitempty"`     // For WRITE_FILE operations
	Content      string                   `json:"content,omitempty"`      // For WRITE_FILE operations
	Directory    string                   `json:"directory,omitempty"`    // Directory to execute command in
	Dependencies []string                 `json:"dependencies,omitempty"` // IDs of nodes this node depends on
	Context      semantic.Context         `json:"context,omitempty"`
}

// Workflow represents a Directed Acyclic Graph (DAG) of operations.
type Workflow struct {
	Nodes []*Node `json:"nodes"`
}

// NewWorkflow creates an empty workflow.
func NewWorkflow() *Workflow {
	return &Workflow{
		Nodes: []*Node{},
	}
}

// AddNode adds a node to the workflow.
func (w *Workflow) AddNode(node *Node) error {
	for _, existingNode := range w.Nodes {
		if existingNode.ID == node.ID {
			return fmt.Errorf("node with ID '%s' already exists", node.ID)
		}
	}
	w.Nodes = append(w.Nodes, node)
	return nil
}

// GetNode retrieves a node by its ID.
func (w *Workflow) GetNode(id string) *Node {
	for _, node := range w.Nodes {
		if node.ID == id {
			return node
		}
	}
	return nil
}

// TopologicalSort performs a topological sort of the workflow nodes.
// It returns a slice of node IDs in topological order, or an error if a cycle is detected.
func (w *Workflow) TopologicalSort() ([]string, error) {
	// Kahn's algorithm for topological sort
	inDegree := make(map[string]int)
	adj := make(map[string][]string)

	for _, node := range w.Nodes {
		inDegree[node.ID] = 0
		adj[node.ID] = []string{}
	}

	for _, node := range w.Nodes {
		for _, depID := range node.Dependencies {
			if _, ok := inDegree[depID]; !ok {
				return nil, fmt.Errorf("dependency '%s' for node '%s' not found in workflow", depID, node.ID)
			}
			inDegree[node.ID]++
			adj[depID] = append(adj[depID], node.ID)
		}
	}

	queue := []string{}
	for id, degree := range inDegree {
		if degree == 0 {
			queue = append(queue, id)
		}
	}

	sortedList := []string{}
	for len(queue) > 0 {
		u := queue[0]
		queue = queue[1:]
		sortedList = append(sortedList, u)

		for _, v := range adj[u] {
			inDegree[v]--
			if inDegree[v] == 0 {
				queue = append(queue, v)
			}
		}
	}

	if len(sortedList) != len(w.Nodes) {
		return nil, fmt.Errorf("cycle detected in workflow")
	}

	return sortedList, nil
}
