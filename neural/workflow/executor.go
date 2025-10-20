package workflow

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"
)

// Executor is responsible for executing the nodes in a workflow.
type Executor struct {
	// Add any necessary dependencies, e.g., a client for interacting with external systems
}

// NewExecutor creates a new Executor.
func NewExecutor() *Executor {
	return &Executor{}
}

// ExecuteWorkflow executes the given workflow by traversing its nodes
// in topological order.
func (e *Executor) ExecuteWorkflow(wf *Workflow) error {
	if wf == nil {
		return fmt.Errorf("workflow cannot be nil")
	}

	// Get nodes in topological order
	sortedNodeIDs, err := wf.TopologicalSort()
	if err != nil {
		return fmt.Errorf("failed to get topological sort of workflow: %w", err)
	}

	log.Printf("Executing workflow with %d nodes in topological order: %v", len(sortedNodeIDs), sortedNodeIDs)

	for _, nodeID := range sortedNodeIDs {
		node := wf.GetNode(nodeID)
		if node == nil {
			return fmt.Errorf("node with ID '%s' not found in workflow", nodeID)
		}

		log.Printf("Executing node '%s': Operation=%s, ResourceType=%s, ResourceName=%s, Directory=%s",
			node.ID, node.Operation, node.Resource.Type, node.Resource.Name, node.Directory)

		if err := e.executeNode(node); err != nil {
			return fmt.Errorf("failed to execute node '%s': %w", node.ID, err)
		}
	}

	log.Println("Workflow execution completed successfully.")
	return nil
}

// executeNode performs the action defined by a single workflow node.
func (e *Executor) executeNode(node *Node) error {
	switch node.Operation {
	case OperationCreate:
		return e.handleCreate(node)
	case OperationExecute:
		return e.handleExecute(node)
	case OperationStart:
		return e.handleStart(node)
	case OperationDelete:
		return e.handleDelete(node)
	case OperationWriteFile:
		return e.handleWriteFile(node)
	default:
		return fmt.Errorf("unsupported operation type: %s", node.Operation)
	}
}

func (e *Executor) handleCreate(node *Node) error {
	log.Printf("Handling CREATE operation for resource: Type=%s, Name=%s, Properties=%v",
		node.Resource.Type, node.Resource.Name, node.Resource.Properties)

	switch node.Resource.Type {
	case "Filesystem::Folder":
		folderPath := node.Resource.Name
		if node.Directory != "" {
			folderPath = node.Directory + "/" + node.Resource.Name
		}
		log.Printf("Creating folder: %s", folderPath)
		if err := os.MkdirAll(folderPath, 0755); err != nil {
			return fmt.Errorf("failed to create folder %s: %w", folderPath, err)
		}
	case "Deployment::GoWebserver":
		log.Printf("Simulating Go Webserver deployment: %s on port %v with image %s",
			node.Resource.Name, node.Resource.Properties["port"], node.Resource.Properties["runtime_image"])
	default:
		log.Printf("No specific CREATE handler for resource type: %s", node.Resource.Type)
	}

	return nil
}

func (e *Executor) handleExecute(node *Node) error {
	if node.Command == "" {
		return fmt.Errorf("EXECUTE operation requires a command")
	}
	log.Printf("Executing command: %s in directory: %s", node.Command, node.Directory)

	// Special handling for 'go mod init'
	if strings.HasPrefix(node.Command, "go mod init") {
		goModPath := "go.mod"
		if node.Directory != "" {
			goModPath = node.Directory + "/go.mod"
		}
		if _, err := os.Stat(goModPath); err == nil {
			log.Printf("go.mod already exists, skipping 'go mod init'.")
			return nil
		}
	}

	cmd := exec.Command("bash", "-c", node.Command)
	if node.Directory != "" {
		cmd.Dir = node.Directory
	}
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("command execution failed: %s, output: %s", err, string(output))
	}

	log.Printf("Command output: %s", string(output))
	return nil
}

func (e *Executor) handleWriteFile(node *Node) error {
	if node.FilePath == "" {
		return fmt.Errorf("WRITE_FILE operation requires a file path")
	}

	filePath := node.FilePath
	if node.Directory != "" {
		filePath = node.Directory + "/" + filePath
	}

	log.Printf("Writing file: %s", filePath)
	// Write content to the file, creating it if it doesn't exist, and overwriting it if it does.
	if err := os.WriteFile(filePath, []byte(node.Content), 0644); err != nil {
		return fmt.Errorf("failed to write file %s: %w", filePath, err)
	}

	return nil
}

func (e *Executor) handleStart(node *Node) error {
	if node.Command == "" {
		return fmt.Errorf("START operation requires a command")
	}
	log.Printf("Starting process: %s (in background) in directory: %s", node.Command, node.Directory)

	cmd := exec.Command("bash", "-c", node.Command)
	if node.Directory != "" {
		cmd.Dir = node.Directory
	}
	// Start the command without waiting for it to finish
	err := cmd.Start()
	if err != nil {
		return fmt.Errorf("failed to start command: %w", err)
	}
	log.Printf("Process started with PID: %d", cmd.Process.Pid)

	// Note: In a real application, you'd want to store cmd.Process for later management (e.g., stopping).
	return nil
}

func (e *Executor) handleDelete(node *Node) error {
	log.Printf("Handling DELETE operation for resource: Type=%s, Name=%s",
		node.Resource.Type, node.Resource.Name)
	// Placeholder for actual resource deletion logic
	log.Printf("Simulating deletion of resource: %s %s", node.Resource.Type, node.Resource.Name)
	return nil
}
