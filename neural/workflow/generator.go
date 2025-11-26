package workflow

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/zendrulat/nlptagger/neural/semantic"
)

// GenerateWorkflow takes a SemanticOutput and converts it into a Workflow DAG.
func GenerateWorkflow(so *semantic.SemanticOutput) (*Workflow, error) {
	if so == nil {
		return nil, fmt.Errorf("SemanticOutput cannot be nil")
	}

	wf := NewWorkflow()

	// The primary target resource (e.g., the folder)
	primaryResource := so.TargetResource
	primaryNodeID := generateNodeID(primaryResource.Type, primaryResource.Name, 0)
	primaryNode := &Node{
		ID:        primaryNodeID,
		Operation: OperationType(so.Operation),
		Resource:  *primaryResource,
		Directory: primaryResource.Directory,
	}
	// If the operation is EXECUTE, set the command from semantic output
	if primaryNode.Operation == OperationExecute {
		primaryNode.Command = so.Command
		if primaryNode.Command == "" {
			fmt.Print("\nEnter a command to execute: ")
			reader := bufio.NewReader(os.Stdin)
			input, _ := reader.ReadString('\n')
			input = strings.TrimSpace(input)

			if input == "" {
				return nil, fmt.Errorf("command cannot be empty for EXECUTE operation")
			}
			primaryNode.Command = input
		}
	}

	// If resource name is empty, prompt the user for it
	if primaryNode.Resource.Name == "" {
		fmt.Print("\nEnter a resource name: ")
		reader := bufio.NewReader(os.Stdin)
		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)

		if input == "" {
			return nil, fmt.Errorf("resource name cannot be empty")
		}
		primaryNode.Resource.Name = input
	}

	if err := wf.AddNode(primaryNode); err != nil {
		return nil, fmt.Errorf("failed to add primary node to workflow: %w", err)
	}

	// Handle file creation if specified in properties
	if file, ok := so.TargetResource.Properties["file"]; ok {
		if fileName, ok := file.(string); ok {
			fileNodeID := generateNodeID("file", fileName, 0)
			fileNode := &Node{
				ID:           fileNodeID,
				Operation:    OperationWriteFile,
				FilePath:     fileName,
				Content:      "", // Empty content for now
				Dependencies: []string{primaryNodeID},
				Directory:    fmt.Sprintf("%s/%s", so.TargetResource.Directory, so.TargetResource.Name),
			}
			if err := wf.AddNode(fileNode); err != nil {
				return nil, fmt.Errorf("failed to add file node: %w", err)
			}
		}
	}

	if primaryResource.Type == "Filesystem::File" && so.Operation == "CREATE" {
		createFileNodeID := generateNodeID("file", "createfile", 0)
		createFileNode := &Node{
			ID:           createFileNodeID,
			Operation:    OperationWriteFile,
			Resource:     *primaryResource,
			FilePath:     primaryResource.Name,
			Content:      primaryResource.Content,
			Dependencies: []string{primaryNodeID},
			Directory:    primaryResource.Directory, // Use the directory from the resource
		}
		if err := wf.AddNode(createFileNode); err != nil {
			return nil, fmt.Errorf("failed to add create file node to workflow: %w", err)
		}
	}

	if err := addChildrenNodes(*primaryResource, primaryNodeID, wf, so.Operation, primaryResource.Name); err != nil {
		return nil, err
	}

	return wf, nil
}

// addChildrenNodes recursively adds nodes for child resources.
// currentParentPath is the full relative path to the parentResource from the application's CWD.
func addChildrenNodes(parentResource semantic.Resource, parentNodeID string, wf *Workflow, operation string, currentParentPath string) error {
	for i, childResource := range parentResource.Children {
		childNodeID := generateNodeID(childResource.Type, childResource.Name, i)

		// The directory for the child node is the currentParentPath
		childNodeDirectory := currentParentPath

		childNode := &Node{
			ID:           childNodeID,
			Operation:    OperationType(operation),
			Resource:     childResource,
			Dependencies: []string{parentNodeID},
			Directory:    childNodeDirectory, // This is the parent directory where the child resource will be created
		}
		if err := wf.AddNode(childNode); err != nil {
			return fmt.Errorf("failed to add child node to workflow: %w", err)
		}

		// If the child is a GoWebserver, add setup and start commands
		if childResource.Type == "Deployment::GoWebserver" && operation == "CREATE" {
		} else if childResource.Type == "Filesystem::File" && operation == "CREATE" {
			// Create an empty file
			createFileNodeID := generateNodeID("file", "createfile", i)
			createFileNode := &Node{
				ID:           createFileNodeID,
				Operation:    OperationWriteFile,
				FilePath:     childResource.Name,
				Content:      childResource.Content, // Use content from semantic resource
				Dependencies: []string{childNodeID},
				Directory:    childNodeDirectory, // Corrected: This is the parent directory where the file will be written
			}
			if err := wf.AddNode(createFileNode); err != nil {
				return fmt.Errorf("failed to add create file node to workflow: %w", err)
			}
		}

		// Recursively add children of the current child
		if err := addChildrenNodes(childResource, childNodeID, wf, operation, childNodeDirectory+"/"+childResource.Name); err != nil {
			return err
		}
	}
	return nil
}

func generateNodeID(resourceType, resourceName string, index int) string {
	return fmt.Sprintf("%s-%s-%d", resourceType, resourceName, index)
}
