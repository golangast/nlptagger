package workflow

import (
	"fmt"
	"path"

	"nlptagger/neural/semantic"
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
		Resource:  primaryResource,
	}
	if err := wf.AddNode(primaryNode); err != nil {
		return nil, fmt.Errorf("failed to add primary node to workflow: %w", err)
	}

	if err := addChildrenNodes(primaryResource, primaryNodeID, wf, so.Operation, primaryResource.Name); err != nil {
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
			// The webserver's directory is its own path: currentParentPath/childResource.Name
			webserverDirectory := path.Join(currentParentPath, childResource.Name)

			// Create go.mod
			setupCommand := fmt.Sprintf("go mod init %s", childResource.Name)
			setupNodeID := generateNodeID("command", "gomodinit", i)
			setupNode := &Node{
				ID:           setupNodeID,
				Operation:    OperationExecute,
				Command:      setupCommand,
				Dependencies: []string{childNodeID}, // Setup depends on webserver creation
				Directory:    webserverDirectory,    // Execute in the webserver's own folder
			}
			if err := wf.AddNode(setupNode); err != nil {
				return fmt.Errorf("failed to add setup command node to workflow: %w", err)
			}

			// Create main.go
			mainGoContent := `package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
)

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "Hello, you've requested: %s\n", r.URL.Path)
	})

	log.Printf("Server starting on port %s\n", port)
	log.Fatal(http.ListenAndServe(fmt.Sprintf(":%s", port), nil))
}
`
			createMainGoNodeID := generateNodeID("file", "createmain.go", i)
			createMainGoNode := &Node{
				ID:           createMainGoNodeID,
				Operation:    OperationWriteFile,
				FilePath:     "main.go",
				Content:      mainGoContent,
				Dependencies: []string{setupNodeID}, // main.go creation depends on go mod init
				Directory:    webserverDirectory,    // Write in the child resource's own directory
			}
			if err := wf.AddNode(createMainGoNode); err != nil {
				return fmt.Errorf("failed to add create main.go node to workflow: %w", err)
			}

			// Verify go.mod exists
			verifyGoModNodeID := generateNodeID("command", "verifygomod", i)
			verifyGoModNode := &Node{
				ID:           verifyGoModNodeID,
				Operation:    OperationExecute,
				Command:      "test -f go.mod",
				Dependencies: []string{createMainGoNodeID},
				Directory:    webserverDirectory,
			}
			if err := wf.AddNode(verifyGoModNode); err != nil {
				return fmt.Errorf("failed to add verify go.mod node to workflow: %w", err)
			}

			// Verify main.go exists
			verifyMainGoNodeID := generateNodeID("command", "verifymain.go", i)
			verifyMainGoNode := &Node{
				ID:           verifyMainGoNodeID,
				Operation:    OperationExecute,
				Command:      "test -f main.go",
				Dependencies: []string{verifyGoModNodeID},
				Directory:    webserverDirectory,
			}
			if err := wf.AddNode(verifyMainGoNode); err != nil {
				return fmt.Errorf("failed to add verify main.go node to workflow: %w", err)
			}

			// Start webserver
			startNodeID := generateNodeID("command", "startwebserver", i)
			startNode := &Node{
				ID:           startNodeID,
				Operation:    OperationStart,
				Command:      fmt.Sprintf("PORT=%v go run main.go", childResource.Properties["port"]),
				Dependencies: []string{verifyMainGoNodeID},
				Directory:    webserverDirectory,
			}
			if err := wf.AddNode(startNode); err != nil {
				return fmt.Errorf("failed to add start command node to workflow: %w", err)
			}
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

		// Recursive call for grandchildren
		if len(childResource.Children) > 0 {
			// The path for the grandchildren's parent is the path to the current child resource
			nextParentPath := path.Join(currentParentPath, childResource.Name)
			if err := addChildrenNodes(childResource, childNodeID, wf, operation, nextParentPath); err != nil {
				return err
			}
		}
	}
	return nil
}

func generateNodeID(resourceType, resourceName string, index int) string {
	return fmt.Sprintf("%s-%s-%d", resourceType, resourceName, index)
}