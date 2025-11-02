package workflow

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// RuleCondition defines the conditions for a rule to be applied.
type RuleCondition struct {
	Operation    OperationType
	ResourceType string
}

// RuleAction defines the function to be executed if a rule's conditions are met.
type RuleAction func(node *Node) error

// Rule represents a single rule in the rule engine.
type Rule struct {
	Condition RuleCondition
	Action    RuleAction
}

// RuleEngine manages and executes a set of rules.
type RuleEngine struct {
	rules []Rule
}

// NewRuleEngine creates a new RuleEngine.
func NewRuleEngine() *RuleEngine {
	return &RuleEngine{
		rules: make([]Rule, 0),
	}
}

// RegisterRule adds a new rule to the engine.
func (re *RuleEngine) RegisterRule(condition RuleCondition, action RuleAction) {
	re.rules = append(re.rules, Rule{Condition: condition, Action: action})
}

// ExecuteRules evaluates the rules against a given node and executes the action of the first matching rule.
func (re *RuleEngine) ExecuteRules(node *Node) error {
	for _, rule := range re.rules {
		// Corrected condition: if rule.Condition.ResourceType is empty, it matches any resource type.
		// Otherwise, it must match the node's resource type.
		if rule.Condition.Operation == node.Operation && (rule.Condition.ResourceType == "" || rule.Condition.ResourceType == node.Resource.Type) {
			log.Printf("Applying rule for Operation=%s, ResourceType=%s", node.Operation, node.Resource.Type)
			return rule.Action(node)
		}
	}
	return fmt.Errorf("no rule found for Operation=%s, ResourceType=%s", node.Operation, node.Resource.Type)
}

// RegisterDefaultCreateRules registers the default rules for CREATE operations.
func (re *RuleEngine) RegisterDefaultCreateRules() {
	re.RegisterRule(
		RuleCondition{Operation: OperationCreate, ResourceType: "Filesystem::Folder"},
		func(node *Node) error {
			folderPath := node.Resource.Name
			if node.Directory != "" {
				folderPath = filepath.Join(node.Directory, node.Resource.Name)
			}
			log.Printf("Creating folder: %s", folderPath)
			if err := os.MkdirAll(folderPath, 0755); err != nil {
				return fmt.Errorf("failed to create folder %s: %w", folderPath, err)
			}
			return nil
		},
	)

	re.RegisterRule(
		RuleCondition{Operation: OperationCreate, ResourceType: "Deployment::GoWebserver"},
		func(node *Node) error {
			log.Printf("Simulating Go Webserver deployment: %s on port %v with image %s",
				node.Resource.Name, node.Resource.Properties["port"], node.Resource.Properties["runtime_image"])
			folderPath := node.Resource.Name
			if node.Directory != "" {
				folderPath = filepath.Join(node.Directory, node.Resource.Name)
			}
			log.Printf("Creating folder for GoWebserver: %s", folderPath)
			if err := os.MkdirAll(folderPath, 0755); err != nil {
				return fmt.Errorf("failed to create folder %s: %w", folderPath, err)
			}
			return nil
		},
	)
	// Add a rule for creating Filesystem::File
	re.RegisterRule(
		RuleCondition{Operation: OperationCreate, ResourceType: "Filesystem::File"},
		func(node *Node) error {
			filePath := node.Resource.Name
			if node.Directory != "" {
				filePath = filepath.Join(node.Directory, filePath)
			}

			log.Printf("Creating file: %s", filePath)

			// Create the directory if it doesn't exist
			dir := filepath.Dir(filePath)
			if dir != "." && dir != "/" {
				if err := os.MkdirAll(dir, 0755); err != nil {
					return fmt.Errorf("failed to create directory %s: %w", dir, err)
				}
			}

			// Create an empty file
			file, err := os.Create(filePath)
			if err != nil {
				return fmt.Errorf("failed to create file %s: %w", filePath, err)
			}
			file.Close()

			return nil
		},
	)
}

// RegisterDefaultExecuteRules registers the default rules for EXECUTE operations.
func (re *RuleEngine) RegisterDefaultExecuteRules() {
	re.RegisterRule(
		RuleCondition{Operation: OperationExecute, ResourceType: ""}, // Apply to any resource type for EXECUTE
		func(node *Node) error {
			if node.Command == "" {
				return fmt.Errorf("EXECUTE operation requires a command")
			}
			log.Printf("Executing command: %s in directory: %s", node.Command, node.Directory)

			// Special handling for 'go mod init'
			if strings.HasPrefix(node.Command, "go mod init") {
				goModPath := "go.mod"
				if node.Directory != "" {
					goModPath = filepath.Join(node.Directory, "go.mod")
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
		},
	)
}

// RegisterDefaultWriteFileRules registers the default rules for WRITE_FILE operations.
func (re *RuleEngine) RegisterDefaultWriteFileRules() {
	re.RegisterRule(
		RuleCondition{Operation: OperationWriteFile, ResourceType: ""}, // Apply to any resource type for WRITE_FILE
		func(node *Node) error {
			if node.FilePath == "" {
				return fmt.Errorf("WRITE_FILE operation requires a file path")
			}

			filePath := node.FilePath
			if node.Directory != "" {
				filePath = filepath.Join(node.Directory, filePath)
			}

			log.Printf("Writing file: %s", filePath)
			// Write content to the file, creating it if it doesn't exist, and overwriting it if it does.
			if err := os.WriteFile(filePath, []byte(node.Content), 0644); err != nil {
				return fmt.Errorf("failed to write file %s: %w", filePath, err)
			}
			return nil
		},
	)
}

// RegisterDefaultStartRules registers the default rules for START operations.
func (re *RuleEngine) RegisterDefaultStartRules() {
	re.RegisterRule(
		RuleCondition{Operation: OperationStart, ResourceType: ""}, // Apply to any resource type for START
		func(node *Node) error {
			if node.Command == "" {
				return fmt.Errorf("START operation requires a command")
			}
			log.Printf("Starting process: %s (in background) in directory: %s", node.Command, node.Directory)

			cmd := exec.Command("bash", " -c", node.Command)
			if node.Directory != "" {
				cmd.Dir = node.Directory
			}
			// Start the command without waiting for it to finish
			err := cmd.Start()
			if err != nil {
				return fmt.Errorf("failed to start command: %w", err)
			}
			log.Printf("Process started with PID: %d", cmd.Process.Pid)
			return nil
		},
	)
}

// RegisterDefaultDeleteRules registers the default rules for DELETE operations.
func (re *RuleEngine) RegisterDefaultDeleteRules() {
	re.RegisterRule(
		RuleCondition{Operation: OperationDelete, ResourceType: ""}, // Apply to any resource type for DELETE
		func(node *Node) error {
			log.Printf("Handling DELETE operation for resource: Type=%s, Name=%s",
				node.Resource.Type, node.Resource.Name)
			// Placeholder for actual resource deletion logic
			log.Printf("Simulating deletion of resource: %s %s", node.Resource.Type, node.Resource.Name)
			return nil
		},
	)
}