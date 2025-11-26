package workflow

import (
	"fmt"
	"log"

	"github.com/zendrulat/nlptagger/neural/semantic"
)

// Executor is responsible for executing the nodes in a workflow.
type Executor struct {
	ruleEngine *RuleEngine // Add the rule engine
}

// NewExecutor creates a new Executor.
func NewExecutor() *Executor {
	re := NewRuleEngine()
	re.RegisterDefaultCreateRules()
	re.RegisterDefaultExecuteRules()
	re.RegisterDefaultWriteFileRules()
	re.RegisterDefaultStartRules()
	re.RegisterDefaultDeleteRules()
	re.RegisterDefaultMoveRules()

	return &Executor{
		ruleEngine: re,
	}
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
	// Apply inference rules to fill in missing properties
	semantic.InferProperties(&node.Resource)

	// Validate the resource against defined policies
	if err := semantic.ValidateResource(&node.Resource, node.Context); err != nil {
		return fmt.Errorf("resource validation failed for node '%s': %w", node.ID, err)
	}

	log.Printf("Node Resource Type before rule execution: %s", node.Resource.Type)
	// Use the rule engine to execute the node's operation
	if err := e.ruleEngine.ExecuteRules(node); err != nil {
		return fmt.Errorf("failed to execute node operation via rule engine: %w", err)
	}

	return nil
}