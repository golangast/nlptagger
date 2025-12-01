package agent

import (
	"fmt"
	"strings"
	"time"
)

// FeedbackDecision represents the outcome of evaluating a tool execution
type FeedbackDecision struct {
	ShouldContinue bool   `json:"should_continue"` // Continue to next subtask
	ShouldRetry    bool   `json:"should_retry"`    // Retry current subtask
	ShouldReplan   bool   `json:"should_replan"`   // Replan remaining tasks
	ErrorMessage   string `json:"error_message,omitempty"`
	Critique       string `json:"critique,omitempty"` // Self-assessment
}

// Observation captures the result of a tool execution
type Observation struct {
	ToolName   string
	ToolResult ToolResult
	Timestamp  time.Time
	SubtaskID  string
}

// PolicyValidator enforces safety and operational constraints
type PolicyValidator interface {
	ValidatePath(path string) error
	ValidateOperation(operation string) error
}

// FeedbackLoop evaluates tool outputs and controls agent behavior
type FeedbackLoop struct {
	maxIterations    int
	currentIteration int
	policy           PolicyValidator
	observations     []Observation
}

// NewFeedbackLoop creates a new feedback loop
func NewFeedbackLoop(maxIterations int) *FeedbackLoop {
	if maxIterations <= 0 {
		maxIterations = 10 // Default safety limit
	}

	return &FeedbackLoop{
		maxIterations:    maxIterations,
		currentIteration: 0,
		policy:           NewDefaultPolicy(),
		observations:     make([]Observation, 0),
	}
}

// Evaluate analyzes a tool execution result and decides next action
func (fl *FeedbackLoop) Evaluate(obs Observation, plan *Plan) (FeedbackDecision, error) {
	fl.currentIteration++
	fl.observations = append(fl.observations, obs)

	decision := FeedbackDecision{
		ShouldContinue: true,
		ShouldRetry:    false,
		ShouldReplan:   false,
	}

	// Check iteration limit
	if fl.currentIteration >= fl.maxIterations {
		decision.ShouldContinue = false
		decision.ErrorMessage = fmt.Sprintf("max iterations (%d) reached", fl.maxIterations)
		decision.Critique = "Agent may be stuck in a loop. Consider breaking down the goal differently."
		return decision, nil
	}

	// Evaluate tool result
	if !obs.ToolResult.Success {
		decision.Critique = fl.generateCritique(obs, plan)

		// Decide whether to retry or replan
		if fl.shouldRetry(obs) {
			decision.ShouldRetry = true
			decision.ShouldContinue = false
		} else {
			decision.ShouldReplan = true
			decision.ShouldContinue = false
		}

		decision.ErrorMessage = fmt.Sprintf("Tool '%s' failed: %v", obs.ToolName, obs.ToolResult.Error)
		return decision, nil
	}

	// Check if plan is complete
	if plan.IsComplete() {
		decision.ShouldContinue = false
		decision.Critique = "All subtasks completed successfully. Goal achieved!"
		return decision, nil
	}

	// All good, continue
	decision.Critique = fmt.Sprintf("Tool '%s' succeeded. Proceeding to next subtask.", obs.ToolName)
	return decision, nil
}

// Reset clears the iteration counter and observations
func (fl *FeedbackLoop) Reset() {
	fl.currentIteration = 0
	fl.observations = make([]Observation, 0)
}

// generateCritique creates a self-assessment of what went wrong
func (fl *FeedbackLoop) generateCritique(obs Observation, plan *Plan) string {
	var critique strings.Builder

	critique.WriteString(fmt.Sprintf("Tool '%s' failed. ", obs.ToolName))

	if obs.ToolResult.Error != nil {
		critique.WriteString(fmt.Sprintf("Error: %v. ", obs.ToolResult.Error))
	}

	// Analyze error patterns
	errorMsg := ""
	if obs.ToolResult.Error != nil {
		errorMsg = obs.ToolResult.Error.Error()
	}

	if strings.Contains(errorMsg, "not found") {
		critique.WriteString("The target resource doesn't exist. ")
		critique.WriteString("Consider creating required dependencies first. ")
	} else if strings.Contains(errorMsg, "already exists") {
		critique.WriteString("Resource already exists. ")
		critique.WriteString("Consider skipping this step or using a different name. ")
	} else if strings.Contains(errorMsg, "permission") {
		critique.WriteString("Permission denied. ")
		critique.WriteString("Verify access rights or policy constraints. ")
	} else {
		critique.WriteString("Unexpected error. ")
	}

	return critique.String()
}

// shouldRetry determines if a failed subtask should be retried
func (fl *FeedbackLoop) shouldRetry(obs Observation) bool {
	// Simple heuristic: retry transient-looking errors
	if obs.ToolResult.Error == nil {
		return false
	}

	errorMsg := obs.ToolResult.Error.Error()

	// Don't retry these
	permanentErrors := []string{
		"not found",
		"already exists",
		"invalid",
		"forbidden",
	}

	for _, perm := range permanentErrors {
		if strings.Contains(strings.ToLower(errorMsg), perm) {
			return false
		}
	}

	// Could retry
	return true
}

// GetObservations returns all recorded observations
func (fl *FeedbackLoop) GetObservations() []Observation {
	return fl.observations
}

// DefaultPolicy implements basic safety constraints
type DefaultPolicy struct{}

// NewDefaultPolicy creates a default policy validator
func NewDefaultPolicy() *DefaultPolicy {
	return &DefaultPolicy{}
}

// ValidatePath ensures paths don't escape project boundaries
func (dp *DefaultPolicy) ValidatePath(path string) error {
	// Prevent directory traversal
	if strings.Contains(path, "..") {
		return fmt.Errorf("path contains directory traversal: %s", path)
	}

	// Prevent absolute paths outside project
	if strings.HasPrefix(path, "/etc") ||
		strings.HasPrefix(path, "/sys") ||
		strings.HasPrefix(path, "/proc") ||
		strings.HasPrefix(path, "/dev") {
		return fmt.Errorf("path accesses system directories: %s", path)
	}

	return nil
}

// ValidateOperation checks if an operation is allowed
func (dp *DefaultPolicy) ValidateOperation(operation string) error {
	// Blacklist dangerous operations
	dangerous := []string{
		"rm -rf /",
		"format",
		"mkfs",
	}

	lower := strings.ToLower(operation)
	for _, danger := range dangerous {
		if strings.Contains(lower, danger) {
			return fmt.Errorf("operation blocked by policy: %s", operation)
		}
	}

	return nil
}
