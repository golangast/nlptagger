package agent

import (
	"fmt"
	"strings"

	"github.com/zendrulat/nlptagger/neural/semantic"
)

// AgentConfig holds configuration for the agent
type AgentConfig struct {
	MaxIterations     int
	EnablePersistence bool
	PersistencePath   string
	WorkingDirectory  string // Base directory for generated projects
}

// Agent orchestrates the four pillars of agent scaffolding
type Agent struct {
	// Four Pillars
	PlanningEngine *PlanningEngine
	Memory         MemoryStore
	ToolRegistry   *ToolRegistry
	FeedbackLoop   *FeedbackLoop

	// Goal-Driven Development
	GoalEngine     *GoalEngine
	SessionManager *SessionManager
	GoalStore      GoalStore

	// Existing NLP components (for compatibility)
	VFS             *semantic.VFSTree
	BlueprintEngine *semantic.BlueprintEngine
	DependencyGraph *semantic.DependencyGraph
	RoleRegistry    *semantic.RoleRegistry

	// Config
	Config AgentConfig
}

// NewAgent creates a new agent with all four pillars initialized
func NewAgent(config AgentConfig) *Agent {
	// Set default working directory
	if config.WorkingDirectory == "" {
		config.WorkingDirectory = "./generated_projects"
	}

	agent := &Agent{
		PlanningEngine: NewPlanningEngine(),
		ToolRegistry:   NewToolRegistry(),
		FeedbackLoop:   NewFeedbackLoop(config.MaxIterations),
		Config:         config,
	}

	// Initialize memory
	if config.EnablePersistence && config.PersistencePath != "" {
		agent.Memory = NewPersistentStore(config.PersistencePath)
	} else {
		agent.Memory = NewInMemoryStore()
	}

	// Initialize goal-driven components
	agent.GoalEngine = NewGoalEngine()
	agent.GoalStore = NewFileBasedStore("./goal_data")
	agent.SessionManager = NewSessionManager(agent.GoalStore)

	// Initialize VFS and other components
	agent.VFS = semantic.NewVFSTree()
	agent.BlueprintEngine = semantic.NewBlueprintEngine()
	agent.DependencyGraph = semantic.NewDependencyGraph()
	agent.RoleRegistry = semantic.NewRoleRegistry()

	// Register built-in tools
	agent.registerBuiltinTools()

	return agent
}

// registerBuiltinTools adds standard tools to the registry
func (a *Agent) registerBuiltinTools() {
	// Register VFS tools (for tracking)
	a.ToolRegistry.RegisterTool(NewCreateFolderTool(a.VFS, a.RoleRegistry))

	// Register REAL filesystem tools (for actual code generation)
	a.ToolRegistry.RegisterTool(NewRealApplyTemplateTool(a.VFS, a.RoleRegistry, a.Config.WorkingDirectory))
	a.ToolRegistry.RegisterTool(NewRealExecuteCommandTool(a.VFS, a.Config.WorkingDirectory))
	a.ToolRegistry.RegisterTool(NewRealFilesystemTool(a.VFS, a.RoleRegistry, a.Config.WorkingDirectory))

	// Add verify_files stub for now
	a.ToolRegistry.RegisterTool(NewVerifyFilesTool(a.Config.WorkingDirectory))
	a.ToolRegistry.RegisterTool(NewRealDeleteFolderTool(a.VFS, a.Config.WorkingDirectory))
}

// ProcessGoal is the main agentic loop
func (a *Agent) ProcessGoal(userGoal string) error {
	// Store user message
	a.Memory.AppendMessage("user", userGoal)

	// 1. Planning Phase
	plan, err := a.PlanningEngine.DecomposeGoal(userGoal)
	if err != nil {
		return fmt.Errorf("planning failed: %w", err)
	}

	// Log plan creation
	a.Memory.LogAction(Action{
		Type:    "plan_created",
		Data:    plan,
		Success: true,
	})

	// Reset feedback loop for new goal
	a.FeedbackLoop.Reset()

	// 2. Execution Loop
	for {
		// Get next ready subtask
		subtask := plan.GetNextSubtask()
		if subtask == nil {
			// No more ready tasks
			if plan.IsComplete() {
				break // Success!
			} else if plan.HasErrors() {
				return fmt.Errorf("plan has errors, cannot continue")
			} else {
				return fmt.Errorf("no ready subtasks but plan incomplete (circular dependency?)")
			}
		}

		// Mark as running
		subtask.Status = StatusRunning

		// 3. Tool Execution
		tool, err := a.ToolRegistry.GetTool(subtask.Tool)
		if err != nil {
			subtask.Status = StatusFailed
			subtask.Error = err.Error()
			continue
		}

		result, err := tool.Execute(subtask.Args)
		if err != nil {
			subtask.Status = StatusFailed
			subtask.Error = err.Error()
			continue
		}

		// Update subtask with result
		if result.Success {
			subtask.Status = StatusCompleted
			subtask.Result = result.Output
		} else {
			subtask.Status = StatusFailed
			subtask.Error = result.Error.Error()
		}

		// Log execution
		errMsg := ""
		if result.Error != nil {
			errMsg = result.Error.Error()
		}

		// Log execution
		a.Memory.LogAction(Action{
			Type:    "tool_executed",
			Data:    subtask,
			Success: result.Success,
			Error:   errMsg,
		})

		// 4. Create observation
		obs := Observation{
			ToolName:   subtask.Tool,
			ToolResult: result,
			SubtaskID:  subtask.ID,
		}

		// 5. Feedback & Control
		decision, err := a.FeedbackLoop.Evaluate(obs, plan)
		if err != nil {
			return fmt.Errorf("feedback evaluation failed: %w", err)
		}

		// Store critique
		a.Memory.AppendMessage("assistant", decision.Critique)

		// Handle decision
		if !decision.ShouldContinue {
			if decision.ErrorMessage != "" {
				return fmt.Errorf("execution stopped: %s", decision.ErrorMessage)
			}
			break // Goal achieved or max iterations
		}

		if decision.ShouldReplan {
			err = a.PlanningEngine.UpdatePlan(plan, decision.Critique)
			if err != nil {
				return fmt.Errorf("replanning failed: %w", err)
			}
		}

		if decision.ShouldRetry {
			// Reset subtask to pending
			subtask.Status = StatusPending
		}
	}

	// Final summary
	summary := fmt.Sprintf("Goal completed! Executed %d subtasks successfully.", len(plan.Subtasks))
	a.Memory.AppendMessage("assistant", summary)

	return nil
}

// GetConversationSummary returns recent conversation history
func (a *Agent) GetConversationSummary(limit int) string {
	messages, _ := a.Memory.GetConversationHistory(limit)

	summary := "Recent Conversation:\n"
	for _, msg := range messages {
		summary += fmt.Sprintf("[%s] %s\n", msg.Role, msg.Content)
	}

	return summary
}

// GetActionSummary returns recent action history
func (a *Agent) GetActionSummary(limit int) string {
	actions, _ := a.Memory.GetActionHistory(ActionFilters{Limit: limit})

	summary := "Recent Actions:\n"
	for _, action := range actions {
		status := "✓"
		if !action.Success {
			status = "✗"
		}
		summary += fmt.Sprintf("%s [%s] %s\n", status, action.Type, action.Timestamp.Format("15:04:05"))
	}

	return summary
}

// ===== Goal-Driven Development Methods =====

// CreateGoalFromDescription creates and decomposes a new goal
func (a *Agent) CreateGoalFromDescription(description string) (*Goal, error) {
	goal := a.GoalEngine.CreateGoal(description)

	err := a.GoalEngine.DecomposeGoal(goal)
	if err != nil {
		return nil, fmt.Errorf("goal decomposition failed: %w", err)
	}

	// Save goal
	err = a.GoalStore.SaveGoal(goal)
	if err != nil {
		return nil, fmt.Errorf("failed to save goal: %w", err)
	}

	return goal, nil
}

// GetGoal retrieves a goal by ID
func (a *Agent) GetGoal(goalID string) (*Goal, error) {
	return a.GoalStore.LoadGoal(goalID)
}

// ListGoals returns all goals
func (a *Agent) ListGoals() ([]*Goal, error) {
	return a.GoalStore.ListGoals()
}

// WorkOnGoalStep executes one step toward a goal
func (a *Agent) WorkOnGoalStep(goal *Goal) (*Subtask, error) {
	// Get next action
	task := a.GoalEngine.GetNextAction(goal)
	if task == nil {
		return nil, nil // No more tasks
	}

	// Mark task as running
	task.Status = StatusRunning

	//Execute task using existing tool system
	tool, err := a.ToolRegistry.GetTool(task.Tool)
	if err != nil {
		task.Status = StatusFailed
		task.Error = err.Error()
		a.GoalStore.SaveGoal(goal)
		return task, err
	}

	result, err := tool.Execute(task.Args)
	if err != nil || !result.Success {
		task.Status = StatusFailed
		if err != nil {
			task.Error = err.Error()
		} else if result.Error != nil {
			task.Error = result.Error.Error()
		}
	} else {
		task.Status = StatusCompleted
		task.Result = result.Output
	}

	// Update progress
	a.GoalEngine.EvaluateProgress(goal)

	// Save updated goal
	a.GoalStore.SaveGoal(goal)

	return task, nil
}

// GetGoalStatus returns a formatted status string for a goal
func (a *Agent) GetGoalStatus(goal *Goal) string {
	var status strings.Builder

	status.WriteString(fmt.Sprintf("🎯 Goal: %s\n", goal.Description))
	status.WriteString(fmt.Sprintf("Status: %s (%.0f%% complete)\n", goal.Status, goal.Progress*100))
	status.WriteString(fmt.Sprintf("Created: %s\n\n", goal.CreatedAt.Format("2006-01-02 15:04")))

	status.WriteString("Milestones:\n")
	for i, milestone := range goal.Milestones {
		statusIcon := "⏳"
		switch milestone.Status {
		case MilestoneCompleted:
			statusIcon = "✅"
		case MilestoneInProgress:
			statusIcon = "🔄"
		case MilestoneFailed:
			statusIcon = "❌"
		}

		completedTasks := 0
		for _, task := range milestone.Tasks {
			if task.Status == StatusCompleted {
				completedTasks++
			}
		}

		status.WriteString(fmt.Sprintf("  %d. %s %s (%d/%d tasks)\n",
			i+1, statusIcon, milestone.Description, completedTasks, len(milestone.Tasks)))
	}

	return status.String()
}
