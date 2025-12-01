package agent

import (
	"fmt"
	"strings"
	"time"
)

// GoalEngine handles goal decomposition and management
type GoalEngine struct {
	idCounter int
}

// NewGoalEngine creates a new goal engine
func NewGoalEngine() *GoalEngine {
	return &GoalEngine{
		idCounter: 0,
	}
}

// CreateGoal creates a new goal from a description
func (ge *GoalEngine) CreateGoal(description string) *Goal {
	return &Goal{
		ID:          ge.generateGoalID(),
		Description: description,
		Status:      GoalNotStarted,
		Progress:    0.0,
		Milestones:  make([]*Milestone, 0),
		CreatedAt:   time.Now(),
		UpdatedAt:   time.Now(),
		Metadata:    make(map[string]interface{}),
	}
}

// DecomposeGoal breaks down a high-level goal into milestones and tasks
func (ge *GoalEngine) DecomposeGoal(goal *Goal) error {
	lower := strings.ToLower(goal.Description)

	milestones := make([]*Milestone, 0)

	// Use template registry to infer project type
	templateReg := NewTemplateRegistry()
	templateName := templateReg.InferTemplate(goal.Description)

	// Extract project name from description
	projectName := ge.extractProjectName(goal.Description)
	if projectName == "" {
		projectName = "project"
	}

	// Pattern: Web/API projects
	if strings.Contains(lower, "rest") || strings.Contains(lower, "api") ||
		strings.Contains(lower, "webserver") || strings.Contains(lower, "web server") ||
		strings.Contains(lower, "http") {

		// Milestone 1: Project Setup
		setupTasks := []*Subtask{
			{
				ID:          ge.generateTaskID(),
				Description: fmt.Sprintf("Create project folder '%s'", projectName),
				Status:      StatusPending,
				Tool:        "create_folder",
				Args:        map[string]interface{}{"name": projectName},
			},
			{
				ID:          ge.generateTaskID(),
				Description: fmt.Sprintf("Generate code from '%s' template", templateName),
				Status:      StatusPending,
				Tool:        "apply_template",
				Args: map[string]interface{}{
					"template": templateName,
					"folder":   projectName,
				},
			},
		}
		milestones = append(milestones, ge.createMilestone(goal.ID, "Project Setup", setupTasks, nil))

		// Milestone 2: Documentation
		docTasks := []*Subtask{
			{
				ID:          ge.generateTaskID(),
				Description: "Verify README exists",
				Status:      StatusPending,
				Tool:        "verify_files",
				Args: map[string]interface{}{
					"folder": projectName,
					"files":  []string{"README.md", "main.go"},
				},
			},
		}
		milestones = append(milestones, ge.createMilestone(goal.ID, "Documentation", docTasks, []string{milestones[0].ID}))

	} else if strings.Contains(lower, "handler") {
		// Just create handlers file
		setupTasks := []*Subtask{
			{
				ID:          ge.generateTaskID(),
				Description: "Generate handlers code",
				Status:      StatusPending,
				Tool:        "apply_template",
				Args: map[string]interface{}{
					"template": "handlers",
					"folder":   projectName,
				},
			},
		}
		milestones = append(milestones, ge.createMilestone(goal.ID, "Generate Handlers", setupTasks, nil))
	} else if strings.Contains(lower, "delete") || strings.Contains(lower, "remove") {
		// Just delete the folder
		deleteTasks := []*Subtask{
			{
				ID:          ge.generateTaskID(),
				Description: fmt.Sprintf("Delete project folder '%s'", projectName),
				Status:      StatusPending,
				Tool:        "delete_folder",
				Args:        map[string]interface{}{"name": projectName},
			},
		}
		milestones = append(milestones, ge.createMilestone(goal.ID, "Project Deletion", deleteTasks, nil))
	} else {
		// Generic project pattern
		setupTasks := []*Subtask{
			{
				ID:          ge.generateTaskID(),
				Description: fmt.Sprintf("Create project folder '%s'", projectName),
				Status:      StatusPending,
				Tool:        "create_folder",
				Args:        map[string]interface{}{"name": projectName},
			},
			{
				ID:          ge.generateTaskID(),
				Description: fmt.Sprintf("Generate code from '%s' template", templateName),
				Status:      StatusPending,
				Tool:        "apply_template",
				Args: map[string]interface{}{
					"template": templateName,
					"folder":   projectName,
				},
			},
		}
		milestones = append(milestones, ge.createMilestone(goal.ID, "Project Setup", setupTasks, nil))
	}

	goal.Milestones = milestones
	goal.Status = GoalNotStarted
	goal.UpdatedAt = time.Now()

	return nil
}

// extractProjectName attempts to extract a project name from the goal description
func (ge *GoalEngine) extractProjectName(description string) string {
	words := strings.Fields(strings.ToLower(description))

	// Look for "called X", "named X", "for X"
	for i, word := range words {
		if (word == "called" || word == "named" || word == "for") && i+1 < len(words) {
			name := words[i+1]
			// Clean up name (remove articles, punctuation)
			name = strings.Trim(name, ".,!?;:")
			if name != "a" && name != "an" && name != "the" {
				return name
			}
		}
	}

	// Look for specific types
	if strings.Contains(description, "todo") {
		return "todo"
	}
	if strings.Contains(description, "blog") {
		return "blog"
	}
	if strings.Contains(description, "chat") {
		return "chat"
	}

	return "project"
}

// GetNextAction determines the next task to execute for a goal
func (ge *GoalEngine) GetNextAction(goal *Goal) *Subtask {
	// Find the first milestone that's ready and not complete
	for _, milestone := range goal.Milestones {
		if milestone.IsReady(goal) {
			// Mark as in progress
			if milestone.Status == MilestoneNotStarted {
				milestone.Status = MilestoneInProgress
			}

			// Get next task from this milestone
			task := milestone.GetNextTask()
			if task != nil {
				return task
			}
		} else if milestone.Status == MilestoneInProgress {
			// Continue working on current milestone
			task := milestone.GetNextTask()
			if task != nil {
				return task
			}
		}
	}

	return nil
}

// EvaluateProgress updates goal progress based on milestone completion
func (ge *GoalEngine) EvaluateProgress(goal *Goal) float64 {
	// Update all milestone progress
	for _, milestone := range goal.Milestones {
		milestone.UpdateProgress()
	}

	// Update goal progress
	goal.UpdateProgress()

	return goal.Progress
}

// Helper: Create milestone with tasks
func (ge *GoalEngine) createMilestone(goalID, description string, tasks []*Subtask, dependencies []string) *Milestone {
	milestone := &Milestone{
		ID:           ge.generateMilestoneID(),
		GoalID:       goalID,
		Description:  description,
		Tasks:        tasks,
		Status:       MilestoneNotStarted,
		Progress:     0.0,
		Dependencies: dependencies,
		CreatedAt:    time.Now(),
		UpdatedAt:    time.Now(),
		Metadata:     make(map[string]interface{}),
	}
	return milestone
}

// Infer which tool to use based on task description
func (ge *GoalEngine) inferToolFromDescription(description string) string {
	lower := strings.ToLower(description)

	if strings.Contains(lower, "create project") || strings.Contains(lower, "project structure") {
		return "create_folder"
	}
	if strings.Contains(lower, "initialize") || strings.Contains(lower, "go.mod") {
		return "execute_command"
	}
	if strings.Contains(lower, "add") || strings.Contains(lower, "install") {
		return "execute_command"
	}
	if strings.Contains(lower, "implement") || strings.Contains(lower, "create") {
		return "create_file"
	}

	return "execute_command"
}

// Infer arguments from task description
func (ge *GoalEngine) inferArgsFromDescription(description string) map[string]interface{} {
	args := make(map[string]interface{})
	lower := strings.ToLower(description)

	if strings.Contains(lower, "go.mod") {
		args["command"] = "go mod init"
	} else if strings.Contains(lower, "project structure") {
		args["name"] = "project"
	}

	// Will be expanded with more sophisticated parsing
	return args
}

func (ge *GoalEngine) generateGoalID() string {
	ge.idCounter++
	return fmt.Sprintf("goal_%d", ge.idCounter)
}

func (ge *GoalEngine) generateMilestoneID() string {
	ge.idCounter++
	return fmt.Sprintf("milestone_%d", ge.idCounter)
}

func (ge *GoalEngine) generateTaskID() string {
	ge.idCounter++
	return fmt.Sprintf("task_%d", ge.idCounter)
}
