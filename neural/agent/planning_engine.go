package agent

import (
	"fmt"
	"strings"
	"time"
)

// SubtaskStatus represents the execution state of a subtask
type SubtaskStatus string

const (
	StatusPending   SubtaskStatus = "pending"
	StatusRunning   SubtaskStatus = "running"
	StatusCompleted SubtaskStatus = "completed"
	StatusFailed    SubtaskStatus = "failed"
	StatusSkipped   SubtaskStatus = "skipped"
)

// Subtask represents an individual executable step in a plan
type Subtask struct {
	ID          string                 `json:"id"`
	Description string                 `json:"description"`
	Status      SubtaskStatus          `json:"status"`
	Tool        string                 `json:"tool"`       // Tool name to execute
	Args        map[string]interface{} `json:"args"`       // Tool arguments
	DependsOn   []string               `json:"depends_on"` // IDs of prerequisite subtasks
	Result      interface{}            `json:"result,omitempty"`
	Error       string                 `json:"error,omitempty"`
	StartTime   time.Time              `json:"start_time,omitempty"`
	EndTime     time.Time              `json:"end_time,omitempty"`
}

// IsReady checks if all dependencies are satisfied
func (st *Subtask) IsReady(plan *Plan) bool {
	if st.Status != StatusPending {
		return false
	}

	for _, depID := range st.DependsOn {
		depTask := plan.GetSubtask(depID)
		if depTask == nil || depTask.Status != StatusCompleted {
			return false
		}
	}

	return true
}

// Plan represents a decomposed workflow with multiple subtasks
type Plan struct {
	ID        string                 `json:"id"`
	Goal      string                 `json:"goal"`
	Subtasks  []*Subtask             `json:"subtasks"`
	Reasoning string                 `json:"reasoning"` // Chain-of-Thought explanation
	CreatedAt time.Time              `json:"created_at"`
	UpdatedAt time.Time              `json:"updated_at"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// GetSubtask retrieves a subtask by ID
func (p *Plan) GetSubtask(id string) *Subtask {
	for _, task := range p.Subtasks {
		if task.ID == id {
			return task
		}
	}
	return nil
}

// GetNextSubtask returns the next ready-to-execute subtask
func (p *Plan) GetNextSubtask() *Subtask {
	for _, task := range p.Subtasks {
		if task.IsReady(p) {
			return task
		}
	}
	return nil
}

// IsComplete checks if all subtasks are completed
func (p *Plan) IsComplete() bool {
	for _, task := range p.Subtasks {
		if task.Status != StatusCompleted && task.Status != StatusSkipped {
			return false
		}
	}
	return true
}

// HasErrors checks if any subtask has failed
func (p *Plan) HasErrors() bool {
	for _, task := range p.Subtasks {
		if task.Status == StatusFailed {
			return true
		}
	}
	return false
}

// Progress returns completion percentage
func (p *Plan) Progress() float64 {
	if len(p.Subtasks) == 0 {
		return 1.0
	}

	completed := 0
	for _, task := range p.Subtasks {
		if task.Status == StatusCompleted || task.Status == StatusSkipped {
			completed++
		}
	}

	return float64(completed) / float64(len(p.Subtasks))
}

// PlanningEngine decomposes complex goals into executable subtasks
type PlanningEngine struct {
	idCounter int
}

// NewPlanningEngine creates a new planning engine
func NewPlanningEngine() *PlanningEngine {
	return &PlanningEngine{
		idCounter: 0,
	}
}

// DecomposeGoal analyzes a user request and creates a plan with subtasks
func (pe *PlanningEngine) DecomposeGoal(userRequest string) (*Plan, error) {
	plan := &Plan{
		ID:        pe.generatePlanID(),
		Goal:      userRequest,
		Subtasks:  make([]*Subtask, 0),
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
		Metadata:  make(map[string]interface{}),
	}

	// Simple rule-based decomposition for now
	// This would be replaced with LLM-based reasoning in production
	subtasks, reasoning := pe.ruleBasedDecomposition(userRequest)
	plan.Subtasks = subtasks
	plan.Reasoning = reasoning

	return plan, nil
}

// ruleBasedDecomposition performs simple rule-based task breakdown
func (pe *PlanningEngine) ruleBasedDecomposition(request string) ([]*Subtask, string) {
	lower := strings.ToLower(request)
	subtasks := make([]*Subtask, 0)
	reasoning := "Decomposing request based on detected patterns:\n"

	// Pattern: create project/app
	if strings.Contains(lower, "project") || strings.Contains(lower, "app") || strings.Contains(lower, "application") {
		reasoning += "- Detected project creation request\n"

		// Extract project type
		projectType := pe.detectProjectType(lower)
		reasoning += fmt.Sprintf("- Project type: %s\n", projectType)

		// Create folder
		subtasks = append(subtasks, &Subtask{
			ID:          pe.generateSubtaskID(),
			Description: "Create project root folder",
			Status:      StatusPending,
			Tool:        "create_folder",
			Args: map[string]interface{}{
				"name": pe.extractProjectName(request),
			},
			DependsOn: []string{},
		})

		// Apply template if detected
		if projectType != "generic" {
			subtasks = append(subtasks, &Subtask{
				ID:          pe.generateSubtaskID(),
				Description: fmt.Sprintf("Apply %s template", projectType),
				Status:      StatusPending,
				Tool:        "apply_template",
				Args: map[string]interface{}{
					"template": projectType,
					"folder":   pe.extractProjectName(request),
				},
				DependsOn: []string{subtasks[0].ID},
			})
		}

		// Add specific components if mentioned
		if strings.Contains(lower, "auth") {
			subtasks = append(subtasks, &Subtask{
				ID:          pe.generateSubtaskID(),
				Description: "Add authentication module",
				Status:      StatusPending,
				Tool:        "add_feature",
				Args: map[string]interface{}{
					"feature": "auth",
					"folder":  pe.extractProjectName(request),
				},
				DependsOn: []string{subtasks[len(subtasks)-1].ID},
			})
		}

		if strings.Contains(lower, "database") || strings.Contains(lower, "db") {
			subtasks = append(subtasks, &Subtask{
				ID:          pe.generateSubtaskID(),
				Description: "Add database integration",
				Status:      StatusPending,
				Tool:        "add_feature",
				Args: map[string]interface{}{
					"feature": "database",
					"folder":  pe.extractProjectName(request),
				},
				DependsOn: []string{subtasks[len(subtasks)-1].ID},
			})
		}
	} else if strings.Contains(lower, "folder") && (strings.Contains(lower, "file") || pe.extractFileName(request) != "") {
		// Pattern: create folder with files
		reasoning += "- Detected folder and file creation\n"

		folderName := pe.extractFolderName(request)
		fileName := pe.extractFileName(request)

		subtasks = append(subtasks, &Subtask{
			ID:          pe.generateSubtaskID(),
			Description: fmt.Sprintf("Create folder '%s'", folderName),
			Status:      StatusPending,
			Tool:        "create_folder",
			Args: map[string]interface{}{
				"name": folderName,
			},
			DependsOn: []string{},
		})

		if fileName != "" {
			subtasks = append(subtasks, &Subtask{
				ID:          pe.generateSubtaskID(),
				Description: fmt.Sprintf("Create file '%s' in '%s'", fileName, folderName),
				Status:      StatusPending,
				Tool:        "create_file",
				Args: map[string]interface{}{
					"name":   fileName,
					"folder": folderName,
				},
				DependsOn: []string{subtasks[0].ID},
			})
		}
	} else {
		// Fallback: single-step execution
		reasoning += "- Simple single-step request\n"
		subtasks = append(subtasks, &Subtask{
			ID:          pe.generateSubtaskID(),
			Description: request,
			Status:      StatusPending,
			Tool:        "execute_command",
			Args: map[string]interface{}{
				"command": request,
			},
			DependsOn: []string{},
		})
	}

	return subtasks, reasoning
}

// Helper functions for pattern extraction
func (pe *PlanningEngine) detectProjectType(request string) string {
	patterns := map[string][]string{
		"webserver":    {"webserver", "web server", "http server", "api server"},
		"rest-api":     {"rest api", "rest", "restful"},
		"cli":          {"cli", "command line", "command-line"},
		"microservice": {"microservice", "micro service"},
	}

	for projectType, keywords := range patterns {
		for _, keyword := range keywords {
			if strings.Contains(request, keyword) {
				return projectType
			}
		}
	}

	return "generic"
}

func (pe *PlanningEngine) extractProjectName(request string) string {
	words := strings.Fields(request)
	for i, word := range words {
		if (word == "project" || word == "app" || word == "application") && i+1 < len(words) {
			return words[i+1]
		}
	}
	return "myproject"
}

func (pe *PlanningEngine) extractFolderName(request string) string {
	words := strings.Fields(request)
	for i, word := range words {
		if (word == "folder" || word == "directory") && i+1 < len(words) {
			return words[i+1]
		}
	}
	return "newfolder"
}

func (pe *PlanningEngine) extractFileName(request string) string {
	words := strings.Fields(request)
	for _, word := range words {
		if strings.Contains(word, ".") {
			return word
		}
	}
	return ""
}

func (pe *PlanningEngine) generatePlanID() string {
	pe.idCounter++
	return fmt.Sprintf("plan_%d", pe.idCounter)
}

func (pe *PlanningEngine) generateSubtaskID() string {
	pe.idCounter++
	return fmt.Sprintf("task_%d", pe.idCounter)
}

// UpdatePlan adjusts the plan based on feedback
func (pe *PlanningEngine) UpdatePlan(plan *Plan, feedback string) error {
	plan.UpdatedAt = time.Now()

	// Simple replan logic: if a task failed, mark remaining tasks as pending
	if strings.Contains(strings.ToLower(feedback), "error") ||
		strings.Contains(strings.ToLower(feedback), "fail") {
		for _, task := range plan.Subtasks {
			if task.Status == StatusFailed {
				// Mark dependent tasks as pending for retry
				for _, depTask := range plan.Subtasks {
					for _, depID := range depTask.DependsOn {
						if depID == task.ID {
							depTask.Status = StatusPending
						}
					}
				}
			}
		}
	}

	return nil
}

// String returns a human-readable representation of the plan
func (p *Plan) String() string {
	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("Plan: %s\n", p.Goal))
	sb.WriteString(fmt.Sprintf("  Progress: %.0f%%\n", p.Progress()*100))
	sb.WriteString("  Subtasks:\n")

	for i, task := range p.Subtasks {
		status := "⬜"
		switch task.Status {
		case StatusCompleted:
			status = "✅"
		case StatusRunning:
			status = "🔄"
		case StatusFailed:
			status = "❌"
		case StatusSkipped:
			status = "⏭️"
		}

		sb.WriteString(fmt.Sprintf("    %d. %s %s\n", i+1, status, task.Description))
	}

	return sb.String()
}
