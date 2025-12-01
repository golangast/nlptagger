package agent

import (
	"time"
)

// GoalStatus represents the current state of a goal
type GoalStatus string

const (
	GoalNotStarted GoalStatus = "not_started"
	GoalInProgress GoalStatus = "in_progress"
	GoalBlocked    GoalStatus = "blocked"
	GoalCompleted  GoalStatus = "completed"
	GoalFailed     GoalStatus = "failed"
)

// MilestoneStatus represents the current state of a milestone
type MilestoneStatus string

const (
	MilestoneNotStarted MilestoneStatus = "not_started"
	MilestoneInProgress MilestoneStatus = "in_progress"
	MilestoneCompleted  MilestoneStatus = "completed"
	MilestoneFailed     MilestoneStatus = "failed"
)

// Goal represents a high-level objective for the agent to work toward
type Goal struct {
	ID                 string                 `json:"id"`
	Description        string                 `json:"description"`
	AcceptanceCriteria []string               `json:"acceptance_criteria,omitempty"`
	Status             GoalStatus             `json:"status"`
	Progress           float64                `json:"progress"` // 0.0 to 1.0
	Milestones         []*Milestone           `json:"milestones"`
	CreatedAt          time.Time              `json:"created_at"`
	UpdatedAt          time.Time              `json:"updated_at"`
	CompletedAt        *time.Time             `json:"completed_at,omitempty"`
	BlockReason        string                 `json:"block_reason,omitempty"`
	Metadata           map[string]interface{} `json:"metadata,omitempty"`
}

// Milestone represents a major checkpoint within a goal
type Milestone struct {
	ID           string                 `json:"id"`
	GoalID       string                 `json:"goal_id"`
	Description  string                 `json:"description"`
	Tasks        []*Subtask             `json:"tasks"`
	Status       MilestoneStatus        `json:"status"`
	Progress     float64                `json:"progress"`               // 0.0 to 1.0
	Dependencies []string               `json:"dependencies,omitempty"` // IDs of prerequisite milestones
	CreatedAt    time.Time              `json:"created_at"`
	UpdatedAt    time.Time              `json:"updated_at"`
	CompletedAt  *time.Time             `json:"completed_at,omitempty"`
	Metadata     map[string]interface{} `json:"metadata,omitempty"`
}

// IsComplete checks if the goal is completed
func (g *Goal) IsComplete() bool {
	return g.Status == GoalCompleted
}

// IsBlocked checks if the goal is blocked
func (g *Goal) IsBlocked() bool {
	return g.Status == GoalBlocked
}

// UpdateProgress recalculates goal progress based on milestone completion
func (g *Goal) UpdateProgress() {
	if len(g.Milestones) == 0 {
		g.Progress = 0.0
		return
	}

	totalProgress := 0.0
	for _, milestone := range g.Milestones {
		totalProgress += milestone.Progress
	}

	g.Progress = totalProgress / float64(len(g.Milestones))
	g.UpdatedAt = time.Now()

	// Update status based on progress
	if g.Progress >= 1.0 {
		g.Status = GoalCompleted
		now := time.Now()
		g.CompletedAt = &now
	} else if g.Progress > 0.0 {
		g.Status = GoalInProgress
	}
}

// IsReady checks if all dependencies for this milestone are satisfied
func (m *Milestone) IsReady(goal *Goal) bool {
	if m.Status != MilestoneNotStarted {
		return false
	}

	// Check all dependencies
	for _, depID := range m.Dependencies {
		for _, milestone := range goal.Milestones {
			if milestone.ID == depID && milestone.Status != MilestoneCompleted {
				return false
			}
		}
	}

	return true
}

// UpdateProgress recalculates milestone progress based on task completion
func (m *Milestone) UpdateProgress() {
	if len(m.Tasks) == 0 {
		m.Progress = 0.0
		return
	}

	completed := 0
	for _, task := range m.Tasks {
		if task.Status == StatusCompleted {
			completed++
		}
	}

	m.Progress = float64(completed) / float64(len(m.Tasks))
	m.UpdatedAt = time.Now()

	// Update status based on progress
	if m.Progress >= 1.0 {
		m.Status = MilestoneCompleted
		now := time.Now()
		m.CompletedAt = &now
	} else if m.Progress > 0.0 {
		m.Status = MilestoneInProgress
	}
}

// GetNextTask returns the next ready-to-execute task in this milestone
func (m *Milestone) GetNextTask() *Subtask {
	for _, task := range m.Tasks {
		if task.Status == StatusPending {
			// Check if all dependencies are met (using simplified check for now)
			allDepsMet := true
			for _, depID := range task.DependsOn {
				for _, t := range m.Tasks {
					if t.ID == depID && t.Status != StatusCompleted {
						allDepsMet = false
						break
					}
				}
			}
			if allDepsMet {
				return task
			}
		}
	}
	return nil
}
