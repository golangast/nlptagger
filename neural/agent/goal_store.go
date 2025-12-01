package agent

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// GoalStore interface for persisting goals and sessions
type GoalStore interface {
	SaveGoal(goal *Goal) error
	LoadGoal(id string) (*Goal, error)
	ListGoals() ([]*Goal, error)
	DeleteGoal(id string) error

	SaveSession(session *Session) error
	LoadSession(id string) (*Session, error)
	ListSessions() ([]*Session, error)
	DeleteSession(id string) error
}

// FileBasedStore implements GoalStore using JSON files
type FileBasedStore struct {
	BasePath string
}

// NewFileBasedStore creates a new file-based goal store
func NewFileBasedStore(basePath string) *FileBasedStore {
	// Ensure directories exist
	os.MkdirAll(filepath.Join(basePath, "goals"), 0755)
	os.MkdirAll(filepath.Join(basePath, "sessions"), 0755)

	return &FileBasedStore{
		BasePath: basePath,
	}
}

// SaveGoal saves a goal to disk
func (fs *FileBasedStore) SaveGoal(goal *Goal) error {
	filename := filepath.Join(fs.BasePath, "goals", goal.ID+".json")
	data, err := json.MarshalIndent(goal, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal goal: %w", err)
	}

	err = os.WriteFile(filename, data, 0644)
	if err != nil {
		return fmt.Errorf("failed to write goal file: %w", err)
	}

	return nil
}

// LoadGoal loads a goal from disk
func (fs *FileBasedStore) LoadGoal(id string) (*Goal, error) {
	filename := filepath.Join(fs.BasePath, "goals", id+".json")
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read goal file: %w", err)
	}

	var goal Goal
	err = json.Unmarshal(data, &goal)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal goal: %w", err)
	}

	return &goal, nil
}

// ListGoals lists all goals
func (fs *FileBasedStore) ListGoals() ([]*Goal, error) {
	goalsDir := filepath.Join(fs.BasePath, "goals")
	entries, err := os.ReadDir(goalsDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read goals directory: %w", err)
	}

	goals := make([]*Goal, 0)
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".json" {
			continue
		}

		id := entry.Name()[:len(entry.Name())-5] // Remove .json
		goal, err := fs.LoadGoal(id)
		if err != nil {
			continue // Skip invalid files
		}
		goals = append(goals, goal)
	}

	return goals, nil
}

// DeleteGoal deletes a goal
func (fs *FileBasedStore) DeleteGoal(id string) error {
	filename := filepath.Join(fs.BasePath, "goals", id+".json")
	return os.Remove(filename)
}

// SaveSession saves a session to disk
func (fs *FileBasedStore) SaveSession(session *Session) error {
	filename := filepath.Join(fs.BasePath, "sessions", session.ID+".json")
	data, err := json.MarshalIndent(session, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal session: %w", err)
	}

	err = os.WriteFile(filename, data, 0644)
	if err != nil {
		return fmt.Errorf("failed to write session file: %w", err)
	}

	return nil
}

// LoadSession loads a session from disk
func (fs *FileBasedStore) LoadSession(id string) (*Session, error) {
	filename := filepath.Join(fs.BasePath, "sessions", id+".json")
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read session file: %w", err)
	}

	var session Session
	err = json.Unmarshal(data, &session)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal session: %w", err)
	}

	return &session, nil
}

// ListSessions lists all sessions
func (fs *FileBasedStore) ListSessions() ([]*Session, error) {
	sessionsDir := filepath.Join(fs.BasePath, "sessions")
	entries, err := os.ReadDir(sessionsDir)
	if err != nil {
		return nil, fmt.Errorf("failed to read sessions directory: %w", err)
	}

	sessions := make([]*Session, 0)
	for _, entry := range entries {
		if entry.IsDir() || filepath.Ext(entry.Name()) != ".json" {
			continue
		}

		id := entry.Name()[:len(entry.Name())-5] // Remove .json
		session, err := fs.LoadSession(id)
		if err != nil {
			continue // Skip invalid files
		}
		sessions = append(sessions, session)
	}

	return sessions, nil
}

// DeleteSession deletes a session
func (fs *FileBasedStore) DeleteSession(id string) error {
	filename := filepath.Join(fs.BasePath, "sessions", id+".json")
	return os.Remove(filename)
}
