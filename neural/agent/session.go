package agent

import (
	"time"
)

// Session represents a work session focused on a goal
type Session struct {
	ID         string    `json:"id"`
	GoalID     string    `json:"goal_id"`
	StartedAt  time.Time `json:"started_at"`
	LastActive time.Time `json:"last_active"`
	Status     string    `json:"status"` // active, paused, completed
}

// SessionManager manages work sessions
type SessionManager struct {
	store          GoalStore
	idCounter      int
	activeSessions map[string]*Session
}

// NewSessionManager creates a new session manager
func NewSessionManager(store GoalStore) *SessionManager {
	return &SessionManager{
		store:          store,
		idCounter:      0,
		activeSessions: make(map[string]*Session),
	}
}

// CreateSession creates a new session for a goal
func (sm *SessionManager) CreateSession(goalID string) *Session {
	sm.idCounter++
	session := &Session{
		ID:         sm.generateSessionID(),
		GoalID:     goalID,
		StartedAt:  time.Now(),
		LastActive: time.Now(),
		Status:     "active",
	}

	sm.activeSessions[session.ID] = session
	sm.store.SaveSession(session)

	return session
}

// GetSession retrieves a session by ID
func (sm *SessionManager) GetSession(sessionID string) (*Session, error) {
	// Check active sessions first
	if session, exists := sm.activeSessions[sessionID]; exists {
		return session, nil
	}

	// Load from store
	return sm.store.LoadSession(sessionID)
}

// UpdateSession updates session last active time
func (sm *SessionManager) UpdateSession(session *Session) error {
	session.LastActive = time.Now()
	sm.activeSessions[session.ID] = session
	return sm.store.SaveSession(session)
}

// PauseSession pauses an active session
func (sm *SessionManager) PauseSession(sessionID string) error {
	session, err := sm.GetSession(sessionID)
	if err != nil {
		return err
	}

	session.Status = "paused"
	session.LastActive = time.Now()
	return sm.store.SaveSession(session)
}

// ResumeSession resumes a paused session
func (sm *SessionManager) ResumeSession(sessionID string) (*Session, error) {
	session, err := sm.GetSession(sessionID)
	if err != nil {
		return nil, err
	}

	session.Status = "active"
	session.LastActive = time.Now()
	sm.activeSessions[sessionID] = session
	sm.store.SaveSession(session)

	return session, nil
}

// ListSessions returns all sessions
func (sm *SessionManager) ListSessions() ([]*Session, error) {
	return sm.store.ListSessions()
}

func (sm *SessionManager) generateSessionID() string {
	return time.Now().Format("sess_20060102_150405")
}
