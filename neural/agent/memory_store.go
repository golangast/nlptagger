package agent

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// Message represents a conversation turn
type Message struct {
	Role      string                 `json:"role"` // "user" or "assistant" or "system"
	Content   string                 `json:"content"`
	Timestamp time.Time              `json:"timestamp"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// KnowledgeItem represents a piece of stored knowledge
type KnowledgeItem struct {
	Key       string                 `json:"key"`
	Content   string                 `json:"content"`
	Embedding []float32              `json:"embedding,omitempty"` // For vector search later
	Tags      []string               `json:"tags,omitempty"`
	CreatedAt time.Time              `json:"created_at"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// Action represents a logged agent action
type Action struct {
	Type      string                 `json:"type"` // "plan_created", "tool_executed", etc.
	Data      interface{}            `json:"data"`
	Timestamp time.Time              `json:"timestamp"`
	Success   bool                   `json:"success"`
	Error     string                 `json:"error,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// ActionFilters for querying action history
type ActionFilters struct {
	Type      string
	StartTime time.Time
	EndTime   time.Time
	Limit     int
}

// MemoryStore interface for different storage backends
type MemoryStore interface {
	// Conversation history
	AppendMessage(role string, content string) error
	GetConversationHistory(limit int) ([]Message, error)
	ClearConversation() error

	// Knowledge base (RAG-ready)
	StoreKnowledge(key string, content string, tags []string) error
	GetKnowledge(key string) (*KnowledgeItem, error)
	SearchKnowledge(query string, topK int) ([]KnowledgeItem, error)

	// Historical actions
	LogAction(action Action) error
	GetActionHistory(filters ActionFilters) ([]Action, error)

	// Persistence
	Save() error
	Load() error
}

// InMemoryStore implements MemoryStore with in-memory storage
type InMemoryStore struct {
	mu            sync.RWMutex
	conversations []Message
	knowledge     map[string]KnowledgeItem
	actions       []Action
	maxMessages   int  // Max conversation history to keep
	persist       bool // Whether to persist to disk
	persistPath   string
}

// NewInMemoryStore creates a new in-memory store
func NewInMemoryStore() *InMemoryStore {
	return &InMemoryStore{
		conversations: make([]Message, 0),
		knowledge:     make(map[string]KnowledgeItem),
		actions:       make([]Action, 0),
		maxMessages:   100,
		persist:       false,
	}
}

// NewPersistentStore creates an in-memory store with disk persistence
func NewPersistentStore(path string) *InMemoryStore {
	store := &InMemoryStore{
		conversations: make([]Message, 0),
		knowledge:     make(map[string]KnowledgeItem),
		actions:       make([]Action, 0),
		maxMessages:   100,
		persist:       true,
		persistPath:   path,
	}

	// Try to load existing data
	_ = store.Load()

	return store
}

// AppendMessage adds a message to conversation history
func (ims *InMemoryStore) AppendMessage(role string, content string) error {
	ims.mu.Lock()
	defer ims.mu.Unlock()

	message := Message{
		Role:      role,
		Content:   content,
		Timestamp: time.Now(),
		Metadata:  make(map[string]interface{}),
	}

	ims.conversations = append(ims.conversations, message)

	// Trim if exceeds max
	if len(ims.conversations) > ims.maxMessages {
		ims.conversations = ims.conversations[len(ims.conversations)-ims.maxMessages:]
	}

	if ims.persist {
		return ims.save()
	}

	return nil
}

// GetConversationHistory retrieves recent messages
func (ims *InMemoryStore) GetConversationHistory(limit int) ([]Message, error) {
	ims.mu.RLock()
	defer ims.mu.RUnlock()

	if limit <= 0 || limit > len(ims.conversations) {
		limit = len(ims.conversations)
	}

	start := len(ims.conversations) - limit
	if start < 0 {
		start = 0
	}

	return ims.conversations[start:], nil
}

// ClearConversation removes all conversation history
func (ims *InMemoryStore) ClearConversation() error {
	ims.mu.Lock()
	defer ims.mu.Unlock()

	ims.conversations = make([]Message, 0)

	if ims.persist {
		return ims.save()
	}

	return nil
}

// StoreKnowledge adds a knowledge item
func (ims *InMemoryStore) StoreKnowledge(key string, content string, tags []string) error {
	ims.mu.Lock()
	defer ims.mu.Unlock()

	item := KnowledgeItem{
		Key:       key,
		Content:   content,
		Tags:      tags,
		CreatedAt: time.Now(),
		Metadata:  make(map[string]interface{}),
	}

	ims.knowledge[key] = item

	if ims.persist {
		return ims.save()
	}

	return nil
}

// GetKnowledge retrieves a knowledge item by key
func (ims *InMemoryStore) GetKnowledge(key string) (*KnowledgeItem, error) {
	ims.mu.RLock()
	defer ims.mu.RUnlock()

	item, exists := ims.knowledge[key]
	if !exists {
		return nil, fmt.Errorf("knowledge item not found: %s", key)
	}

	return &item, nil
}

// SearchKnowledge performs simple keyword-based search
// TODO: Replace with vector similarity search for better RAG
func (ims *InMemoryStore) SearchKnowledge(query string, topK int) ([]KnowledgeItem, error) {
	ims.mu.RLock()
	defer ims.mu.RUnlock()

	results := make([]KnowledgeItem, 0)

	// Simple keyword matching for now
	for _, item := range ims.knowledge {
		if containsKeyword(item.Content, query) || containsKeyword(item.Key, query) {
			results = append(results, item)
			if len(results) >= topK {
				break
			}
		}
	}

	return results, nil
}

// LogAction records an agent action
func (ims *InMemoryStore) LogAction(action Action) error {
	ims.mu.Lock()
	defer ims.mu.Unlock()

	action.Timestamp = time.Now()
	ims.actions = append(ims.actions, action)

	if ims.persist {
		return ims.save()
	}

	return nil
}

// GetActionHistory retrieves action history with filters
func (ims *InMemoryStore) GetActionHistory(filters ActionFilters) ([]Action, error) {
	ims.mu.RLock()
	defer ims.mu.RUnlock()

	results := make([]Action, 0)

	for _, action := range ims.actions {
		// Apply filters
		if filters.Type != "" && action.Type != filters.Type {
			continue
		}

		if !filters.StartTime.IsZero() && action.Timestamp.Before(filters.StartTime) {
			continue
		}

		if !filters.EndTime.IsZero() && action.Timestamp.After(filters.EndTime) {
			continue
		}

		results = append(results, action)

		if filters.Limit > 0 && len(results) >= filters.Limit {
			break
		}
	}

	return results, nil
}

// Save persists memory to disk
func (ims *InMemoryStore) Save() error {
	if !ims.persist || ims.persistPath == "" {
		return nil
	}

	ims.mu.RLock()
	defer ims.mu.RUnlock()

	return ims.save()
}

// save writes data to disk (assumes lock is held)
func (ims *InMemoryStore) save() error {
	// Create directory if needed
	dir := filepath.Dir(ims.persistPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	// Prepare data structure
	data := map[string]interface{}{
		"conversations": ims.conversations,
		"knowledge":     ims.knowledge,
		"actions":       ims.actions,
		"saved_at":      time.Now(),
	}

	// Marshal to JSON
	jsonData, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal data: %w", err)
	}

	// Write to file
	if err := os.WriteFile(ims.persistPath, jsonData, 0644); err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}

	return nil
}

// Load reads memory from disk
func (ims *InMemoryStore) Load() error {
	if !ims.persist || ims.persistPath == "" {
		return nil
	}

	// Check if file exists
	if _, err := os.Stat(ims.persistPath); os.IsNotExist(err) {
		return nil // Not an error, just no saved data yet
	}

	// Read file
	jsonData, err := os.ReadFile(ims.persistPath)
	if err != nil {
		return fmt.Errorf("failed to read file: %w", err)
	}

	// Unmarshal
	var data map[string]interface{}
	if err := json.Unmarshal(jsonData, &data); err != nil {
		return fmt.Errorf("failed to unmarshal data: %w", err)
	}

	ims.mu.Lock()
	defer ims.mu.Unlock()

	// Load conversations
	if convData, ok := data["conversations"]; ok {
		convJSON, _ := json.Marshal(convData)
		json.Unmarshal(convJSON, &ims.conversations)
	}

	// Load knowledge
	if knowData, ok := data["knowledge"]; ok {
		knowJSON, _ := json.Marshal(knowData)
		json.Unmarshal(knowJSON, &ims.knowledge)
	}

	// Load actions
	if actData, ok := data["actions"]; ok {
		actJSON, _ := json.Marshal(actData)
		json.Unmarshal(actJSON, &ims.actions)
	}

	return nil
}

// Helper function for simple keyword matching
func containsKeyword(text, query string) bool {
	return len(query) > 0 && len(text) > 0 &&
		(text == query ||
			(len(text) > len(query) &&
				(text[:len(query)] == query ||
					text[len(text)-len(query):] == query)))
}
