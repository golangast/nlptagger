package sqlite_db

import (
	"database/sql"
	"fmt"
	_ "github.com/glebarez/sqlite" // Pure Go SQLite driver
	"log"
	"os"
	"path/filepath"
)

// InitDB initializes an SQLite database at the given path.
// It creates the database file if it doesn't exist and sets up a 'messages' table.
func InitDB(dataSourceName string) (*sql.DB, error) {
	// Extract the directory from the dataSourceName
	dir := filepath.Dir(dataSourceName)
	// Create the directory if it doesn't exist
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		err := os.MkdirAll(dir, 0755) // Use MkdirAll to create parent directories as well
		if err != nil {
			return nil, fmt.Errorf("failed to create directory %s: %w", dir, err)
		}
	}

	// The driver name for github.com/glebarez/sqlite is "sqlite" or "sqlite3"
	db, err := sql.Open("sqlite", dataSourceName)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	// Create messages table
	createTableSQL := `CREATE TABLE IF NOT EXISTS messages (
		"id" INTEGER PRIMARY KEY AUTOINCREMENT,
		"role" TEXT NOT NULL,
		"content" TEXT NOT NULL,
		"timestamp" DATETIME DEFAULT CURRENT_TIMESTAMP
	);`

	_, err = db.Exec(createTableSQL)
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to create messages table: %w", err)
	}

	log.Printf("SQLite database initialized at %s", dataSourceName)
	return db, nil
}

// SaveMessage saves a message to the 'messages' table.
func SaveMessage(db *sql.DB, role, content string) error {
	insertSQL := `INSERT INTO messages(role, content) VALUES (?, ?)`
	_, err := db.Exec(insertSQL, role, content)
	if err != nil {
		return fmt.Errorf("failed to insert message: %w", err)
	}
	return nil
}

// Message represents a message stored in the database.
type Message struct {
	ID        int
	Role      string
	Content   string
	Timestamp string
}

// GetMessages retrieves all messages from the 'messages' table.
func GetMessages(db *sql.DB) ([]Message, error) {
	query := `SELECT id, role, content, timestamp FROM messages ORDER BY timestamp ASC`
	rows, err := db.Query(query)
	if err != nil {
		return nil, fmt.Errorf("failed to query messages: %w", err)
	}
	defer rows.Close()

	var messages []Message
	for rows.Next() {
		var msg Message
		if err := rows.Scan(&msg.ID, &msg.Role, &msg.Content, &msg.Timestamp); err != nil {
			return nil, fmt.Errorf("failed to scan message: %w", err)
		}
		messages = append(messages, msg)
	}

	return messages, nil
}
