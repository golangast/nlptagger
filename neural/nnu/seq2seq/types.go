package seq2seq

import "encoding/json"

// GeneralResponse represents the common structure for all intent outputs.
type GeneralResponse struct {
	Intent   string          `json:"intent"`
	Entities json.RawMessage `json:"entities,omitempty"` // Use RawMessage for flexible entity parsing
	Description string       `json:"description,omitempty"` // For intents like "describe_book"
}

// GenerateWebserverEntities represents entities for "generate_webserver" intent.
type GenerateWebserverEntities struct {
	Object       string `json:"object"`
	Name         string `json:"name"`
	LocationType string `json:"location_type"`
	LocationName string `json:"location_name"`
}

// CreateDatabaseEntities represents entities for "create_database" intent.
type CreateDatabaseEntities struct {
	Object       string `json:"object"`
	Name         string `json:"name"`
	LocationType string `json:"location_type"`
	LocationName string `json:"location_name"`
}
