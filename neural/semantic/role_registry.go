package semantic

import "strings"

// SemanticRole represents a semantic role for files/folders
type SemanticRole string

const (
	RoleEntrypoint      SemanticRole = "entrypoint"       // Main entry file
	RoleHandler         SemanticRole = "handler"          // Request/event handler
	RoleTest            SemanticRole = "test"             // Test file
	RoleConfig          SemanticRole = "config"           // Configuration file
	RoleAssetDirectory  SemanticRole = "asset_directory"  // Assets/static files folder
	RoleModel           SemanticRole = "model"            // Data model
	RoleRoute           SemanticRole = "route"            // Route definition
	RoleMiddleware      SemanticRole = "middleware"       // Middleware
	RoleUtil            SemanticRole = "util"             // Utility functions
	RoleController      SemanticRole = "controller"       // Controller
	RoleService         SemanticRole = "service"          // Service layer
	RoleRepository      SemanticRole = "repository"       // Data access layer
	RoleProjectRoot     SemanticRole = "project_root"     // Top-level project folder
	RoleSourceDirectory SemanticRole = "source_directory" // Source code folder
)

// RoleRegistry manages semantic roles and their mappings
type RoleRegistry struct {
	// Maps role keywords to roles
	keywordMap map[string]SemanticRole

	// Maps file patterns to roles
	patternMap map[string]SemanticRole
}

// NewRoleRegistry creates a new role registry
func NewRoleRegistry() *RoleRegistry {
	registry := &RoleRegistry{
		keywordMap: make(map[string]SemanticRole),
		patternMap: make(map[string]SemanticRole),
	}

	registry.registerDefaults()
	return registry
}

// registerDefaults registers default role mappings
func (rr *RoleRegistry) registerDefaults() {
	// Keyword mappings (for natural language)
	rr.keywordMap["entrypoint"] = RoleEntrypoint
	rr.keywordMap["entry"] = RoleEntrypoint
	rr.keywordMap["main"] = RoleEntrypoint
	rr.keywordMap["start"] = RoleEntrypoint

	rr.keywordMap["handler"] = RoleHandler
	rr.keywordMap["route handler"] = RoleHandler

	rr.keywordMap["test"] = RoleTest
	rr.keywordMap["tests"] = RoleTest

	rr.keywordMap["config"] = RoleConfig
	rr.keywordMap["configuration"] = RoleConfig
	rr.keywordMap["settings"] = RoleConfig

	rr.keywordMap["assets"] = RoleAssetDirectory
	rr.keywordMap["static"] = RoleAssetDirectory
	rr.keywordMap["templates"] = RoleAssetDirectory

	rr.keywordMap["model"] = RoleModel
	rr.keywordMap["schema"] = RoleModel

	rr.keywordMap["route"] = RoleRoute
	rr.keywordMap["router"] = RoleRoute
	rr.keywordMap["routes"] = RoleRoute

	rr.keywordMap["middleware"] = RoleMiddleware

	rr.keywordMap["util"] = RoleUtil
	rr.keywordMap["utility"] = RoleUtil
	rr.keywordMap["helper"] = RoleUtil

	rr.keywordMap["controller"] = RoleController
	rr.keywordMap["service"] = RoleService
	rr.keywordMap["repository"] = RoleRepository

	// Pattern mappings (for file names)
	rr.patternMap["main.go"] = RoleEntrypoint
	rr.patternMap["server.go"] = RoleEntrypoint
	rr.patternMap["app.go"] = RoleEntrypoint
	rr.patternMap["index.js"] = RoleEntrypoint
	rr.patternMap["index.ts"] = RoleEntrypoint
	rr.patternMap["main.py"] = RoleEntrypoint

	rr.patternMap["handler.go"] = RoleHandler
	rr.patternMap["handlers.go"] = RoleHandler

	rr.patternMap["config.json"] = RoleConfig
	rr.patternMap["config.yaml"] = RoleConfig
	rr.patternMap["settings.json"] = RoleConfig
	rr.patternMap[".env"] = RoleConfig

	// Folders
	rr.patternMap["templates"] = RoleAssetDirectory
	rr.patternMap["static"] = RoleAssetDirectory
	rr.patternMap["assets"] = RoleAssetDirectory
	rr.patternMap["public"] = RoleAssetDirectory

	rr.patternMap["src"] = RoleSourceDirectory
	rr.patternMap["source"] = RoleSourceDirectory
}

// InferRole infers a role from a file/folder name and context
func (rr *RoleRegistry) InferRole(name string, nodeType string, parentRole string) string {
	lowerName := strings.ToLower(name)

	// Check pattern map first
	if role, exists := rr.patternMap[lowerName]; exists {
		return string(role)
	}

	// Check for test files
	if strings.Contains(lowerName, "_test.") || strings.Contains(lowerName, ".test.") {
		return string(RoleTest)
	}

	// Check for config files
	if strings.Contains(lowerName, "config") || strings.Contains(lowerName, "settings") {
		return string(RoleConfig)
	}

	// Infer from parent context
	if parentRole == string(RoleProjectRoot) && nodeType == "folder" {
		if lowerName == "src" || lowerName == "source" {
			return string(RoleSourceDirectory)
		}
	}

	// Default: no specific role
	return ""
}

// GetRoleFromKeyword extracts a role from natural language keywords
func (rr *RoleRegistry) GetRoleFromKeyword(keyword string) (SemanticRole, bool) {
	lowerKeyword := strings.ToLower(strings.TrimSpace(keyword))

	if role, exists := rr.keywordMap[lowerKeyword]; exists {
		return role, true
	}

	return "", false
}

// GetRoleDescription returns a human-readable description of a role
func (rr *RoleRegistry) GetRoleDescription(role SemanticRole) string {
	descriptions := map[SemanticRole]string{
		RoleEntrypoint:      "Application entry point",
		RoleHandler:         "Request/event handler",
		RoleTest:            "Test file",
		RoleConfig:          "Configuration file",
		RoleAssetDirectory:  "Assets/static files directory",
		RoleModel:           "Data model",
		RoleRoute:           "Route definition",
		RoleMiddleware:      "Middleware function",
		RoleUtil:            "Utility functions",
		RoleController:      "Controller",
		RoleService:         "Service layer",
		RoleRepository:      "Data access layer",
		RoleProjectRoot:     "Project root directory",
		RoleSourceDirectory: "Source code directory",
	}

	if desc, exists := descriptions[role]; exists {
		return desc
	}

	return "Unknown role"
}
