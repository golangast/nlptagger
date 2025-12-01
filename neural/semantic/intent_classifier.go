package semantic

import (
	"strings"
)

// IntentClassifier classifies user queries into intent types
type IntentClassifier struct{}

// NewIntentClassifier creates a new intent classifier
func NewIntentClassifier() *IntentClassifier {
	return &IntentClassifier{}
}

// Classify determines the intent type from a query string
// Uses simple rule-based classification based on keywords
func (ic *IntentClassifier) Classify(query string) IntentType {
	query = strings.ToLower(strings.TrimSpace(query))

	// Check for add_feature intent FIRST (before filesystem operations)
	// Pattern: "add [feature] to [component]" with feature and component keywords
	hasFeatureKeyword := containsAny(query, []string{
		"jwt", "auth", "authentication", "authorization",
		"logging", "logger", "log",
		"caching", "cache",
		"validation", "validator",
		"encryption",
		"oauth", "session",
		"cors", "csrf",
		"websocket", "graphql", "rest",
		"rate-limiting", "throttle",
		"metrics", "monitoring",
		"database", "db", "sql", "sqlite",
	})

	hasComponentKeyword := containsAny(query, []string{
		"webserver", "server",
		"api",
		"service",
		"handler", "controller",
		"middleware",
		"router",
		"application", "app",
		"component",
	})

	// If query has "add X to Y" pattern with feature and component keywords
	if containsAny(query, []string{"add", "implement", "integrate", "enable", "connect", "use", "configure"}) &&
		containsAny(query, []string{"to", "in", "into"}) &&
		hasFeatureKeyword &&
		hasComponentKeyword {
		return IntentAddFeature
	}

	// Check for database creation
	if containsAny(query, []string{"add", "create", "make", "new"}) &&
		containsAny(query, []string{"database", "db", "sql", "sqlite"}) {
		return IntentAddFeature
	}

	// Check for modify_code intent
	if containsAny(query, []string{"modify", "update", "change", "refactor"}) &&
		containsAny(query, []string{
			"code", "component", "service", "api", "handler",
			"webserver", "server", "middleware", "controller",
		}) {
		return IntentModifyCode
	}

	// Check for folder with file creation
	// Detect patterns like "add folder X and add file.go in it"
	hasFolder := containsAny(query, []string{"folder", "directory", "dir"})
	hasFile := hasFileReference(query)
	hasConjunction := containsAny(query, []string{"and", "with", "in it", "inside", "into"})
	hasCreateAction := containsAny(query, []string{"add", "create", "make", "new"})

	if hasFolder && hasFile && (hasConjunction || hasCreateAction) {
		return IntentCreateFolderWithFile
	}

	// Check for folder creation only
	if containsAny(query, []string{"add", "create", "make", "new"}) &&
		containsAny(query, []string{"folder", "directory", "dir"}) &&
		!hasFeatureKeyword && !hasComponentKeyword {
		return IntentCreateFolder
	}

	// Check for file creation
	if containsAny(query, []string{"add", "create", "make", "new"}) &&
		(containsAny(query, []string{"file"}) || hasFileReference(query)) &&
		!hasFeatureKeyword {
		return IntentCreateFile
	}

	// Check for file deletion
	if containsAny(query, []string{"delete", "remove", "rm"}) &&
		containsAny(query, []string{"file"}) {
		return IntentDeleteFile
	}

	// Check for folder deletion
	if containsAny(query, []string{"delete", "remove", "rm"}) &&
		containsAny(query, []string{"folder", "directory", "dir"}) {
		return IntentDeleteFolder
	}

	// Check for file reading/opening
	if containsAny(query, []string{"read", "open", "view", "show"}) &&
		containsAny(query, []string{"file"}) {
		return IntentReadFile
	}

	// Check for move operations
	hasMoveAction := containsAny(query, []string{"move", "relocate", "mv", "transfer"})
	hasDirectional := containsAny(query, []string{"into", "to", "in"})

	if hasMoveAction && hasDirectional {
		// Check for file first (a file move may mention a destination folder)
		if hasFileReference(query) {
			return IntentMoveFile
		} else if containsAny(query, []string{"folder", "directory", "dir"}) {
			return IntentMoveFolder
		}
	}

	// Check for rename operations
	hasRenameAction := containsAny(query, []string{"rename", "renaming"})
	hasToKeyword := containsAny(query, []string{"to"})

	if hasRenameAction && hasToKeyword {
		// Determine if renaming file or folder
		if hasFileReference(query) {
			return IntentRenameFile
		} else if containsAny(query, []string{"folder", "directory", "dir"}) {
			return IntentRenameFolder
		}
	}

	return IntentUnknown
}

// containsAny checks if the text contains any of the keywords
func containsAny(text string, keywords []string) bool {
	for _, keyword := range keywords {
		if strings.Contains(text, keyword) {
			return true
		}
	}
	return false
}

// hasFileReference checks if the query contains a file reference
// This includes the literal word "file" or common file extensions
func hasFileReference(query string) bool {
	// Check for literal "file" keyword
	if strings.Contains(query, "file") {
		return true
	}

	// Common file extensions
	extensions := []string{
		".go", ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".hpp",
		".txt", ".md", ".json", ".yaml", ".yml", ".xml", ".html", ".css",
		".sh", ".bash", ".sql", ".rs", ".rb", ".php", ".swift", ".kt",
		".scala", ".r", ".m", ".pl", ".lua", ".vim", ".toml", ".ini",
		".conf", ".cfg", ".env", ".proto", ".graphql", ".vue", ".jsx", ".tsx",
	}

	for _, ext := range extensions {
		if strings.Contains(query, ext) {
			return true
		}
	}

	return false
}
