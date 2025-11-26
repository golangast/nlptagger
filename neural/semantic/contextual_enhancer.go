package semantic

import "strings"

// ContextualQueryEnhancer enhances queries with filesystem context
type ContextualQueryEnhancer struct {
	context *FilesystemContext
}

// NewContextualQueryEnhancer creates a new query enhancer
func NewContextualQueryEnhancer(context *FilesystemContext) *ContextualQueryEnhancer {
	return &ContextualQueryEnhancer{
		context: context,
	}
}

// Enhance transforms a query by adding contextual information
func (cqe *ContextualQueryEnhancer) Enhance(query string) string {
	words := strings.Fields(query)
	enhanced := make([]string, 0, len(words)+5)

	i := 0
	for i < len(words) {
		word := words[i]
		lowerWord := strings.ToLower(word)

		// Handle "add <file> to <folder>" pattern
		if lowerWord == "add" && i+2 < len(words) && strings.ToLower(words[i+2]) == "to" {
			// Pattern: add <filename> to <foldername>
			fileName := words[i+1]

			// Check if there's a folder reference after "to"
			if i+3 < len(words) {
				folderRef := words[i+3]

				// Try to find the folder in context
				if _, found := cqe.context.FindFolder(folderRef); found {
					// Convert "add X to Y" → "create file X in folder Y"
					enhanced = append(enhanced, "create", "file", fileName, "in", "folder", folderRef)
					i += 4 // Skip "add file to folder"
					continue
				}
			}
		}

		// Handle "add <file>" without destination (use current directory)
		if lowerWord == "add" && i+1 < len(words) {
			nextWord := words[i+1]
			// If next word looks like a file (has extension or "file" keyword coming)
			if strings.Contains(nextWord, ".") || (i+2 < len(words) && strings.ToLower(words[i+2]) != "to") {
				enhanced = append(enhanced, "create", "file")
				i++ // Move to next word, will be added in next iteration
				continue
			}
		}

		// Check if word is a known folder name being referenced
		if i > 0 && (strings.ToLower(words[i-1]) == "in" || strings.ToLower(words[i-1]) == "to") {
			if _, found := cqe.context.FindFolder(word); found {
				// Add "folder" keyword if not already there
				if i+1 >= len(words) || strings.ToLower(words[i+1]) != "folder" {
					enhanced = append(enhanced, "folder")
				}
			}
		}

		enhanced = append(enhanced, word)
		i++
	}

	result := strings.Join(enhanced, " ")

	// Only log if we actually changed something
	if result != query {
		// Enhanced query
		return result
	}

	return query
}
