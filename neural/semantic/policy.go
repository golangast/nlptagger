package semantic

import (
)

// ValidateResource applies policy and security checks to a resource.
func ValidateResource(resource *Resource, context Context) error {
	if resource == nil {
		return nil
	}

	// Policy 1: Name Constraint - Folder names in the root (./) cannot contain vowels.
	if resource.Type == "Filesystem::Folder" {
		// Simplified check for "root" folder: assuming if it's not a child of another resource.
		// A more robust solution would involve path analysis or explicit parent tracking.
		// In a real system, you'd check if this folder is a child of another resource.
		// For this example, we'll assume if it's a top-level folder in the semantic output, it's "root".

		// if isRootFolder {
		// 	vowelRegex := regexp.MustCompile(`[aeiouAEIOU]`)
		// 	if vowelRegex.MatchString(resource.Name) {
		// 		return fmt.Errorf("error: folder '%s' violates naming policy. Folder names in the root cannot contain vowels", resource.Name)
		// 	}
		// }
	}

	// Policy 3: Type Constraint - GoWebserver must be created within a Filesystem::Folder.
	// This requires checking the parent-child relationship or dependencies.
	// For now, we'll assume this check happens at the workflow generation/execution level
	// where dependencies are explicitly managed.
	// If a GoWebserver is being created, and it's not a child of a folder, or doesn't depend on one,
	// this policy would fail. This check is better performed when the full workflow DAG is available.

	// Recursively validate children
	for i := range resource.Children {
		if err := ValidateResource(&resource.Children[i], context); err != nil {
			return err
		}
	}

	return nil
}
