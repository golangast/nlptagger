package semantic

// InferProperties applies dynamic property inference to a resource.
func InferProperties(resource *Resource) {
	if resource == nil {
		return
	}

	// Apply inference rules based on resource type
	switch resource.Type {
	case "Deployment::GoWebserver":
		if _, ok := resource.Properties["port"]; !ok {
			if resource.Properties == nil {
				resource.Properties = make(map[string]interface{})
			}
			resource.Properties["port"] = 8080
		}
		if _, ok := resource.Properties["runtime_image"]; !ok {
			if resource.Properties == nil {
				resource.Properties = make(map[string]interface{})
			}
			resource.Properties["runtime_image"] = "golang:latest"
		}
	case "Filesystem::Folder":
		if _, ok := resource.Properties["permissions"]; !ok {
			if resource.Properties == nil {
				resource.Properties = make(map[string]interface{})
			}
			resource.Properties["permissions"] = "0755"
		}
	case "Filesystem::File":
		if resource.Name == "" {
			resource.Name = "untitled.txt" // Default file name
		}
		if resource.Directory == "" {
			resource.Directory = "/home/user/documents" // Default directory
		}
	}

	// Recursively infer properties for children
	for i := range resource.Children {
		InferProperties(&resource.Children[i])
	}
}