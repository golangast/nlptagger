package agent

import (
	"fmt"

	"github.com/zendrulat/nlptagger/neural/semantic"
)

// CreateFolderTool creates folders in the VFS
type CreateFolderTool struct {
	vfs     *semantic.VFSTree
	roleReg *semantic.RoleRegistry
}

// NewCreateFolderTool creates a new folder creation tool
func NewCreateFolderTool(vfs *semantic.VFSTree, roleReg *semantic.RoleRegistry) *CreateFolderTool {
	return &CreateFolderTool{
		vfs:     vfs,
		roleReg: roleReg,
	}
}

func (cft *CreateFolderTool) Name() string {
	return "create_folder"
}

func (cft *CreateFolderTool) Description() string {
	return "Creates a new folder in the virtual filesystem"
}

func (cft *CreateFolderTool) Schema() ToolSchema {
	return ToolSchema{
		Parameters: map[string]ParameterDef{
			"name": {
				Type:        "string",
				Description: "Name of the folder to create",
			},
		},
		Required: []string{"name"},
	}
}

func (cft *CreateFolderTool) Execute(args map[string]interface{}) (ToolResult, error) {
	name, ok := args["name"].(string)
	if !ok || name == "" {
		return ToolResult{
			Success: false,
			Error:   fmt.Errorf("folder name is required"),
		}, nil
	}

	path := "/" + name
	role := cft.roleReg.InferRole(name, "folder", "")
	cft.vfs.CreateFolder(path, role)

	return ToolResult{
		Success: true,
		Output:  fmt.Sprintf("Created folder: %s", path),
	}, nil
}

// CreateFileTool creates files in the VFS
type CreateFileTool struct {
	vfs     *semantic.VFSTree
	roleReg *semantic.RoleRegistry
}

// NewCreateFileTool creates a new file creation tool
func NewCreateFileTool(vfs *semantic.VFSTree, roleReg *semantic.RoleRegistry) *CreateFileTool {
	return &CreateFileTool{
		vfs:     vfs,
		roleReg: roleReg,
	}
}

func (cft *CreateFileTool) Name() string {
	return "create_file"
}

func (cft *CreateFileTool) Description() string {
	return "Creates a new file in the virtual filesystem"
}

func (cft *CreateFileTool) Schema() ToolSchema {
	return ToolSchema{
		Parameters: map[string]ParameterDef{
			"name": {
				Type:        "string",
				Description: "Name of the file to create",
			},
			"folder": {
				Type:        "string",
				Description: "Parent folder path",
			},
			"content": {
				Type:        "string",
				Description: "File content",
				Default:     "",
			},
		},
		Required: []string{"name"},
	}
}

func (cft *CreateFileTool) Execute(args map[string]interface{}) (ToolResult, error) {
	name, ok := args["name"].(string)
	if !ok || name == "" {
		return ToolResult{
			Success: false,
			Error:   fmt.Errorf("file name is required"),
		}, nil
	}

	folder, _ := args["folder"].(string)
	content, _ := args["content"].(string)

	var path string
	if folder != "" {
		path = "/" + folder + "/" + name
	} else {
		path = "/" + name
	}

	role := cft.roleReg.InferRole(name, "file", content)
	cft.vfs.CreateFile(path, role, content)

	return ToolResult{
		Success: true,
		Output:  fmt.Sprintf("Created file: %s", path),
	}, nil
}

// ApplyTemplateTool applies blueprints to projects
type ApplyTemplateTool struct {
	vfs       *semantic.VFSTree
	blueprint *semantic.BlueprintEngine
	roleReg   *semantic.RoleRegistry
}

// NewApplyTemplateTool creates a new template application tool
func NewApplyTemplateTool(vfs *semantic.VFSTree, blueprint *semantic.BlueprintEngine, roleReg *semantic.RoleRegistry) *ApplyTemplateTool {
	return &ApplyTemplateTool{
		vfs:       vfs,
		blueprint: blueprint,
		roleReg:   roleReg,
	}
}

func (att *ApplyTemplateTool) Name() string {
	return "apply_template"
}

func (att *ApplyTemplateTool) Description() string {
	return "Applies a project template/blueprint to a folder"
}

func (att *ApplyTemplateTool) Schema() ToolSchema {
	return ToolSchema{
		Parameters: map[string]ParameterDef{
			"template": {
				Type:        "string",
				Description: "Template name (e.g., 'webserver', 'cli')",
			},
			"folder": {
				Type:        "string",
				Description: "Target folder",
			},
		},
		Required: []string{"template", "folder"},
	}
}

func (att *ApplyTemplateTool) Execute(args map[string]interface{}) (ToolResult, error) {
	template, ok := args["template"].(string)
	if !ok || template == "" {
		return ToolResult{
			Success: false,
			Error:   fmt.Errorf("template name is required"),
		}, nil
	}

	folder, ok := args["folder"].(string)
	if !ok || folder == "" {
		return ToolResult{
			Success: false,
			Error:   fmt.Errorf("folder is required"),
		}, nil
	}

	// Get blueprint
	bp, exists := att.blueprint.GetBlueprint(template)
	if !exists {
		return ToolResult{
			Success: false,
			Error:   fmt.Errorf("template not found: %s", template),
		}, nil
	}

	// Extract parameters (simple for now)
	params := make(map[string]interface{})

	// Generate main content
	mainContent, _ := att.blueprint.Execute(template, params)

	// Create main file
	folderPath := "/" + folder
	mainPath := folderPath + "/main.go"
	att.vfs.CreateFile(mainPath, string(semantic.RoleEntrypoint), mainContent)

	// Create additional blueprint files
	for _, bpFile := range bp.Files {
		content, _ := att.blueprint.ExecuteFile(bp, bpFile, params)
		filePath := folderPath + "/" + bpFile.Name
		att.vfs.CreateFile(filePath, bpFile.Role, content)
	}

	return ToolResult{
		Success: true,
		Output:  fmt.Sprintf("Applied template '%s' to %s", template, folder),
		Metadata: map[string]interface{}{
			"files_created": len(bp.Files) + 1,
		},
	}, nil
}

// ExecuteCommandTool simulates command execution
type ExecuteCommandTool struct {
	vfs *semantic.VFSTree
}

// NewExecuteCommandTool creates a new command execution tool
func NewExecuteCommandTool(vfs *semantic.VFSTree) *ExecuteCommandTool {
	return &ExecuteCommandTool{
		vfs: vfs,
	}
}

func (ect *ExecuteCommandTool) Name() string {
	return "execute_command"
}

func (ect *ExecuteCommandTool) Description() string {
	return "Executes a shell command (simulated in VFS)"
}

func (ect *ExecuteCommandTool) Schema() ToolSchema {
	return ToolSchema{
		Parameters: map[string]ParameterDef{
			"command": {
				Type:        "string",
				Description: "Command to execute",
			},
		},
		Required: []string{"command"},
	}
}

func (ect *ExecuteCommandTool) Execute(args map[string]interface{}) (ToolResult, error) {
	command, ok := args["command"].(string)
	if !ok || command == "" {
		return ToolResult{
			Success: false,
			Error:   fmt.Errorf("command is required"),
		}, nil
	}

	// Simulate "go mod init"
	if len(command) > 11 && command[:11] == "go mod init" {
		moduleName := command[12:]
		// Heuristic: assume project folder is named "project" or matches module name
		// For this specific goal engine logic, we know it's "project"
		path := "/project/go.mod"
		content := fmt.Sprintf("module %s\n\ngo 1.20\n", moduleName)
		ect.vfs.CreateFile(path, "config", content)
		return ToolResult{
			Success: true,
			Output:  fmt.Sprintf("Initialized go.mod for module %s", moduleName),
		}, nil
	}

	return ToolResult{
		Success: true,
		Output:  fmt.Sprintf("Executed command: %s (simulated)", command),
	}, nil
}
