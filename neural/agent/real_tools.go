package agent

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/zendrulat/nlptagger/neural/semantic"
)

// RealFilesystemTool writes files to the actual filesystem
type RealFilesystemTool struct {
	vfs         *semantic.VFSTree
	roleReg     *semantic.RoleRegistry
	templateReg *TemplateRegistry
	baseDir     string // Base directory for project files
}

// NewRealFilesystemTool creates a new real filesystem tool
func NewRealFilesystemTool(vfs *semantic.VFSTree, roleReg *semantic.RoleRegistry, baseDir string) *RealFilesystemTool {
	return &RealFilesystemTool{
		vfs:         vfs,
		roleReg:     roleReg,
		templateReg: NewTemplateRegistry(),
		baseDir:     baseDir,
	}
}

func (rft *RealFilesystemTool) Name() string {
	return "write_files"
}

func (rft *RealFilesystemTool) Description() string {
	return "Writes files from VFS to the actual filesystem"
}

func (rft *RealFilesystemTool) Schema() ToolSchema {
	return ToolSchema{
		Parameters: map[string]ParameterDef{
			"folder": {
				Type:        "string",
				Description: "Target folder to write to filesystem",
			},
		},
		Required: []string{"folder"},
	}
}

func (rft *RealFilesystemTool) Execute(args map[string]interface{}) (ToolResult, error) {
	folder, ok := args["folder"].(string)
	if !ok || folder == "" {
		return ToolResult{
			Success: false,
			Error:   fmt.Errorf("folder is required"),
		}, nil
	}

	// Get folder node from VFS
	path := "/" + folder
	node, found := rft.vfs.ResolvePath(path)
	if !found {
		return ToolResult{
			Success: false,
			Error:   fmt.Errorf("folder not found in VFS: %s", folder),
		}, nil
	}

	// Create folder on real filesystem
	targetDir := filepath.Join(rft.baseDir, folder)
	if err := os.MkdirAll(targetDir, 0755); err != nil {
		return ToolResult{
			Success: false,
			Error:   err,
		}, nil
	}

	// Write all files in this folder
	filesWritten := 0
	for _, child := range node.Children {
		if child.Type == "file" {
			filePath := filepath.Join(targetDir, child.Name)
			if err := os.WriteFile(filePath, []byte(child.Content), 0644); err != nil {
				return ToolResult{
					Success: false,
					Error:   fmt.Errorf("failed to write %s: %w", child.Name, err),
				}, nil
			}
			filesWritten++
		}
	}

	return ToolResult{
		Success: true,
		Output:  fmt.Sprintf("Wrote %d files to %s", filesWritten, targetDir),
		Metadata: map[string]interface{}{
			"files_written": filesWritten,
			"target_dir":    targetDir,
		},
	}, nil
}

// RealApplyTemplateTool generates real code and writes to filesystem
type RealApplyTemplateTool struct {
	vfs         *semantic.VFSTree
	roleReg     *semantic.RoleRegistry
	templateReg *TemplateRegistry
	baseDir     string
}

// NewRealApplyTemplateTool creates a new real template application tool
func NewRealApplyTemplateTool(vfs *semantic.VFSTree, roleReg *semantic.RoleRegistry, baseDir string) *RealApplyTemplateTool {
	return &RealApplyTemplateTool{
		vfs:         vfs,
		roleReg:     roleReg,
		templateReg: NewTemplateRegistry(),
		baseDir:     baseDir,
	}
}

func (rat *RealApplyTemplateTool) Name() string {
	return "apply_template"
}

func (rat *RealApplyTemplateTool) Description() string {
	return "Generates code from template and writes to filesystem"
}

func (rat *RealApplyTemplateTool) Schema() ToolSchema {
	return ToolSchema{
		Parameters: map[string]ParameterDef{
			"template": {
				Type:        "string",
				Description: "Template name (e.g., 'webserver', 'rest_api')",
			},
			"folder": {
				Type:        "string",
				Description: "Target folder",
			},
		},
		Required: []string{"template", "folder"},
	}
}

func (rat *RealApplyTemplateTool) Execute(args map[string]interface{}) (ToolResult, error) {
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

	// Render template
	params := map[string]interface{}{
		"ProjectName": folder,
		"GoVersion":   "1.21",
	}

	files, err := rat.templateReg.RenderTemplate(template, params)
	if err != nil {
		// List available templates for debugging
		available := []string{}
		for name := range rat.templateReg.templates {
			available = append(available, name)
		}
		return ToolResult{
			Success: false,
			Error:   fmt.Errorf("%w. Available templates: %v", err, available),
		}, nil
	}

	// Create folder on real filesystem
	targetDir := filepath.Join(rat.baseDir, folder)
	if err := os.MkdirAll(targetDir, 0755); err != nil {
		return ToolResult{
			Success: false,
			Error:   err,
		}, nil
	}

	// Write files
	filesWritten := 0
	var fileList []string
	for filename, content := range files {
		filePath := filepath.Join(targetDir, filename)
		if err := os.WriteFile(filePath, []byte(content), 0644); err != nil {
			return ToolResult{
				Success: false,
				Error:   fmt.Errorf("failed to write %s: %w", filename, err),
			}, nil
		}
		filesWritten++
		fileList = append(fileList, filename)

		// Also add to VFS for tracking
		vfsPath := "/" + folder + "/" + filename
		role := rat.roleReg.InferRole(filename, "file", content)
		rat.vfs.CreateFile(vfsPath, role, content)
	}

	return ToolResult{
		Success: true,
		Output:  fmt.Sprintf("Applied template '%s' to %s: created %s", template, targetDir, strings.Join(fileList, ", ")),
		Metadata: map[string]interface{}{
			"files_created": filesWritten,
			"template":      template,
			"target_dir":    targetDir,
		},
	}, nil
}

// RealExecuteCommandTool executes actual shell commands
type RealExecuteCommandTool struct {
	vfs     *semantic.VFSTree
	baseDir string
}

// NewRealExecuteCommandTool creates a new command execution tool
func NewRealExecuteCommandTool(vfs *semantic.VFSTree, baseDir string) *RealExecuteCommandTool {
	return &RealExecuteCommandTool{
		vfs:     vfs,
		baseDir: baseDir,
	}
}

func (ret *RealExecuteCommandTool) Name() string {
	return "execute_command"
}

func (ret *RealExecuteCommandTool) Description() string {
	return "Executes a real shell command"
}

func (ret *RealExecuteCommandTool) Schema() ToolSchema {
	return ToolSchema{
		Parameters: map[string]ParameterDef{
			"command": {
				Type:        "string",
				Description: "Command to execute",
			},
			"workdir": {
				Type:        "string",
				Description: "Working directory (relative to base)",
				Default:     "",
			},
		},
		Required: []string{"command"},
	}
}

func (ret *RealExecuteCommandTool) Execute(args map[string]interface{}) (ToolResult, error) {
	command, ok := args["command"].(string)
	if !ok || command == "" {
		return ToolResult{
			Success: false,
			Error:   fmt.Errorf("command is required"),
		}, nil
	}

	workdir, _ := args["workdir"].(string)
	if workdir == "" {
		workdir = ret.baseDir
	} else {
		workdir = filepath.Join(ret.baseDir, workdir)
	}

	// Parse command
	parts := strings.Fields(command)
	if len(parts) == 0 {
		return ToolResult{
			Success: false,
			Error:   fmt.Errorf("invalid command"),
		}, nil
	}

	// Execute command
	cmd := exec.Command(parts[0], parts[1:]...)
	cmd.Dir = workdir

	output, err := cmd.CombinedOutput()
	if err != nil {
		return ToolResult{
			Success: false,
			Error:   fmt.Errorf("command failed: %w\nOutput: %s", err, string(output)),
		}, nil
	}

	return ToolResult{
		Success: true,
		Output:  fmt.Sprintf("Executed: %s\n%s", command, string(output)),
		Metadata: map[string]interface{}{
			"command": command,
			"workdir": workdir,
		},
	}, nil
}

// VerifyFilesTool verifies that files exist in a directory
type VerifyFilesTool struct {
	baseDir string
}

// NewVerifyFilesTool creates a new file verification tool
func NewVerifyFilesTool(baseDir string) *VerifyFilesTool {
	return &VerifyFilesTool{
		baseDir: baseDir,
	}
}

func (vft *VerifyFilesTool) Name() string {
	return "verify_files"
}

func (vft *VerifyFilesTool) Description() string {
	return "Verifies that specified files exist in a directory"
}

func (vft *VerifyFilesTool) Schema() ToolSchema {
	return ToolSchema{
		Parameters: map[string]ParameterDef{
			"folder": {
				Type:        "string",
				Description: "Folder to check",
			},
			"files": {
				Type:        "array",
				Description: "List of files to verify",
			},
		},
		Required: []string{"folder", "files"},
	}
}

func (vft *VerifyFilesTool) Execute(args map[string]interface{}) (ToolResult, error) {
	folder, ok := args["folder"].(string)
	if !ok || folder == "" {
		return ToolResult{
			Success: false,
			Error:   fmt.Errorf("folder is required"),
		}, nil
	}

	filesInterface, ok := args["files"].([]interface{})
	if !ok {
		// Try string array
		filesSlice, ok := args["files"].([]string)
		if !ok {
			return ToolResult{
				Success: false,
				Error:   fmt.Errorf("files must be an array"),
			}, nil
		}
		filesInterface = make([]interface{}, len(filesSlice))
		for i, f := range filesSlice {
			filesInterface[i] = f
		}
	}

	targetDir := filepath.Join(vft.baseDir, folder)
	missing := []string{}
	found := []string{}

	for _, fileInterface := range filesInterface {
		filename := fmt.Sprintf("%v", fileInterface)
		filePath := filepath.Join(targetDir, filename)

		if _, err := os.Stat(filePath); os.IsNotExist(err) {
			missing = append(missing, filename)
		} else {
			found = append(found, filename)
		}
	}

	if len(missing) > 0 {
		return ToolResult{
			Success: true, // Success means verification ran, not that all files exist
			Output: fmt.Sprintf("Verified: %d found, %d missing. Found: %s. Missing: %s",
				len(found), len(missing), strings.Join(found, ", "), strings.Join(missing, ", ")),
			Metadata: map[string]interface{}{
				"found":   found,
				"missing": missing,
			},
		}, nil
	}

	return ToolResult{
		Success: true,
		Output:  fmt.Sprintf("All %d files verified: %s", len(found), strings.Join(found, ", ")),
		Metadata: map[string]interface{}{
			"found": found,
		},
	}, nil
}

// RealDeleteFolderTool deletes a folder
type RealDeleteFolderTool struct {
	vfs     *semantic.VFSTree
	baseDir string
}

// NewRealDeleteFolderTool creates a new delete folder tool
func NewRealDeleteFolderTool(vfs *semantic.VFSTree, baseDir string) *RealDeleteFolderTool {
	return &RealDeleteFolderTool{
		vfs:     vfs,
		baseDir: baseDir,
	}
}

func (rdt *RealDeleteFolderTool) Name() string {
	return "delete_folder"
}

func (rdt *RealDeleteFolderTool) Description() string {
	return "Deletes a folder and its contents"
}

func (rdt *RealDeleteFolderTool) Schema() ToolSchema {
	return ToolSchema{
		Parameters: map[string]ParameterDef{
			"name": {
				Type:        "string",
				Description: "Name of the folder to delete",
			},
		},
		Required: []string{"name"},
	}
}

func (rdt *RealDeleteFolderTool) Execute(args map[string]interface{}) (ToolResult, error) {
	folder, ok := args["name"].(string)
	if !ok || folder == "" {
		return ToolResult{
			Success: false,
			Error:   fmt.Errorf("folder name is required"),
		}, nil
	}

	targetDir := filepath.Join(rdt.baseDir, folder)

	if _, err := os.Stat(targetDir); os.IsNotExist(err) {
		return ToolResult{
			Success: true,
			Output:  fmt.Sprintf("Folder already deleted: %s", targetDir),
		}, nil
	}

	if err := os.RemoveAll(targetDir); err != nil {
		return ToolResult{
			Success: false,
			Error:   err,
		}, nil
	}

	// Delete from VFS
	vfsPath := "/" + folder
	if err := rdt.vfs.Delete(vfsPath); err != nil {
		// This is not a critical error, so we just log it
		// return ToolResult{
		// 	Success: false,
		// 	Error:   fmt.Errorf("failed to delete from VFS: %w", err),
		// }, nil
	}

	return ToolResult{
		Success: true,
		Output:  fmt.Sprintf("Deleted folder: %s", targetDir),
	}, nil
}
