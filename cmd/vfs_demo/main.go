package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"nlptagger/neural/semantic"
)

func main() {
	fmt.Println("=== VFS Tree Demo ===")
	fmt.Println("Demonstrating persistent state tree with roles and conflict detection")
	fmt.Println()

	// Create VFS
	vfs := semantic.NewVFSTree()
	roleRegistry := semantic.NewRoleRegistry()

	// Demo 1: Create project with roles
	fmt.Println("Demo 1: Creating project structure with roles")
	fmt.Println("----------------------------------------------")

	// Create root folder
	project, _ := vfs.CreateFolder("/myproject", string(semantic.RoleProjectRoot))
	fmt.Printf("Created: %s [%s]\n", project.Path, project.Role)

	// Create source folder
	src, _ := vfs.CreateFolder("/myproject/src", string(semantic.RoleSourceDirectory))
	fmt.Printf("Created: %s [%s]\n", src.Path, src.Role)

	// Create entrypoint
	main, _ := vfs.CreateFile("/myproject/src/main.go", string(semantic.RoleEntrypoint),
		`package main

func main() {
	fmt.Println("Hello, World!")
}`)
	fmt.Printf("Created: %s [%s]\n", main.Path, main.Role)

	// Create handler
	handler, _ := vfs.CreateFile("/myproject/src/handler.go", string(semantic.RoleHandler),
		`package main

func Handler() {
	// Handle request
}`)
	fmt.Printf("Created: %s [%s]\n", handler.Path, handler.Role)

	fmt.Println("\nVFS Tree:")
	fmt.Println(vfs.Tree())

	// Demo 2: Role-based lookup
	fmt.Println("\nDemo 2: Role-based lookup")
	fmt.Println("-------------------------")

	entrypoints := vfs.GetByRole(string(semantic.RoleEntrypoint))
	fmt.Printf("Files with role 'entrypoint': %d\n", len(entrypoints))
	for _, node := range entrypoints {
		fmt.Printf("  - %s\n", node.Path)
	}

	// Demo 3: Conflict detection
	fmt.Println("\nDemo 3: Conflict detection")
	fmt.Println("--------------------------")

	_, err := vfs.CreateFile("/myproject/src/main.go", "", "duplicate content")
	if err != nil {
		fmt.Printf("✓ Conflict detected: %v\n", err)
	}

	// Demo 4: Path resolution
	fmt.Println("\nDemo 4: Path resolution")
	fmt.Println("-----------------------")

	vfs.ChangeDirectory("/myproject/src")
	fmt.Printf("Changed to: %s\n", vfs.CurrentDir.Path)

	resolved, found := vfs.ResolvePath("handler.go")
	if found {
		fmt.Printf("Resolved 'handler.go' to: %s [%s]\n", resolved.Path, resolved.Role)
	}

	// Demo 5: Role inference
	fmt.Println("\nDemo 5: Role inference")
	fmt.Println("----------------------")

	inferredRole := roleRegistry.InferRole("config.json", "file", "")
	fmt.Printf("Inferred role for 'config.json': %s\n", inferredRole)

	inferredRole2 := roleRegistry.InferRole("server.go", "file", "")
	fmt.Printf("Inferred role for 'server.go': %s\n", inferredRole2)

	// Interactive mode
	fmt.Println("\n\n=== Interactive Mode ===")
	fmt.Println("Commands: tree, role <rolename>, find <name>, rename <old> <new>, exit")

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("\nvfs> ")
		if !scanner.Scan() {
			break
		}

		input := strings.TrimSpace(scanner.Text())
		if input == "" {
			continue
		}

		parts := strings.Fields(input)
		cmd := parts[0]

		switch cmd {
		case "exit", "quit":
			fmt.Println("Goodbye!")
			return

		case "tree":
			fmt.Println(vfs.Tree())

		case "role":
			if len(parts) < 2 {
				fmt.Println("Usage: role <rolename>")
				continue
			}
			nodes := vfs.GetByRole(parts[1])
			fmt.Printf("Nodes with role '%s': %d\n", parts[1], len(nodes))
			for _, node := range nodes {
				fmt.Printf("  - %s\n", node.Path)
			}

		case "find":
			if len(parts) < 2 {
				fmt.Println("Usage: find <name>")
				continue
			}
			node, found := vfs.ResolvePath(parts[1])
			if found {
				fmt.Printf("Found: %s [%s]\n", node.Path, node.Role)
			} else {
				fmt.Printf("Not found: %s\n", parts[1])
			}

		case "rename":
			if len(parts) < 3 {
				fmt.Println("Usage: rename <oldpath> <newname>")
				continue
			}
			err := vfs.Rename(parts[1], parts[2])
			if err != nil {
				fmt.Printf("Error: %v\n", err)
			} else {
				fmt.Printf("Renamed %s to %s\n", parts[1], parts[2])
			}

		default:
			fmt.Printf("Unknown command: %s\n", cmd)
		}
	}
}
