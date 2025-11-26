package main

import (
	"fmt"
	"strings"

	"nlptagger/neural/semantic"
)

func main() {
	fmt.Println("=== Advanced Context System Demo ===\n")

	// 1. VFS + Roles
	fmt.Println("1. VFS Tree with Semantic Roles")
	fmt.Println("================================")

	vfs := semantic.NewVFSTree()
	vfs.CreateFolder("/myproject", string(semantic.RoleProjectRoot))
	vfs.CreateFolder("/myproject/src", string(semantic.RoleSourceDirectory))

	fmt.Println(vfs.Tree())

	// 2. Blueprint Engine
	fmt.Println("\n2. Parameterized Blueprints")
	fmt.Println("===========================")

	engine := semantic.NewBlueprintEngine()

	// Execute webserver blueprint with custom parameters
	params := map[string]interface{}{
		"ServerName": "MyAPI",
		"Port":       3000,
	}

	code, _ := engine.Execute("webserver", params)
	fmt.Println("Generated code for 'webserver' with params {ServerName: MyAPI, Port: 3000}:")
	fmt.Println(code)

	// 3. Dependency Graph
	fmt.Println("\n3. Dependency Graph")
	fmt.Println("===================")

	depGraph := semantic.NewDependencyGraph()

	// Add dependencies
	depGraph.AddDependency("/myproject/src/main.go", "/myproject/src/handler.go", semantic.DepImport)
	depGraph.AddDependency("/myproject/src/main.go", "/myproject/config.json", semantic.DepConfig)
	depGraph.AddDependency("/myproject/src/handler.go", "/myproject/src/utils.go", semantic.DepImport)

	fmt.Println(depGraph.Summary())

	// Check for cycles
	hasCycle := depGraph.HasCycle()
	fmt.Printf("Has circular dependencies: %t\n", hasCycle)

	// 4. Auto-Refactoring
	fmt.Println("\n4. Auto-Refactoring on Rename")
	fmt.Println("==============================")

	affected := depGraph.UpdateReferences("/myproject/src/handler.go", "/myproject/src/router.go")
	fmt.Printf("Renamed: handler.go → router.go\n")
	fmt.Printf("Files that need updates: %v\n", affected)
	fmt.Println("\nUpdated dependency graph:")
	fmt.Println(depGraph.Summary())

	// 5. Role-Based Operations
	fmt.Println("\n5. Role-Based Blueprint Mapping")
	fmt.Println("================================")

	roleRegistry := semantic.NewRoleRegistry()

	// Infer role from filename
	role := roleRegistry.InferRole("main.go", "file", "")
	fmt.Printf("Inferred role for 'main.go': %s\n", role)

	// Get blueprint for role
	if role == string(semantic.RoleEntrypoint) {
		fmt.Println("Suggested blueprints for entrypoint: webserver, api")
	}

	// 6. Complete Example
	fmt.Println("\n6. Complete Workflow")
	fmt.Println("====================")

	// Create project with VFS
	vfs.ChangeDirectory("/myproject/src")

	// Generate code with blueprint
	handlerParams := map[string]interface{}{
		"HandlerName": "UserHandler",
	}
	handlerCode, _ := engine.Execute("handler", handlerParams)

	// Create file in VFS with role
	vfs.CreateFile("/myproject/src/user_handler.go", string(semantic.RoleHandler), handlerCode)

	// Add dependency
	depGraph.AddDependency("/myproject/src/main.go", "/myproject/src/user_handler.go", semantic.DepImport)

	fmt.Println("Created user_handler.go with UserHandler blueprint")
	fmt.Println("\nFinal VFS Tree:")
	fmt.Println(vfs.Tree())

	fmt.Println("\nFinal Dependencies:")
	fmt.Println(depGraph.Summary())

	// Summary
	fmt.Println("\n" + strings.Repeat("=", 50))
	fmt.Println("ADVANCED CONTEXT SYSTEM FEATURES:")
	fmt.Println(strings.Repeat("=", 50))
	fmt.Println("✓ VFS Tree with conflict detection")
	fmt.Println("✓ Semantic roles for files/folders")
	fmt.Println("✓ Parameterized blueprints ({{.Variable}} syntax)")
	fmt.Println("✓ Dependency graph with cycle detection")
	fmt.Println("✓ Auto-refactoring on rename")
	fmt.Println("✓ Role-based template mapping")
}
