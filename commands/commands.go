package commands

import (
	"fmt"
	"os"
)

// CommandFunc defines the signature for a command function.
type CommandFunc func(args []string)

var commandRegistry = make(map[string]CommandFunc)

// RegisterCommand adds a command to the registry.
func RegisterCommand(name string, fn CommandFunc) {
	commandRegistry[name] = fn
}

// GetCommands returns the command registry.
func GetCommands() map[string]CommandFunc {
	return commandRegistry
}

func init() {
	RegisterCommand("CREATE_FILE", Createfile)
	RegisterCommand("CREATE_WEBSERVER", createWebserver)
	RegisterCommand("CREATE_DATABASE", CreateDatabase)
	RegisterCommand("GENERATE_HANDLER", GenerateHandler)
}

func Createfile(args []string) {
	// Specify the file name
	fileName := "example.txt"

	// Create the file
	file, err := os.Create(fileName)
	if err != nil {
		fmt.Println("Error creating file:", err)
		return
	}
	defer file.Close() // Ensure the file is closed after use

	// Write to the file
	_, err = file.WriteString("Hello, world!\n")
	if err != nil {
		fmt.Println("Error writing to file:", err)
		return
	}

	fmt.Println("File generated successfully:", fileName)
}

func createWebserver(args []string) {
	if len(args) < 4 {
		fmt.Println("Usage: create webserver <name> with handler <handler_name>")
		return
	}
	serverName := args[0]
	handlerName := args[3]
	fmt.Printf("Creating webserver named '%s' with handler '%s'\n", serverName, handlerName)
	// In a real application, you would generate the webserver code here.
}

func CreateDatabase(args []string) {
	fmt.Println("Executing CREATE_DATABASE command.")
	// In a real app, you'd parse args and create a database.
	// For example, look for a name.
	dbName := ""
	for i, v := range args {
		if v == "named" && i+1 < len(args) {
			dbName = args[i+1]
			break
		}
	}
	if dbName != "" {
		fmt.Printf("Database to be created: %s\n", dbName)
	}
}

func GenerateHandler(args []string) {
	fmt.Println("Executing GENERATE_HANDLER command.")
	// In a real app, you'd parse args and create a handler.
	// For example, look for a name.
	handlerName := ""
	for i, v := range args {
		if v == "named" && i+1 < len(args) {
			handlerName = args[i+1]
			break
		}
	}
	if handlerName != "" {
		fmt.Printf("Handler to be generated: %s\n", handlerName)
	}
}
