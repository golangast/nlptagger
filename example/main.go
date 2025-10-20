package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"nlptagger/neural/parser"
	"nlptagger/neural/workflow"
)

var (
	query = flag.String("query", "", "Natural language query for the parser")
)

func main() {
	flag.Parse()

	// Create parser and executor instances
	p := parser.NewParser()
	executor := workflow.NewExecutor()

	// Process initial query from flag, if provided
	if *query != "" {
		processAndExecuteQuery(*query, p, executor)
	}

	// Start interactive loop
	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("\nEnter a query (e.g., \"create folder jack with a go webserver jill\"): ")
		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)

		if input == "exit" || input == "quit" {
			break
		}

		if input != "" {
			processAndExecuteQuery(input, p, executor)
		}
	}
}

func processAndExecuteQuery(q string, p *parser.Parser, executor *workflow.Executor) {
	log.Printf("Processing query: \"%s\"", q)

	// Parse the query into a workflow
	// The parser now handles semantic validation and inference internally.
	wf, err := p.Parse(q)
	if err != nil {
		log.Printf("Error parsing query: %v", err)
		return
	}

	fmt.Println("\n--- Generated Workflow (after inference and validation) ---")
	for _, node := range wf.Nodes {
		fmt.Printf("Node ID: %s, Operation: %s, Resource Type: %s, Resource Name: %s, Properties: %v, Command: %s, Dependencies: %v\n",
			node.ID, node.Operation, node.Resource.Type, node.Resource.Name, node.Resource.Properties, node.Command, node.Dependencies)
	}

	// Execute the generated workflow
	if err := executor.ExecuteWorkflow(wf); err != nil {
		log.Printf("Error executing workflow: %v", err)
		return
	}
}
