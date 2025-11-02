package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/zendrulat/nlptagger/neural/parser"
	"github.com/zendrulat/nlptagger/neural/workflow"
)

var (
	query = flag.String("query", "", "Natural language query for the executor")
)

func main() {
	flag.Parse()

	// Create parser and executor instances.
	parser := parser.NewParser()
	executor := workflow.NewExecutor()

	// Process initial query from flag, if provided
	if *query != "" {
		processAndExecuteQuery(*query, parser, executor)
	}

	// Start interactive loop
	reader := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("\nEnter a query (e.g., \"create folder my_app\"): ")
		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)

		if input == "exit" || input == "quit" {
			break
		}

		if input != "" {
			processAndExecuteQuery(input, parser, executor)
		}
	}
}

func processAndExecuteQuery(q string, parser *parser.Parser, executor *workflow.Executor) {
	log.Printf("Processing query: \"%s\"", q)

	wf, err := parser.Parse(q)
	if err != nil {
		log.Printf("Error parsing query: %v", err)
		return
	}

	if err := executor.ExecuteWorkflow(wf); err != nil {
		log.Printf("Error executing workflow: %v", err)
		return
	}

	log.Println("Query processed and executed successfully.")
}
