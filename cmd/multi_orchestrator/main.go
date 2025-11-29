package main

import (
	"bufio"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"
)

// Simple task result struct
type TaskResult struct {
	Name   string
	Output string
	Err    error
}

func main() {
	fmt.Println("=== Multi-Agent Orchestrator ===")
	fmt.Println("Enter goal (e.g., create webserver):")
	reader := bufio.NewReader(os.Stdin)
	goal, _ := reader.ReadString('\n')
	goal = strings.TrimSpace(goal)
	if goal == "" {
		fmt.Println("No goal provided. Exiting.")
		return
	}
	// Decompose goal (very naive parsing)
	fmt.Printf("Decomposing goal: %s\n", goal)

	var customHandlerName string
	if strings.Contains(goal, "handler") {
		// Attempt to extract the custom handler name
		// Example: "create MyCustomhandler" -> "MyCustom"
		// Example: "create handler" -> ""
		// Example: "MyHandler" -> "My"
		handlerKeywordIndex := strings.LastIndex(goal, "handler")
		if handlerKeywordIndex != -1 {
			// Search backwards from 'handler' to find the start of the name
			nameStartIndex := -1
			for i := handlerKeywordIndex - 1; i >= 0; i-- {
				if goal[i] == ' ' {
					nameStartIndex = i + 1
					break
				}
			}
			if nameStartIndex == -1 && handlerKeywordIndex > 0 { // No space found, name starts from beginning
				nameStartIndex = 0
			}

			if nameStartIndex != -1 {
				potentialName := goal[nameStartIndex:handlerKeywordIndex]
				potentialName = strings.TrimSpace(potentialName)
				if len(potentialName) > 0 {
					customHandlerName = strings.ToUpper(potentialName[:1]) + potentialName[1:]
				}
			}
		}
		if customHandlerName == "" {
			customHandlerName = "Custom" // Default if no specific name found
		}
	}


	// For demonstration we assume any goal results in three tasks.
	var wg sync.WaitGroup
	results := make(chan TaskResult, 3)

	// Coder Agent – writes main.go for a simple HTTP server
	wg.Add(1)
	go func() {
		defer wg.Done()
		var err error
		out := "Go server written"
		if strings.Contains(goal, "handler") {
			err = writeHandlers(customHandlerName) // Pass customHandlerName
			if err == nil {
				out = fmt.Sprintf("Go server with %sHandler written to generated_projects/project/server.go", customHandlerName)
			}
		} else {
			err = writeGoServer()
			if err == nil {
				out = "Go server written to generated_projects/project/server.go"
			}
		}
		if err != nil {
			out = ""
		}
		results <- TaskResult{Name: "Coder", Output: out, Err: err}
	}()

	// DevOps Agent – writes Dockerfile
	wg.Add(1)
	go func() {
		defer wg.Done()
		err := writeDockerfile()
		out := "Dockerfile written"
		if err != nil {
			out = ""
		}
		results <- TaskResult{Name: "DevOps", Output: out, Err: err}
	}()

	// README Agent – writes README.md
	wg.Add(1)
	go func() {
		defer wg.Done()
		err := writeReadme()
		out := "README written"
		if err != nil {
			out = ""
		}
		results <- TaskResult{Name: "Readme", Output: out, Err: err}
	}()

	// Wait for all generation tasks
	wg.Wait()
	close(results)

	// Collect generation results
	fmt.Println("--- Generation Results ---")
	for r := range results {
		if r.Err != nil {
			fmt.Printf("%s failed: %v\n", r.Name, r.Err)
		} else {
			fmt.Printf("%s succeeded: %s\n", r.Name, r.Output)
		}
	}

	// QA Agent – run go test (if any) and build Docker image
	fmt.Println("--- QA Phase ---")
	if err := runTestsAndBuild(goal, customHandlerName); err != nil { // Pass customHandlerName
		fmt.Printf("QA failed: %v\n", err)
		fmt.Println("Triggering re-run of generation tasks... (simplified)")
		// In a real system we would loop back, but for brevity we just exit.
		return
	}
	fmt.Println("All checks passed. Project ready!")
}

func writeGoServer() error {
	content := `package main

import (
    "fmt"
    "log"
    "net/http"
)

func helloHandler(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintln(w, "Hello, World!")
}

func main() {
    http.HandleFunc("/", helloHandler)
    log.Println("Starting server on :8080")
    if err := http.ListenAndServe(":8080", nil); err != nil {
        log.Fatalf("Server failed: %v", err)
    }
}`
	// Ensure target directory exists
	os.MkdirAll("generated_projects/project", 0755)
	return os.WriteFile("generated_projects/project/server.go", []byte(content), 0644)
}

func writeHandlers(handlerName string) error {
	// Dynamically create the handler function name and path
	dynamicHandlerFuncName := strings.ToLower(handlerName) + "Handler"
	dynamicHandlerPath := "/" + strings.ToLower(handlerName)

	content := fmt.Sprintf(`package main

import (
	"fmt"
	"log"
	"net/http"
)

func helloHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintln(w, "Hello, World!")
}

func %s(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintln(w, "%s received!")
}

func main() {
	http.HandleFunc("/", helloHandler)
	http.HandleFunc("%s", %s)
	log.Println("Starting server on :8080")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}`, dynamicHandlerFuncName, handlerName, dynamicHandlerPath, dynamicHandlerFuncName)
	// Ensure target directory exists
	os.MkdirAll("generated_projects/project", 0755)
	return os.WriteFile("generated_projects/project/server.go", []byte(content), 0644)
}

func writeDockerfile() error {
	content := `FROM golang:1.22-alpine AS builder
WORKDIR /app
COPY . .
RUN go build -o webserver server.go

FROM alpine:latest
WORKDIR /app
COPY --from=builder /app/webserver .
EXPOSE 8080
CMD ["./webserver"]`
	os.MkdirAll("generated_projects/project", 0755)
	return os.WriteFile("generated_projects/project/Dockerfile", []byte(content), 0644)
}

func writeReadme() error {
	content := "# Simple Go Webserver\n\nThis project contains a minimal Go HTTP server and a Dockerfile to containerize it.\n\n## Build & Run\n\n```sh\ngo run server.go\n```\n\nOr with Docker:\n\n```sh\ndocker build -t go-webserver .\n docker run -p 8080:8080 go-webserver\n```\n"
	os.MkdirAll("generated_projects/project", 0755)
	return os.WriteFile("generated_projects/project/README.md", []byte(content), 0644)
}

func runTestsAndBuild(goal string, customHandlerName string) error { // Update signature
	fmt.Println("Building Go server...")
	buildCmd := exec.Command("go", "build", "-o", "generated_projects/project/webserver", "generated_projects/project/server.go")
	if output, err := buildCmd.CombinedOutput(); err != nil {
		return fmt.Errorf("failed to build Go server: %v\n%s", err, output)
	}

	fmt.Println("Running Go server...")
	runCmd := exec.Command("./generated_projects/project/webserver")
	if err := runCmd.Start(); err != nil {
		return fmt.Errorf("failed to run Go server: %v", err)
	}
	defer runCmd.Process.Kill()

	// Give the server a moment to start
	time.Sleep(2 * time.Second)

	if strings.Contains(goal, "handler") {
		dynamicHandlerPath := "/" + strings.ToLower(customHandlerName)
		fmt.Printf("Testing %s endpoint...\n", dynamicHandlerPath)
		resp, err := http.Get("http://localhost:8080" + dynamicHandlerPath)
		if err != nil {
			return fmt.Errorf("failed to connect to %s endpoint: %v", dynamicHandlerPath, err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			return fmt.Errorf("unexpected status code for %s: got %d, want %d", dynamicHandlerPath, resp.StatusCode, http.StatusOK)
		}
		fmt.Printf("%s endpoint test passed.\n", dynamicHandlerPath)
	}

	fmt.Println("QA build and test successful.")
	return nil
}
