package main

import (
	"os"
	"strings"
	"testing"
)

func TestWriteGoServer(t *testing.T) {
	err := writeGoServer()
	if err != nil {
		t.Fatalf("writeGoServer() error = %v", err)
	}

	content, err := os.ReadFile("generated_projects/project/server.go")
	if err != nil {
		t.Fatalf("Failed to read server.go: %v", err)
	}

	if !strings.Contains(string(content), "helloHandler") {
		t.Error("Expected to find helloHandler in server.go")
	}

	// Clean up
	os.RemoveAll("generated_projects")
}

func TestWriteHandlers(t *testing.T) {
	err := writeHandlers("Ping")
	if err != nil {
		t.Fatalf("writeHandlers() error = %v", err)
	}

	content, err := os.ReadFile("generated_projects/project/server.go")
	if err != nil {
		t.Fatalf("Failed to read server.go: %v", err)
	}

	if !strings.Contains(string(content), "helloHandler") {
		t.Error("Expected to find helloHandler in server.go")
	}

	if !strings.Contains(string(content), "pingHandler") {
		t.Error("Expected to find pingHandler in server.go")
	}

	// Clean up
	os.RemoveAll("generated_projects")
}

func TestRunTestsAndBuild(t *testing.T) {
	// Test with "handler" goal
	err := writeHandlers("Ping")
	if err != nil {
		t.Fatalf("writeHandlers() error = %v", err)
	}
	err = runTestsAndBuild("create handler", "Ping")
	if err != nil {
		t.Fatalf("runTestsAndBuild() with handler goal failed: %v", err)
	}

	// Test without "handler" goal
	err = writeGoServer()
	if err != nil {
		t.Fatalf("writeGoServer() error = %v", err)
	}
	err = runTestsAndBuild("create webserver", "")
	if err != nil {
		t.Fatalf("runTestsAndBuild() without handler goal failed: %v", err)
	}

	// Clean up
	os.RemoveAll("generated_projects")
}
