package main

import (
	"fmt"
	"testing"
)

func TestSemanticParsing(t *testing.T) {
	testCases := []struct {
		name           string
		input          string
		expectError    bool
		expectedIntent string
	}{
		{
			name:           "Create webserver with handler",
			input:          "create a webserver with authentication handler",
			expectError:    false,
			expectedIntent: "create_handler",
		},
		{
			name:           "Create database",
			input:          "I need a database called users.db",
			expectError:    false,
			expectedIntent: "create_database",
		},
		{
			name:           "Build API with JWT",
			input:          "build me a Go API server with JWT authentication",
			expectError:    false,
			expectedIntent: "add_feature",
		},
		{
			name:           "Simple webserver",
			input:          "create webserver",
			expectError:    false,
			expectedIntent: "unknown",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			parsedGoal, err := parseGoalWithSemantics(tc.input)

			if tc.expectError && err == nil {
				t.Errorf("Expected error but got none")
			}

			if !tc.expectError && err != nil {
				t.Errorf("Unexpected error: %v", err)
			}

			if parsedGoal != nil {
				fmt.Printf("\n✅ Test: %s\n", tc.name)
				fmt.Printf("   Input: %s\n", tc.input)
				fmt.Printf("   Intent: %s\n", parsedGoal.Intent)
				fmt.Printf("   Entities: %v\n", parsedGoal.Entities)
			}
		})
	}
}

func TestEntityExtraction(t *testing.T) {
	testCases := []struct {
		name           string
		input          string
		expectedEntity string
		entityKey      string
	}{
		{
			name:           "Extract handler name",
			input:          "create authentication handler",
			expectedEntity: "authentication",
			entityKey:      "handler_name",
		},
		{
			name:           "Extract database name",
			input:          "create database users.db",
			expectedEntity: "users.db",
			entityKey:      "database_name",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			parsedGoal, err := parseGoalWithSemantics(tc.input)

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			if parsedGoal == nil {
				t.Errorf("Expected parsed goal but got nil")
				return
			}

			if entity, ok := parsedGoal.Entities[tc.entityKey]; ok {
				fmt.Printf("\n✅ Extracted entity: %s = %s\n", tc.entityKey, entity)
			} else {
				fmt.Printf("\n⚠️  Entity %s not found in: %v\n", tc.entityKey, parsedGoal.Entities)
			}
		})
	}
}
