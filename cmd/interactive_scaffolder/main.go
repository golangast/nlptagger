package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/zendrulat/nlptagger/neural/agent"
)

func main() {
	fmt.Println("=== Advanced NLP Project Scaffolder (Goal-Driven) ===")
	fmt.Println("Intelligent project manager with memory, goals, and context")
	fmt.Println("Commands: 'goal <desc>', 'work [-auto]', 'status', 'goals', 'resume <id>', 'tree', 'exit'\n")

	// Initialize Agent
	config := agent.AgentConfig{
		MaxIterations:     10,
		EnablePersistence: true,
		PersistencePath:   "./agent_memory.json",
		WorkingDirectory:  "./generated_projects",
	}
	ag := agent.NewAgent(config)

	// Current active goal
	var currentGoal *agent.Goal

	scanner := bufio.NewScanner(os.Stdin)

	for {
		if currentGoal != nil {
			fmt.Printf("[%s] > ", currentGoal.ID)
		} else {
			fmt.Print("> ")
		}

		if !scanner.Scan() {
			break
		}

		query := strings.TrimSpace(scanner.Text())
		if query == "" {
			continue
		}

		if query == "exit" || query == "quit" {
			fmt.Println("Goodbye!")
			break
		}

		// Handle commands
		parts := strings.Fields(query)
		command := parts[0]

		switch command {
		case "goal":
			if len(parts) < 2 {
				fmt.Println("Usage: goal <description>")
				continue
			}
			description := strings.Join(parts[1:], " ")
			goal, err := ag.CreateGoalFromDescription(description)
			if err != nil {
				fmt.Printf("Error creating goal: %v\n", err)
				continue
			}
			currentGoal = goal
			fmt.Printf("Goal created: %s (ID: %s)\n", goal.Description, goal.ID)
			fmt.Println(ag.GetGoalStatus(goal))

		case "work":
			if currentGoal == nil {
				fmt.Println("No active goal. Use 'goal <desc>' or 'resume <id>' first.")
				continue
			}

			autoMode := len(parts) > 1 && parts[1] == "-auto"

			if autoMode {
				// Phase 2: Continuous Execution
				fmt.Println("Starting continuous execution...")
				for {
					task, err := ag.WorkOnGoalStep(currentGoal)
					if err != nil {
						fmt.Printf("Error executing step: %v\n", err)
						break
					}
					if task == nil {
						fmt.Println("No more tasks to execute.")
						if currentGoal.Progress >= 1.0 {
							fmt.Println("Goal completed successfully!")
							currentGoal.Status = agent.GoalCompleted
						}
						break
					}
					fmt.Printf("Executed: %s -> %s\n", task.Description, task.Status)

					if currentGoal.Progress >= 1.0 {
						fmt.Println("Goal completed successfully!")
						break
					}
				}
			} else {
				// Single step
				task, err := ag.WorkOnGoalStep(currentGoal)
				if err != nil {
					fmt.Printf("Error executing step: %v\n", err)
					continue
				}
				if task == nil {
					fmt.Println("No more tasks to execute.")
					if currentGoal.Progress >= 1.0 {
						fmt.Println("Goal completed successfully!")
						currentGoal.Status = agent.GoalCompleted
					}
				} else {
					fmt.Printf("Executed: %s -> %s\n", task.Description, task.Status)
					fmt.Printf("Result: %s\n", task.Result)
				}
			}

			// Always show updated status
			fmt.Println(ag.GetGoalStatus(currentGoal))

		case "status":
			if currentGoal == nil {
				fmt.Println("No active goal.")
				continue
			}
			fmt.Println(ag.GetGoalStatus(currentGoal))

		case "goals":
			goals, err := ag.ListGoals()
			if err != nil {
				fmt.Printf("Error listing goals: %v\n", err)
				continue
			}
			fmt.Println("Saved Goals:")
			for _, g := range goals {
				fmt.Printf("- %s: %s (%.0f%%)\n", g.ID, g.Description, g.Progress*100)
			}

		case "resume":
			if len(parts) < 2 {
				fmt.Println("Usage: resume <goal_id>")
				continue
			}
			goalID := parts[1]
			goal, err := ag.GetGoal(goalID)
			if err != nil {
				fmt.Printf("Error loading goal: %v\n", err)
				continue
			}
			currentGoal = goal
			fmt.Printf("Resumed goal: %s\n", goal.Description)
			fmt.Println(ag.GetGoalStatus(goal))

		case "tree":
			fmt.Println(ag.VFS.Tree())

		default:
			fmt.Println("Unknown command. Available: goal, work, status, goals, resume, tree, exit")
		}
	}
}
