package workflow

import (
	"fmt"
	"os"
)

// RegisterDefaultMoveRules registers rules for move operations.
func (re *RuleEngine) RegisterDefaultMoveRules() {
	re.RegisterRule(
		RuleCondition{Operation: "MOVE", ResourceType: "Filesystem::File"},
		func(node *Node) error {
			sourcePath := node.Resource.Name
			destinationPath := node.Resource.Destination
			if err := os.Rename(sourcePath, destinationPath); err != nil {
				return fmt.Errorf("failed to move file from %s to %s: %w", sourcePath, destinationPath, err)
			}
			return nil
		},
	)
}
