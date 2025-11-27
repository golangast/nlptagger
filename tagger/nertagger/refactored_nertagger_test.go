package nertagger

import (
	"reflect"
	"testing"
)

func TestRefactoredTagTokens(t *testing.T) {
	testCases := []struct {
		name     string
		tokens   []string
		posTags  []string
		expected []string
	}{
		{
			name:     "Create folder command",
			tokens:   []string{"create", "folder", "my_folder"},
			posTags:  []string{"VB", "NN", "NN"},
			expected: []string{"COMMAND", "OBJECT_TYPE", "NAME"},
		},
		{
			name:     "Delete file command",
			tokens:   []string{"delete", "file", "test.txt"},
			posTags:  []string{"VB", "NN", "NN"},
			expected: []string{"COMMAND", "OBJECT_TYPE", "NAME"},
		},
		{
			name:     "Move file command",
			tokens:   []string{"move", "file", "a.txt", "to", "b.txt"},
			posTags:  []string{"VB", "NN", "NN", "TO", "NN"},
			expected: []string{"COMMAND", "OBJECT_TYPE", "NAME", "PREPOSITION", "NAME"},
		},
		{
			name:     "Command with determiner",
			tokens:   []string{"create", "a", "webserver"},
			posTags:  []string{"VB", "DT", "NN"},
			expected: []string{"COMMAND", "DETERMINER", "OBJECT_TYPE"},
		},
		{
			name:     "Command with name prefix",
			tokens:   []string{"create", "a", "file", "named", "test.txt"},
			posTags:  []string{"VB", "DT", "NN", "VBN", "NN"},
			expected: []string{"COMMAND", "DETERMINER", "OBJECT_TYPE", "NAME_PREFIX", "NAME"},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := RefactoredTagTokens(tc.tokens, tc.posTags)
			if !reflect.DeepEqual(actual, tc.expected) {
				t.Errorf("RefactoredTagTokens() = %v, expected %v", actual, tc.expected)
			}
		})
	}
}
