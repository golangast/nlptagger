package main

import (
	"os"
	"path/filepath"
	"strings"
)

type KnowledgeBase struct {
	BasePath string
}

func NewKnowledgeBase(path string) *KnowledgeBase {
	return &KnowledgeBase{BasePath: path}
}

// FindRelevantFiles searches for files in the learning directory that match the query
func (kb *KnowledgeBase) FindRelevantFiles(query string) ([]string, error) {
	var matches []string

	// Check if directory exists
	if _, err := os.Stat(kb.BasePath); os.IsNotExist(err) {
		return matches, nil
	}

	err := filepath.Walk(kb.BasePath, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			// Check if filename matches query keywords
			name := strings.ToLower(info.Name())
			queryWords := strings.Fields(strings.ToLower(query))

			score := 0
			for _, word := range queryWords {
				// Skip common words
				if word == "create" || word == "add" || word == "file" || word == "code" {
					continue
				}
				if strings.Contains(name, word) {
					score++
				}
			}

			// If at least one specific keyword matches (e.g. "handler"), include it
			if score > 0 {
				matches = append(matches, path)
			}
		}
		return nil
	})

	return matches, err
}

// ReadFileContent reads the content of a learned file
func (kb *KnowledgeBase) ReadFileContent(path string) (string, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return string(content), nil
}
