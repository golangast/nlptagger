package commands

import (
	"fmt"
	"os"
)

func Createfile() {
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
