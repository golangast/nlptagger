package main

// "fmt"

// "github.com/golangast/nlptagger/tagger"

func main() {

	// t := tagger.Tagging("Create a database named Inventory with the data structure Product containing 2 string fields and 1 integer field.")
	// combined := append(t.Tokens, t.PosTag...)
	// table(combined, 2)
	// fmt.Println("Pos is done")
	// fmt.Println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
	// combined2 := append(t.Tokens, t.NerTag...)
	// table(combined2, 2)
	// fmt.Println("Ner is done")
	// fmt.Println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

	// combined3 := append(combined, t.PhraseTag...)
	// table(combined3, 2)

}

// func table(input []string, cols int) {
// 	maxWidth := 0
// 	for _, s := range input {
// 		if len(s) > maxWidth {
// 			maxWidth = len(s)
// 		}
// 	}
// 	format := fmt.Sprintf("%%d.%%-%ds%%s", maxWidth)

// 	rows := (len(input) + cols - 1) / cols
// 	for row := 0; row < rows; row++ {
// 		for col := 0; col < cols; col++ {
// 			i := col*rows + row
// 			if i >= len(input) {
// 				break // This means the last column is not "full"
// 			}
// 			padding := ""
// 			if i < 9 {
// 				padding = " "
// 			}
// 			fmt.Printf(format, i+1, input[i], padding)
// 		}
// 		fmt.Println()
// 	}
// }
