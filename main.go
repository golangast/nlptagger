package main

import (
	"fmt"
	"strings"

	"github.com/golangast/nlptagger/nn"
	"github.com/golangast/nlptagger/tagger/tag"
)

func main() {
	nnn := nn.NN()
	// Example prediction
	sentence := "generate a webserver with the handler dog with the data structure people"
	predictedTags := nn.PredictTags(nnn, sentence)

	predictedTagStruct := tag.Tag{
		PosTag: predictedTags, // Assign the predicted POS tags to the PosTag field
	}

	// Print the sentence again for clarity
	fmt.Println("Sentence:", sentence)
	// Print the predicted POS tags in a space-separated format
	fmt.Println("Predicted Tag Types:", strings.Join(predictedTagStruct.PosTag, " "))

}
