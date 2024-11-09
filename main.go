package main

import (
	"fmt"
	"strings"

	modeldata "github.com/golangast/nlptagger/nn"
	"github.com/golangast/nlptagger/tagger/tag"
)

func main() {
	//actual model data loaded
	md, err := modeldata.ModelData()
	if err != nil {
		fmt.Println("Error loading or training model:", err)
	}
	// Example prediction
	sentence := "generate a webserver with the handler dog with the data structure people"
	//making prediction
	predictedTags := md.PredictTags(sentence)
	//getting tags
	predictedTagStruct := tag.Tag{
		PosTag: predictedTags, // Assign the predicted POS tags to the PosTag field
	}

	// Print the sentence again for clarity
	fmt.Println("Sentence:", sentence)
	// Print the predicted POS tags in a space-separated format
	fmt.Println("Predicted Tag Types:", strings.Join(predictedTagStruct.PosTag, " "))

}
