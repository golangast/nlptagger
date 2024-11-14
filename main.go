package main

import (
	"fmt"
	"strings"

	modeldata "github.com/golangast/nlptagger/nn"
	"github.com/golangast/nlptagger/tagger/tag"
)

func main() {
	//actual model data loaded
	md, err := modeldata.ModelData("data/training_data.json")
	if err != nil {
		fmt.Println("Error loading or training model:", err)
	}
	// Example prediction
	sentence := "generate a webserver with the handler dog with the data structure people"
	//making prediction
	predictedPosTags, predictedNerTags, predictedPhraseTags, predictedDRTags := md.PredictTags(sentence)
	//getting tags

	predictedTagStruct := tag.Tag{
		PosTag:          predictedPosTags, // Assign the predicted POS tags to the PosTag field
		NerTag:          predictedNerTags,
		PhraseTag:       predictedPhraseTags,
		DepRelationsTag: predictedDRTags,
	}

	// Print the sentence again for clarity
	fmt.Println("Sentence:", sentence)
	// Print the predicted POS tags in a space-separated format
	fmt.Println("Predicted POS Tag Types:", strings.Join(predictedTagStruct.PosTag, " "))
	fmt.Println("Predicted NER Tag Types:", strings.Join(predictedTagStruct.NerTag, " "))
	fmt.Println("Predicted Phrase Tag Types:", strings.Join(predictedTagStruct.PhraseTag, " "))
	fmt.Println("Predicted Dependency Relation Tag Types:", strings.Join(predictedTagStruct.DepRelationsTag, " "))

}
