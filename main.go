package main

import (
	"fmt"

	"github.com/golangast/nlptagger/tagger"
)

func main() {

	t := tagger.Tagging("Create a database named Inventory with the data structure Product containing 2 string fields and 1 integer field.")
	for i := range t.Tokens {
		fmt.Printf("Tokens: %s\t\t\t Ner: %s\t\t Pos: %s\t\t PhraseTag: %s\t\t  \n", t.Tokens[i], t.NerTag[i], t.PosTag[i], t.PhraseTag[i])
	}
	//**VB** create a **NN** webserver **VBN** named **NNP Doggy** with the **NN** handler **NNP Kitty** that **VBZ** has the **NN** data structure **NNP Moose** with **CD** 4 **NN** string fields
	//t := modelname.Trainer("create a webserver named doggy with the handler kitty that has the data structure moose with 4 string fields")
	// for i := range t.Tokens {
	// 	if t.NerTag[i] != "" { // Check if the Ner tag has a value
	// 		fmt.Printf("Epoch %d, Cost: %f, isName: %t, Tokens: %s, Ner: %s,Pos: %s, PhraseTag: %s,  features %v\n", i, t.Cost, t.IsName, t.Tokens[i], t.NerTag[i], t.PosTag[i], t.PhraseTag[i], t.Features[i])
	// 	}
	// }
}
