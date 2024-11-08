package dependencyrelation

import "github.com/golangast/nlptagger/tagger/tag"

// Dependency represents a dependency relation.
type Dependency struct {
	Head      int
	Dependent int
	Relation  string
}

// PredictDependencies simulates predicting dependencies using tag.Tag.
func PredictDependencies(t tag.Tag) (tag.Tag, error) {
	var dependencies []Dependency
	// Identify root (main verb)
	rootIndex := -1
	for i := range t.Tokens {
		if t.PosTag[i] == "VB" || t.PosTag[i] == "VBP" { // Verb or Verb, Present
			rootIndex = i
			break
		}
	}

	if rootIndex != -1 {
		dependencies = append(dependencies, Dependency{Head: -1, Dependent: rootIndex, Relation: "root"})
		// Specific condition for "Create Inventory" relation
		if t.Tokens[rootIndex+1] == "Inventory" && t.NerTag[rootIndex+1] == "OBJECT_NAME" {
			dependencies = append(dependencies, Dependency{Head: rootIndex, Dependent: rootIndex + 1, Relation: "dobj"})
		}
	}
	// Example: Rule-based prediction using UD labels
	for i := range t.Tokens {
		switch {
		case t.PhraseTag[i] == "command:generate_a_webserver" && t.PosTag[i] == "COMMAND_VERB":
			// Set dependency tag to "root"
			dependencies = append(dependencies, Dependency{Head: -1, Dependent: i, Relation: "root"})

			// Change NER tag to "COMMAND"
			t.NerTag[i] = "COMMAND"
		}
		if i >= 1 && i < len(t.NerTag) {

			switch {

			// Generate a webserver case (using PhraseTag and NER)
			case t.PhraseTag[i] == "command:generate_a_webserver" && t.NerTag[i] == "ACTION":
				dependencies = append(dependencies, Dependency{Head: -1, Dependent: i, Relation: "root"})
				// Find the index of the object (webserver) using NER
				for j := i + 1; j < len(t.NerTag); j++ {
					if t.NerTag[j] == "OBJECT_TYPE" {
						dependencies = append(dependencies, Dependency{Head: i, Dependent: j, Relation: "dobj"})
						break
					}
				}

			// "that has the handler named dog" case (using POS, NER, PhraseTag)
			case t.PosTag[i] == "WDT": // "that"
				dependencies = append(dependencies, Dependency{Head: i - 1, Dependent: i, Relation: "nsubj"})
			case t.PosTag[i] == "VBG" && t.NerTag[i] == "AUXILIARY_VERB": // "has"
				dependencies = append(dependencies, Dependency{Head: i - 1, Dependent: i, Relation: "aux"})

			case t.PosTag[i] == "VBN" && t.NerTag[i] == "ACTION": // "named"
				dependencies = append(dependencies, Dependency{Head: i - 1, Dependent: i, Relation: "amod"})

			// "with the data structure" case (using POS, NER)
			case t.PosTag[i] == "IN" && t.NerTag[i] == "RELATION": // "with"
				dependencies = append(dependencies, Dependency{Head: i - 1, Dependent: i, Relation: "case"})

			case i > 1 && t.NerTag[i] == "OBJECT_NAME" && t.NerTag[i-1] == "OBJECT_TYPE":
				dependencies = append(dependencies, Dependency{Head: i - 1, Dependent: i, Relation: "dobj"})

			case i > 1 && t.NerTag[i] == "OBJECT_TYPE" && t.NerTag[i-1] == "ACTION":
				dependencies = append(dependencies, Dependency{Head: i - 1, Dependent: i, Relation: "dobj"})
			case t.NerTag[i] == "ACTION" && t.PosTag[i] == "COMMAND_VERB":
				dependencies = append(dependencies, Dependency{Head: i - 1, Dependent: i, Relation: "root"})

			case t.PosTag[i] == "VBG" && t.Tokens[i] == "has":
				dependencies = append(dependencies, Dependency{Head: i - 1, Dependent: i, Relation: "aux"})
			case t.PosTag[i] == "VBN" && t.Tokens[i] == "named":
				dependencies = append(dependencies, Dependency{Head: i - 1, Dependent: i, Relation: "amod"})
			case t.NerTag[i] == "NAME" && t.PosTag[i] == "NN":
				dependencies = append(dependencies, Dependency{Head: i - 1, Dependent: i, Relation: "dobj"})
			case t.PhraseTag[i] == "handlerName:dog" && t.Tokens[i] == "handler":
				// Find index of "dog" and link them
				for j := i + 1; j < len(t.Tokens); j++ {
					if t.Tokens[j] == "dog" {
						dependencies = append(dependencies, Dependency{Head: i - 1, Dependent: i, Relation: "handlerName"})
						break
					}
				}
			case t.NerTag[i] == "OBJECT_TYPE" && t.PosTag[i] == "NN":
				dependencies = append(dependencies, Dependency{Head: i - 1, Dependent: i, Relation: "nsubj"})
			case i > 0 && t.Tokens[i] == "DATABASE" || t.Tokens[i] == "Database" || t.Tokens[i] == "database" || t.Tokens[i] == "db" || t.Tokens[i] == "DB" || t.Tokens[i] == "data":
				dependencies = append(dependencies, Dependency{Head: i - 1, Dependent: i, Relation: "NOUN"})
			case rootIndex != -1 && t.Tokens[i] == "Inventory" && t.NerTag[i] == "OBJECT_NAME": // Direct object of the root verb (Create Inventory)
				dependencies = append(dependencies, Dependency{Head: rootIndex, Dependent: i, Relation: "dobj"})
			case rootIndex != -1: // Explicit root dependency if rootIndex is valid
				dependencies = append(dependencies, Dependency{Head: -1, Dependent: rootIndex, Relation: "root"})
			case t.Tokens[i] == "with":
				// Find the object of "with"
				for j := i + 1; j < len(t.Tokens); j++ {
					if t.NerTag[j] == "OBJECT_NAME" || t.PosTag[j] == "NN" || t.PosTag[j] == "NNS" {
						dependencies = append(dependencies, Dependency{Head: i, Dependent: j, Relation: "nmod:with"})
						break // Assume only one direct object for "with"
					}
				}
				//Add a case for "containing" which connects Product to data structure
			case i > 0 && t.Tokens[i] == "containing":
				dependencies = append(dependencies, Dependency{Head: i - 1, Dependent: i, Relation: "case"})
				//Add a case for "2" which modifies string
			case i > 0 && t.Tokens[i] == "2" && t.Tokens[i+1] == "string":
				dependencies = append(dependencies, Dependency{Head: i + 1, Dependent: i, Relation: "nummod"})
				//Add a case for "string" which is the object type
			case i > 0 && len(t.Tokens[i])-1 >= i && t.Tokens[i] == "string" && t.Tokens[i+1] == "fields":
				dependencies = append(dependencies, Dependency{Head: i + 1, Dependent: i, Relation: "amod"})
				//Add a case for fields, which is the object type and head of fields
			case i > 0 && t.Tokens[i] == "fields" && (t.Tokens[i-1] == "string" || t.Tokens[i+1] == "and"):
				dependencies = append(dependencies, Dependency{Head: i, Dependent: i, Relation: "nmod"})
			//add a rule for and
			case t.Tokens[i] == "and":
				dependencies = append(dependencies, Dependency{Head: i - 1, Dependent: i, Relation: "cc"})
				dependencies = append(dependencies, Dependency{Head: i - 1, Dependent: i + 2, Relation: "conj"})
			//add a rule for integer
			case i > 0 && t.Tokens[i] == "integer" && t.Tokens[i+1] == "field":
				dependencies = append(dependencies, Dependency{Head: i + 1, Dependent: i, Relation: "amod"})
				//add a rule for field (integer field)
			case i > 0 && t.Tokens[i] == "field" && t.Tokens[i-1] == "integer":
				dependencies = append(dependencies, Dependency{Head: 14, Dependent: i, Relation: "conj"})
				//add a rule for 1 that modifies integer
			case i > 0 && t.Tokens[i] == "1" && t.Tokens[i+1] == "integer":
				dependencies = append(dependencies, Dependency{Head: i + 1, Dependent: i, Relation: "nummod"})

			case t.PosTag[i] == "IN": // Preposition
				// Find the object of the preposition (usually a noun phrase)
				for j := i + 1; j < len(t.Tokens); j++ {
					if t.PosTag[j] == "NN" || t.PosTag[j] == "NNS" {
						dependencies = append(dependencies, Dependency{Head: i, Dependent: j, Relation: "pobj"})
						break
					}
				}
			case t.PosTag[i] == "DT": // Determiner
				// Find the next noun
				nounIndex := -1
				for j := i + 1; j < len(t.Tokens); j++ {
					if t.PosTag[j] == "NN" || t.PosTag[j] == "NNS" || t.NerTag[j] == "OBJECT_NAME" || t.NerTag[j] == "OBJECT_TYPE" {
						nounIndex = j
						break
					}
				}
				if nounIndex != -1 {
					dependencies = append(dependencies, Dependency{Head: nounIndex, Dependent: i, Relation: "det"})
				}

			case rootIndex != -1 && t.NerTag[i] == "OBJECT_NAME" && i > rootIndex: // Direct object of the root verb
				dependencies = append(dependencies, Dependency{Head: rootIndex, Dependent: i, Relation: "dobj"})
			case i > 0 && t.PosTag[i] == "NN" && t.PosTag[i-1] == "VB" && rootIndex != i-1: // Subject for non-root verbs
				dependencies = append(dependencies, Dependency{Head: i - 1, Dependent: i, Relation: "nsubj"})
			case i > 0 && t.PosTag[i] == "NN" && t.PosTag[i-1] == "VB":
				dependencies = append(dependencies, Dependency{Head: i - 1, Dependent: i, Relation: "nsubj"})
			case i > 0 && t.NerTag[i] == "OBJECT_NAME" && t.NerTag[i-1] == "ACTION":
				dependencies = append(dependencies, Dependency{Head: i - 1, Dependent: i, Relation: "dobj"})
			case i > 0 && t.PosTag[i] == "NN" && t.PosTag[i-1] == "DET":
				dependencies = append(dependencies, Dependency{Head: i - 1, Dependent: i, Relation: "det"})
			case i > 0 && t.PosTag[i] == "VB" && t.PosTag[i-1] == "RB":
				dependencies = append(dependencies, Dependency{Head: i - 1, Dependent: i, Relation: "advmod"})
			case i > 1 && t.NerTag[i] == "OBJECT_NAME" && t.NerTag[i-1] == "RELATION":
				dependencies = append(dependencies, Dependency{Head: i - 1, Dependent: i, Relation: "pobj"})
				// Link prepositional phrase to the word it modifies (simplified)
			case i > 0 && t.PosTag[i] == "NN" && t.PosTag[i-1] == "IN":
				// Check if the preposition is part of a prepositional phrase (PP)
				if i > 1 && t.PhraseTag[i-1] == "PP" {
					dependencies = append(dependencies, Dependency{Head: i - 2, Dependent: i, Relation: "pobj"}) // Assuming the head is two words before
				}

			case i > 0 && t.PosTag[i] == "NN" && t.PosTag[i-1] == "VB":
				dependencies = append(dependencies, Dependency{
					Head:      i - 1,
					Dependent: i,
					Relation:  "nsubj",
				})
			case i > 0 && t.Tokens[i] == "and" && t.PosTag[i] == "CC":
				// Check if previous token is string fields and next token is integer field
				if i > 1 && t.Tokens[i-1] == "fields" && t.PosTag[i-1] == "NNS" && i < len(t.Tokens)-1 && t.Tokens[i+1] == "field" && t.PosTag[i+1] == "NN" {
					dependencies = append(dependencies, Dependency{Head: i - 1, Dependent: i, Relation: "cc"})       // coordinating conjunction
					dependencies = append(dependencies, Dependency{Head: i - 1, Dependent: i + 1, Relation: "conj"}) // conjunct
				}

			case i > 0 && t.PosTag[i] == "NN" && t.PosTag[i-1] == "VB" && t.NerTag[i] == "OBJECT_NAME":
				dependencies = append(dependencies, Dependency{Head: i - 1, Dependent: i, Relation: "dobj"})

			case i > 0 && t.PosTag[i] == "NN" && t.PosTag[i-1] == "JJ":
				dependencies = append(dependencies, Dependency{
					Head:      i - 1,
					Dependent: i,
					Relation:  "amod",
				})
			case i > 0 && t.NerTag[i] == "ORG" && t.NerTag[i-1] == "PER":
				dependencies = append(dependencies, Dependency{
					Head:      i - 1,
					Dependent: i,
					Relation:  "nsubj",
				})
			case i > 0 && t.NerTag[i] == "OBJECT_NAME" && t.NerTag[i-1] == "COMMAND":
				dependencies = append(dependencies, Dependency{
					Head:      i - 1,
					Dependent: i,
					Relation:  "dobj",
				})
			case i > 0 && t.NerTag[i] == "FIELDS" && t.NerTag[i-1] == "OBJECT_NAME":
				for j := i - 1; j >= 0; j-- {
					if t.NerTag[j] == "DATA_TYPE" && t.Tokens[j] == "integer" {
						dependencies = append(dependencies, Dependency{Head: j, Dependent: i, Relation: "integer_field"})
						break
					}
				}
			case i > 0 && t.NerTag[i] == "ARGUMENT" && t.NerTag[i-1] == "COMMAND":
				dependencies = append(dependencies, Dependency{
					Head:      i - 1,
					Dependent: i,
					Relation:  "dobj",
				})
			case i > 0 && t.PhraseTag[i] == "ADVP" && t.PhraseTag[i-1] == "VP":
				dependencies = append(dependencies, Dependency{
					Head:      i - 1,
					Dependent: i,
					Relation:  "advmod",
				})

			case i > 0 && t.NerTag[i] == "DATABASE_NAME" && t.NerTag[i-1] == "DATABASE":
				dependencies = append(dependencies, Dependency{Head: i - 1, Dependent: i, Relation: "NOUN"})
			case i > 0 && t.NerTag[i] == "DATA_STRUCTURE_NAME" && t.NerTag[i-1] == "DATABASE":
				dependencies = append(dependencies, Dependency{Head: i - 1, Dependent: i, Relation: "has_data_structure"})
			case i > 0 && t.NerTag[i] == "FIELD" && t.PosTag[i-1] == "CD":
				dependencies = append(dependencies, Dependency{Head: i - 1, Dependent: i, Relation: "nummod"})
			// Database name
			case i > 0 && t.NerTag[i] == "OBJECT_NAME" && t.NerTag[i-1] == "ACTION" && t.Tokens[i] == "Inventory":
				dependencies = append(dependencies, Dependency{Head: i - 1, Dependent: i, Relation: "named"})
			// Data structure
			case i > 0 && t.NerTag[i] == "OBJECT_NAME" && t.NerTag[i-1] == "OBJECT_TYPE" && t.Tokens[i] == "Product":
				dependencies = append(dependencies, Dependency{Head: i - 1, Dependent: i, Relation: "has_data_structure"})
			case t.NerTag[i] == "OBJECT_TYPE" && t.NerTag[i-1] == "DETERMINER":
				dependencies = append(dependencies, Dependency{Head: i, Dependent: i - 1, Relation: "det"})
				// String fields
			case i > 0 && t.NerTag[i] == "DATA_STRUCTURE_FIELD" && t.NerTag[i-1] == "DATA_TYPE" && t.Tokens[i-1] == "string":
				dependencies = append(dependencies, Dependency{Head: i - 1, Dependent: i, Relation: "string_fields"})
			// Integer field
			case i > 0 && t.NerTag[i] == "OBJECT_TYPE" && t.NerTag[i-1] == "DATA_TYPE" && t.Tokens[i-1] == "integer":
				dependencies = append(dependencies, Dependency{Head: i - 1, Dependent: i, Relation: "integer_field"})
			case i < len(t.Tokens)-1 && t.NerTag[i] == "COMMAND":
				dependencies = append(dependencies, Dependency{Head: i, Dependent: i + 1, Relation: "command"})
			case i > 0 && t.Tokens[i-1] == "data" && t.Tokens[i] == "structure":
				dependencies = append(dependencies, Dependency{Head: i, Dependent: i - 1, Relation: "compound"})
			case t.Tokens[i] == "a" && t.PosTag[i] == "DET":
				dependencies = append(dependencies, Dependency{Head: i + 1, Dependent: i, Relation: "det"})
			case t.Tokens[i] == "the" && t.PosTag[i] == "DET":
				dependencies = append(dependencies, Dependency{Head: i + 1, Dependent: i, Relation: "det"})
			case t.Tokens[i] == "named" && t.PosTag[i] == "JJ" || t.PosTag[i] == "NN":
				dependencies = append(dependencies, Dependency{Head: i + 1, Dependent: i, Relation: "det"})

			default:
				dependencies = append(dependencies, Dependency{Head: i, Dependent: i, Relation: " "})
			}
		}
	}
	// Add explicit root dependency if rootIndex is valid
	if rootIndex != -1 {
		dependencies = append(dependencies, Dependency{Head: -1, Dependent: rootIndex, Relation: "root"})
	}
	for i := range t.Tokens {
		switch {
		case t.PosTag[i] == "DT": // Determiner
			// Find the next noun or noun phrase
			nounIndex := -1
			for j := i + 1; j < len(t.Tokens); j++ {
				if t.PosTag[j] == "NN" || t.PosTag[j] == "NNS" || t.NerTag[j] == "OBJECT_NAME" || t.NerTag[j] == "OBJECT_TYPE" {
					nounIndex = j
					break
				}
			}
			if nounIndex != -1 {
				dependencies = append(dependencies, Dependency{Head: nounIndex, Dependent: i, Relation: "det"})
			}
		}
	}

	t.Dependency = nil
	for i := range dependencies {

		t.Dependency = append(t.Dependency, dependencies[i].Relation)

	}
	if len(t.Tokens) != len(t.Dependency) {
		t.Dependency = append(t.Dependency, " ")
		for i := range t.Dependency {
			if t.Dependency[i] == "" {
				t.Dependency[i] = " "
			}
		}
	}

	return t, nil
}
