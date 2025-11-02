package dependencyrelation

import "github.com/zendrulat/nlptagger/tagger/tag"

var (
	DRToID = map[string]int{
		"root":               0,
		"dobj":               1,
		"det":                2,
		"nsubj":              3,
		"aux":                4,
		"amod":               5,
		"case":               6,
		"handlerName":        7,
		"NOUN":               8,
		"nmod:with":          9,
		"nummod":             10,
		"nmod":               11,
		"cc":                 12,
		"conj":               13,
		"pobj":               14,
		"advmod":             15,
		"integer_field":      16,
		"command":            17,
		"compound":           18,
		"named":              19,
		"has_data_structure": 20,
		"string_fields":      21,
	}
)

func DRToIDMap() map[string]int {
	return DRToID
}

// Dependency represents a dependency relation.

// PredictDependencies predicts the dependency relationships between tokens in a sentence.
// It uses a combination of POS tags, NER tags, and phrase tags to infer the relationships.
// The function iterates through each token in the input tag.Tag and applies a set of rules
// to determine the head, dependent, and relation for each token.
func PredictDependencies(t tag.Tag) (tag.Tag, error) {
	// Initialize the root index to -1, indicating no root has been found yet.
	rootIndex := -1

	// Check for the specific phrase "generate a webserver" and handle its dependencies.
	for i := 0; i < len(t.Tokens); i++ {
		if t.PhraseTag[i] == "verbPhrase:generate_a_webserver" {
			// Set 'generate' as the root.
			rootIndex = i
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: -1, Dependent: i, Relation: "root"})
			// Find 'webserver' and set it as the direct object of 'generate'.
			for j := i + 1; j < len(t.Tokens); j++ {
				if t.Tokens[j] == "webserver" {
					t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i, Dependent: j, Relation: "dobj"})
					// Find 'a' and set it as the determiner of 'webserver'.
					for k := i; k < j; k++ {
						if t.Tokens[k] == "a" {
							t.Dependencies = append(t.Dependencies, tag.Dependency{Head: j, Dependent: k, Relation: "det"})
						}
					}
				}
			}
		}
	}
	for i := range t.Tokens {
		switch {
		case t.PhraseTag[i] == "command:generate_a_webserver" && t.PosTag[i] == "COMMAND_VERB":
			// Set tag.Dependency tag to "root"
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: -1, Dependent: i, Relation: "root"})

		// Change NER tag to "COMMAND"
		// "that has the handler named dog" case (using POS, NER, PhraseTag)
		case t.PosTag[i] == "WDT": // "that"
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i - 1, Dependent: i, Relation: "nsubj"})
		case t.PosTag[i] == "VBG" && t.NerTag[i] == "AUXILIARY_VERB": // "has"
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i - 1, Dependent: i, Relation: "aux"})

		case t.PosTag[i] == "VBN" && t.NerTag[i] == "ACTION": // "named"
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i - 1, Dependent: i, Relation: "amod"})

		// "with the data structure" case (using POS, NER)
		case t.PosTag[i] == "IN" && t.NerTag[i] == "RELATION": // "with"
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i - 1, Dependent: i, Relation: "case"})

		case i > 1 && t.NerTag[i] == "OBJECT_NAME" && t.NerTag[i-1] == "OBJECT_TYPE":
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i - 1, Dependent: i, Relation: "dobj"})

		case i > 1 && t.NerTag[i] == "OBJECT_TYPE" && t.NerTag[i-1] == "ACTION":
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i - 1, Dependent: i, Relation: "dobj"})
		case t.NerTag[i] == "ACTION" && t.PosTag[i] == "COMMAND_VERB":
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i - 1, Dependent: i, Relation: "root"})

		case t.PosTag[i] == "VBG" && t.Tokens[i] == "has":
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i - 1, Dependent: i, Relation: "aux"})
		case t.PosTag[i] == "VBN" && t.Tokens[i] == "named":
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i - 1, Dependent: i, Relation: "amod"})
		case t.NerTag[i] == "NAME" && t.PosTag[i] == "NN":
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i - 1, Dependent: i, Relation: "dobj"})
		case t.PhraseTag[i] == "handlerName:dog" && t.Tokens[i] == "handler":
			// Find index of "dog" and link them
			for j := i + 1; j < len(t.Tokens); j++ {
				if t.Tokens[j] == "dog" {
					t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i - 1, Dependent: i, Relation: "handlerName"})
					break
				}
			}
		case t.NerTag[i] == "OBJECT_TYPE" && t.PosTag[i] == "NN":
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i - 1, Dependent: i, Relation: "nsubj"})
		case i > 0 && t.Tokens[i] == "DATABASE" || t.Tokens[i] == "Database" || t.Tokens[i] == "database" || t.Tokens[i] == "db" || t.Tokens[i] == "DB" || t.Tokens[i] == "data":
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i - 1, Dependent: i, Relation: "NOUN"})
		case rootIndex != -1 && t.Tokens[i] == "Inventory" && t.NerTag[i] == "OBJECT_NAME": // Direct object of the root verb (Create Inventory)
		case rootIndex != -1: // Explicit root tag.Dependency if rootIndex is valid
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: -1, Dependent: rootIndex, Relation: "root"})
		case t.Tokens[i] == "with":
			// Find the object of "with"
			for j := i + 1; j < len(t.Tokens); j++ {
				if t.NerTag[j] == "OBJECT_NAME" || t.PosTag[j] == "NN" || t.PosTag[j] == "NNS" {
					t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i, Dependent: j, Relation: "nmod:with"})
					break // Assume only one direct object for "with"
				}
			}
			//Add a case for "containing" which connects Product to data structure
		case i > 0 && t.Tokens[i] == "containing":
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i - 1, Dependent: i, Relation: "case"})
			//Add a case for "2" which modifies string
		case i > 0 && t.Tokens[i] == "2" && t.Tokens[i+1] == "string":
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i + 1, Dependent: i, Relation: "nummod"})
			//Add a case for "string" which is the object type
		case i > 0 && len(t.Tokens[i])-1 >= i && t.Tokens[i] == "string" && t.Tokens[i+1] == "fields":
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i + 1, Dependent: i, Relation: "amod"})
			//Add a case for fields, which is the object type and head of fields
		case i > 0 && t.Tokens[i] == "fields" && (t.Tokens[i-1] == "string" || t.Tokens[i+1] == "and"):
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i, Dependent: i, Relation: "nmod"})
		//add a rule for and
		case t.Tokens[i] == "and":
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i - 1, Dependent: i, Relation: "cc"})
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i - 1, Dependent: i + 2, Relation: "conj"})
		//add a rule for integer
		case i > 0 && t.Tokens[i] == "integer" && t.Tokens[i+1] == "field":
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i + 1, Dependent: i, Relation: "amod"})
			//add a rule for field (integer field)
		case i > 0 && t.Tokens[i] == "field" && t.Tokens[i-1] == "integer":
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: 14, Dependent: i, Relation: "conj"})
			//add a rule for 1 that modifies integer
		case i > 0 && t.Tokens[i] == "1" && t.Tokens[i+1] == "integer":
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i + 1, Dependent: i, Relation: "nummod"})

		case t.PosTag[i] == "IN": // Preposition

			// Find the object of the preposition (usually a noun phrase)
			for j := i + 1; j < len(t.Tokens); j++ {
				if t.PosTag[j] == "NN" || t.PosTag[j] == "NNS" {
					t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i, Dependent: j, Relation: "pobj"})
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
				t.Dependencies = append(t.Dependencies, tag.Dependency{Head: nounIndex, Dependent: i, Relation: "det"})
			}

			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i - 1, Dependent: i, Relation: "nsubj"})
		case i > 0 && t.PosTag[i] == "NN" && t.PosTag[i-1] == "VB":
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i - 1, Dependent: i, Relation: "nsubj"})
		case i > 0 && t.NerTag[i] == "OBJECT_NAME" && t.NerTag[i-1] == "ACTION":
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i - 1, Dependent: i, Relation: "dobj"})
		case i > 0 && t.PosTag[i] == "NN" && t.PosTag[i-1] == "DET":
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i - 1, Dependent: i, Relation: "det"})
		case i > 0 && t.PosTag[i] == "VB" && t.PosTag[i-1] == "RB":
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i - 1, Dependent: i, Relation: "advmod"})
		case i > 1 && t.NerTag[i] == "OBJECT_NAME" && t.NerTag[i-1] == "RELATION":
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i - 1, Dependent: i, Relation: "pobj"})
			// Link prepositional phrase to the word it modifies (simplified)
		case i > 0 && t.PosTag[i] == "NN" && t.PosTag[i-1] == "IN":
			// Check if the preposition is part of a prepositional phrase (PP)
			if i > 1 && t.PhraseTag[i-1] == "PP" {
				t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i - 2, Dependent: i, Relation: "pobj"}) // Assuming the head is two words before
			}

		case i > 0 && t.PosTag[i] == "NN" && t.PosTag[i-1] == "VB":
			t.Dependencies = append(t.Dependencies, tag.Dependency{
				Head:      i - 1,
				Dependent: i,
				Relation:  "nsubj",
			})
		case i > 0 && t.Tokens[i] == "and" && t.PosTag[i] == "CC":
			// Check if previous token is string fields and next token is integer field
			if i > 1 && t.Tokens[i-1] == "fields" && t.PosTag[i-1] == "NNS" && i < len(t.Tokens)-1 && t.Tokens[i+1] == "field" && t.PosTag[i+1] == "NN" {
				t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i - 1, Dependent: i, Relation: "cc"})       // coordinating conjunction
				t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i - 1, Dependent: i + 1, Relation: "conj"}) // conjunct
			}

		case i > 0 && t.PosTag[i] == "NN" && t.PosTag[i-1] == "VB" && t.NerTag[i] == "OBJECT_NAME":
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i - 1, Dependent: i, Relation: "dobj"})

		case i > 0 && t.PosTag[i] == "NN" && t.PosTag[i-1] == "JJ":
			t.Dependencies = append(t.Dependencies, tag.Dependency{
				Head:      i - 1,
				Dependent: i,
				Relation:  "amod",
			})
		case i > 0 && t.NerTag[i] == "ORG" && t.NerTag[i-1] == "PER":
			t.Dependencies = append(t.Dependencies, tag.Dependency{
				Head:      i - 1,
				Dependent: i,
				Relation:  "nsubj",
			})
		case i > 0 && t.NerTag[i] == "OBJECT_NAME" && t.NerTag[i-1] == "COMMAND":
			t.Dependencies = append(t.Dependencies, tag.Dependency{
				Head:      i - 1,
				Dependent: i,
				Relation:  "dobj",
			})
		case i > 0 && t.NerTag[i] == "FIELDS" && t.NerTag[i-1] == "OBJECT_NAME":
			for j := i - 1; j >= 0; j-- {
				if t.NerTag[j] == "DATA_TYPE" && t.Tokens[j] == "integer" {
					t.Dependencies = append(t.Dependencies, tag.Dependency{Head: j, Dependent: i, Relation: "integer_field"})
					break
				}
			}
		case i > 0 && t.NerTag[i] == "ARGUMENT" && t.NerTag[i-1] == "COMMAND":
			t.Dependencies = append(t.Dependencies, tag.Dependency{
				Head:      i - 1,
				Dependent: i,
				Relation:  "dobj",
			})
		case i > 0 && t.PhraseTag[i] == "ADVP" && t.PhraseTag[i-1] == "VP":
			t.Dependencies = append(t.Dependencies, tag.Dependency{
				Head:      i - 1,
				Dependent: i,
				Relation:  "advmod",
			})

		case i > 0 && t.NerTag[i] == "DATABASE_NAME" && t.NerTag[i-1] == "DATABASE":
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i - 1, Dependent: i, Relation: "NOUN"})
		case i > 0 && t.NerTag[i] == "DATA_STRUCTURE_NAME" && t.NerTag[i-1] == "DATABASE":
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i - 1, Dependent: i, Relation: "has_data_structure"})
		case i > 0 && t.NerTag[i] == "FIELD" && t.PosTag[i-1] == "CD":
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i - 1, Dependent: i, Relation: "nummod"})
		// Database name
		case i > 0 && t.NerTag[i] == "OBJECT_NAME" && t.NerTag[i-1] == "ACTION" && t.Tokens[i] == "Inventory":
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i - 1, Dependent: i, Relation: "named"})
		// Data structure
		case i > 0 && t.NerTag[i] == "OBJECT_NAME" && t.NerTag[i-1] == "OBJECT_TYPE" && t.Tokens[i] == "Product":
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i - 1, Dependent: i, Relation: "has_data_structure"})
		case t.NerTag[i] == "OBJECT_TYPE" && t.NerTag[i-1] == "DETERMINER":
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i, Dependent: i - 1, Relation: "det"})
			// String fields
		case i > 0 && t.NerTag[i] == "DATA_STRUCTURE_FIELD" && t.NerTag[i-1] == "DATA_TYPE" && t.Tokens[i-1] == "string":
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i - 1, Dependent: i, Relation: "string_fields"})
		// Integer field
		case i > 0 && t.NerTag[i] == "OBJECT_TYPE" && t.NerTag[i-1] == "DATA_TYPE" && t.Tokens[i-1] == "integer":
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i - 1, Dependent: i, Relation: "integer_field"})
		case i < len(t.Tokens)-1 && t.NerTag[i] == "COMMAND":
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i, Dependent: i + 1, Relation: "command"})
		case i > 0 && t.Tokens[i-1] == "data" && t.Tokens[i] == "structure":
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i, Dependent: i - 1, Relation: "compound"})
		case t.Tokens[i] == "the" && t.PosTag[i] == "DET":
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i + 1, Dependent: i, Relation: "det"})
		case t.Tokens[i] == "named" && t.PosTag[i] == "JJ" || t.PosTag[i] == "NN":
			t.Dependencies = append(t.Dependencies, tag.Dependency{Head: i + 1, Dependent: i, Relation: "det"})
			if rootIndex != -1 {
				t.Dependencies = append(t.Dependencies, tag.Dependency{Head: -1, Dependent: rootIndex, Relation: "root"})
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
						t.Dependencies = append(t.Dependencies, tag.Dependency{Head: nounIndex, Dependent: i, Relation: "det"})
					}
				}
			}

			if len(t.Tokens) != len(t.Dependencies) {
				diff := len(t.Tokens) - len(t.Dependencies)

				// Add padding to t.Dependencies
				if diff > 0 {
					for i := 0; i < diff; i++ {
						t.Dependencies = append(t.Dependencies, tag.Dependency{Relation: " "}) // Correctly append
					}
				} else {
					// Handle case where t.Dependencies is longer: truncate
					t.Dependencies = t.Dependencies[:len(t.Tokens)]
				}
			}
		}
	}
	return t, nil

}