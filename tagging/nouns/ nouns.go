package nouns

import "strings"

// checking if it is a noun
func NounTag(token, posTag, originalToken string, i int, tokens []string) string {
	var isProper bool
	// Check for known proper nouns (using a larger gazetteer - you'll need to expand this)
	knownProperNouns := map[string]bool{
		// Companies and Organizations
		"Apple": true, "Google": true, "Microsoft": true, "Amazon": true, "Facebook": true,
		"Netflix": true, "Adobe": true, "Tesla": true, "SpaceX": true, "NASA": true,
		"United Nations": true, "European Union": true, "Red Cross": true, "World Bank": true,
		"International Monetary Fund": true,
		// Cities and Countries
		"London": true, "Paris": true, "New York": true, "Tokyo": true, "Beijing": true,
		"Los Angeles": true, "Berlin": true, "Rome": true, "Moscow": true, "Sydney": true,
		"United States": true, "United Kingdom": true, "Canada": true, "France": true,
		"Germany": true, "Japan": true, "China": true, "India": true, "Brazil": true,
		"Australia": true,
		// People (Common Names - Be careful with these!)
		"John": true, "Jane": true, "David": true, "Michael": true, "Sarah": true,
		"Emily": true, "Jessica": true, "Ashley": true, "Brian": true, "Christopher": true,
		// Historical Figures and Characters
		"Shakespeare": true, "Einstein": true, "Gandhi": true, "Napoleon": true,
		"Cleopatra": true, "Sherlock Holmes": true, "Harry Potter": true,
		// Landmarks and Monuments
		"Eiffel Tower": true, "Great Wall of China": true, "Statue of Liberty": true,
		"Taj Mahal": true, "Pyramids of Giza": true, "Big Ben": true,
		// Brands and Products
		"iPhone": true, "Windows": true, "Coca-Cola": true, "Nike": true, "Adidas": true,
		// ... (Add more names, places, organizations, etc. relevant to your domain) ...
	}
	if knownProperNouns[token] {
		isProper = true
	}

	// Check for words that are more likely to be common nouns (you'll need to expand this)
	commonNouns := map[string]bool{
		// Basic Concepts
		"person": true, "thing": true, "place": true, "time": true, "way": true,
		"year": true, "day": true, "man": true, "woman": true, "child": true,
		// Objects
		"book": true, "house": true, "car": true, "tree": true, "door": true,
		"table": true, "chair": true, "window": true, "computer": true, "phone": true,
		"pen": true, "pencil": true, "paper": true, "bag": true, "box": true,
		// Living Things
		"animal": true, "plant": true, "dog": true, "cat": true, "bird": true,
		"fish": true, "insect": true, "flower": true,
		// Concepts and Ideas
		"idea": true, "thought": true, "feeling": true, "opinion": true, "problem": true,
		"solution": true, "story": true, "question": true, "answer": true, "reason": true,
		// Actions and Events
		"action": true, "event": true, "meeting": true, "party": true, "trip": true,
		"game": true, "movie": true, "song": true, "dance": true,
		// Other
		"group": true, "family": true, "company": true, "government": true, "country": true,
		"city": true, "school": true, "hospital": true, "world": true, "life": true,
		"work": true, "money": true, "food": true, "water": true, "air": true,
		// Objects
		"happiness": true, "freedom": true, "democracy": true, /* ... Add more concepts */
		// Actions
		"walking": true, "reading": true, "eating": true,
		// ... (Add more common nouns relevant to your domain) ...
	}
	if commonNouns[token] {
		isProper = false
	}

	// Apply the rule
	if isProper {
		posTag = "NNP"
	} else {
		posTag = "NN"
	}
	// Complex rule for NN vs. NNP
	if posTag == "NN" && token != "with" && token != "of" {
		isProper = false
	} else if posTag == "NNP" {
		// Check for capitalization (not at beginning of sentence)
		if strings.Title(token) == originalToken && i > 0 && tokens[i-1] != "." {
			isProper = true
		}

		// Check for preceding titles and honorifics
		if i > 0 && (tokens[i-1] == "Mr." || tokens[i-1] == "Ms." || tokens[i-1] == "Dr." || tokens[i-1] == "Professor") {
			isProper = true
		}

		// Check for following context (e.g., "of", "in", possessive pronouns), but exclude "the"
		if i < len(tokens)-1 && token != "the" {
			nextToken := tokens[i+1]
			if nextToken == "of" || nextToken == "in" || nextToken == "'s" || nextToken == "his" || nextToken == "her" {
				isProper = true
			}
		}
		if posTag == "NN" && strings.HasSuffix(token, "ing") {
			// Check if the word is preceded by an auxiliary verb (e.g., "is," "are," "was," "were")
			if i > 0 && (tokens[i-1] == "is" || tokens[i-1] == "are" || tokens[i-1] == "was" || tokens[i-1] == "were") {
				posTag = "VBG" // Present participle
			}
		}
		return posTag
	}
	return posTag
}
