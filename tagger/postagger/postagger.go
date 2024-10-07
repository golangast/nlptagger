package postagger

import (
	"regexp"
	"strconv"
	"strings"

	"github.com/golangast/nlptagger/tagger/tag"
)

var posTags = map[string]string{
	`\b(?:with|of|in|on|at|by|for|from|to|through|over|under|around|between|among|across|near|behind|beside|against|along|up|down|inside|outside)\b`: "IN", // Prepositions
	`\b(?:[Aa]n?|[Tt]he)\b`: "DET", // Article
	`\b(?:a|an|the)\b`:      "DET", // Determiners
	`\bwebserver\b`:         "NN",
	`\b[A-Za-z]{3,}(?:ness|ment|tion|ance|ence|ship|ity|ism)\b`:                            "NN",
	`\b[A-Za-z]{3,}[A-Za-z]+\b`:                                                            "NN",   // More gen
	`\b(?:[Mm]y|[Yy]our|[Hh]is|[Hh]er|[Ii]ts|[Oo]ur|[Tt]heir)\b`:                           "PRP$", // Possessive pronoun
	`\b(?:[Ii]|you|he|she|it|we|they)\b`:                                                   "PRP",  // Personal pronoun
	` \b(?:[A-Za-z]{3,}(?:ness|ment|tion|ance|ence|ship|ity|ism)\b|[A-Za-z]{3,}[A-Za-z]+)`: "NN",   // Abstract nouns
	`\b(?:[A-Za-z]+(?:er|or|ist))\b`:                                                       "NN",   // Agent nouns
	`\b(?:[A-Za-z]+s)\b`:                                                                   "NNS",  // Plural nouns (regular)
	`\b(?:[A-Za-z]+(?:es|ies))\b`:                                                          "NNS",  // Plural nouns (special cases)
	`\b(?:[0-9]+)\b`:                                                                       "CD",   // Cardinal number
	`\b(?:(?:[A-Za-z]+ly)|[Vv]ery|[Ee]xtremely|[Ss]o)\b`:                                   "RB",   // Adverb
	`\b(?:what(?:ever)?|which(?:ever)?|whose|that)\b`:                                      "WDT",  //Wh-determiner
	`\b(?:when|where|why|how)\b`:                                                           "WP",   //Wh-pronoun
	`\b(?:(?:[A-Za-z]+er)|[Mm]ore|[Ll]ess)\b`:                                              "RBR",  // Comparative adverb
	`\b(?:(?:[A-Za-z]+est)|[Mm]ost|[Ll]east)\b`:                                            "RBS",  // Superlative adverb
	`\b(?:[Bb]e|[Hh]ave|[Dd]o|[Ss]ay|[Gg]o|[Gg]et|[Mm]ake|[Kk]now|[Tt]hink|[Ss]ee|[Ff]eel|[Ww]ant|[Cc]reate)\b`: "VB", // Base form verb
	`\b(?:am|is|are|was|were|been)\b`: "VBZ", // Verb (to be)
	`\b(?:[Hh]as|[Hh]ad)\b`:           "VBD", // Verb (to have)
	// Verb phrases (basic)
	`\b(?:(?:[Hh]as|have|[Hh]ad) been \w+ing)\b`: "VBG", // Present perfect continuous
	`\b(?:(?:[Ww]as|were) \w+ing)\b`:             "VBG", // Past continuous
	`\b(?:\w+ed)\b`:                              "VBN", // Verb (past participle)
	`\b(?:\w+s)\b`:                               "VBZ", // Verb (present tense, 3rd person singular)
	`\b(?:could|would|should|might|must)\b`:      "MD",  // Modal verb
	`\b[A-Za-z]+(ly)\b`:                          "JJ",  // Adjective
	`\b(?:and|or|but|nor|for|yet|so)\b`:          "CC",  // Coordinating conjunction
	`\b(?:[Aa]fter|[Aa]s|[Bb]ecause|[Bb]efore|[Ii]f|[Ss]ince|[Uu]ntil|[Ww]hile|[Aa]lthough|[Tt]hough)\b`:             "IN", // Subordinating conjunction
	`\b(?:to|from|in|on|at|by|with|about|against|between|into|through|during|before|after|above|below|under|over)\b`: "IN", // Preposition
	`\b(?:[Oo]h|[Aa]h|[Ww]ow|[Uu]h|[Hh]uh|[Yy]eah|[Nn]o|[Pp]lease|[Tt]hanks)\b`:                                      "UH", // Interjection
	`[,.;:!?]`: ".",     // Punctuation
	`[\(\)]`:   "-LRB-", // Left/Right Round Bracket
	`[\{\}]`:   "-LCB-", // Left/Right Curly Bracket
	`[\[\]]`:   "-LSB-", // Left/Right Square Bracket
	// Dates (very basic)
	`\b(?:\d{1,2}/\d{1,2}/\d{2,4})\b`: "DATE",
	// Times (basic)
	`\b(?:\d{1,2}:\d{2}(?::\d{2})?(?:\s?(?:AM|PM))?)\b`: "TIME",
	// Emails (simplified)
	`\b(?:\w+@\w+\.\w+)\b`: "EMAIL",
	// URLs (simplified)
	`\b(?:https?://\S+)\b`: "URL",
	// Proper nouns (very basic - capitalized words)
	//`\b[A-Za-z]\w+\b`: "NNP", // Proper noun (singular)
	// More complex verb phrases (be careful with overmatching)
	`\b(?:(?:[Hh]as|have|[Hh]ad) \w+)\b`: "VBN", // Present perfect
	`\b(?:(?:[Ww]as|were) \w+)\b`:        "VBN", // Past perfect

}

// take in text and return a map of tokens to NER tags
func Postagger(text string) tag.Tag {

	// Create a regular expression to match periods and commas
	// Regular expression to match all punctuation characters
	re := regexp.MustCompile(`[[:punct:]]`)
	// Replace all matches with an empty string
	striptext := re.ReplaceAllString(text, "")

	tokens := strings.Fields(striptext)
	t := tag.Tag{}
	for _, token := range tokens { // Iterate over each token

		for pattern, tag := range posTags {
			re := regexp.MustCompile(pattern)
			if re.MatchString(token) {

				t.PosTag = append(t.PosTag, tag)
				t.Tokens = append(t.Tokens, token)
				break // Move to the next token once a match is found
			}
		}

	}

	t.Tokens = addMissingValues(t.Tokens, tokens)
	t.PosTag = padMissingvaluesWithNN(t.PosTag, tokens)
	return t
}

func VerbCheck(t tag.Tag) tag.Tag {
	commandVerbs := map[string]bool{
		"create":      true,
		"make":        true,
		"build":       true,
		"generate":    true,
		"execute":     true,
		"launch":      true,
		"initiate":    true,
		"set":         true,
		"define":      true,
		"establish":   true,
		"form":        true,
		"compose":     true,
		"construct":   true,
		"design":      true,
		"develop":     true,
		"engineer":    true,
		"fabricate":   true,
		"manufacture": true,
		"produce":     true,
		"serve":       true,
		"host":        true,
		"deploy":      true,
		"run":         true,
		"start":       true,
		"stop":        true,
		"restart":     true,
		"configure":   true,
		"manage":      true,
		"monitor":     true,
	}
	pastTenseVerbs := map[string]bool{
		"went":       true,
		"gone":       true,
		"began":      true,
		"begun":      true,
		"blew":       true,
		"broke":      true,
		"brought":    true,
		"built":      true,
		"bought":     true,
		"caught":     true,
		"chose":      true,
		"came":       true,
		"come":       true,
		"cut":        true,
		"did":        true,
		"drew":       true,
		"drank":      true,
		"drove":      true,
		"ate":        true,
		"fell":       true,
		"felt":       true,
		"fought":     true,
		"found":      true,
		"flew":       true,
		"forgot":     true,
		"forgave":    true,
		"got":        true,
		"gave":       true,
		"grew":       true,
		"had":        true,
		"heard":      true,
		"hid":        true,
		"hit":        true,
		"held":       true,
		"hurt":       true,
		"kept":       true,
		"knew":       true,
		"laid":       true,
		"led":        true,
		"left":       true,
		"lent":       true,
		"let":        true,
		"lay":        true,
		"lit":        true,
		"lost":       true,
		"made":       true,
		"meant":      true,
		"met":        true,
		"paid":       true,
		"put":        true,
		"quit":       true,
		"read":       true,
		"rode":       true,
		"rang":       true,
		"rose":       true,
		"ran":        true,
		"said":       true,
		"saw":        true,
		"sold":       true,
		"sent":       true,
		"set":        true,
		"shook":      true,
		"shone":      true,
		"shot":       true,
		"showed":     true,
		"shut":       true,
		"sang":       true,
		"sat":        true,
		"slept":      true,
		"slid":       true,
		"spoke":      true,
		"spent":      true,
		"stood":      true,
		"stole":      true,
		"stuck":      true,
		"stung":      true,
		"stank":      true,
		"struck":     true,
		"swore":      true,
		"swept":      true,
		"swam":       true,
		"took":       true,
		"taught":     true,
		"tore":       true,
		"told":       true,
		"thought":    true,
		"threw":      true,
		"understood": true,
		"woke":       true,
		"wore":       true,
		"won":        true,
		"wrote":      true,
	}

	pastParticipleVerbs := map[string]bool{
		"done":      true,
		"bitten":    true,
		"blown":     true,
		"broken":    true,
		"chosen":    true,
		"drawn":     true,
		"drunk":     true,
		"driven":    true,
		"eaten":     true,
		"fallen":    true,
		"flown":     true,
		"forgotten": true,
		"forgiven":  true,
		"gotten":    true,
		"given":     true,
		"gone":      true,
		"grown":     true,
		"hidden":    true,
		"known":     true,
		"lain":      true,
		"lighted":   true,
		"ridden":    true,
		"rung":      true,
		"risen":     true,
		"run":       true,
		"seen":      true,
		"shaken":    true,
		"shined":    true,
		"shown":     true,
		"sung":      true,
		"spoken":    true,
		"stolen":    true,
		"stunk":     true,
		"sworn":     true,
		"swum":      true,
		"taken":     true,
		"torn":      true,
		"thrown":    true,
		"woken":     true,
		"worn":      true,
		"written":   true,
	}

	for i, token := range t.Tokens {
		if commandVerbs[token] {
			t.PosTag[i] = "COMMAND_VERB"
		} else if t.PosTag[i] == "VB" && strings.HasSuffix(token, "ing") {
			t.PosTag[i] = "VBG"
		} else if pastTenseVerbs[token] {
			t.PosTag[i] = "VBD"
		} else if pastParticipleVerbs[token] {
			t.PosTag[i] = "VBN"
		} else if token == "be" || token == "am" || token == "is" || token == "are" || token == "was" || token == "were" || token == "been" {
			t.PosTag[i] = "VB"
		} else if token == "has" || token == "have" || token == "had" {
			t.PosTag[i] = "VBG"
		} else if token == "will" || token == "would" || token == "should" || token == "could" || token == "might" || token == "must" {
			t.PosTag[i] = "MD"
		} else if token == "can" || token == "may" {
			t.PosTag[i] = "MD"
		} else if token == "do" || token == "did" {
			t.PosTag[i] = "VBD"
		} else if token == "does" {
			t.PosTag[i] = "VBZ"
		} else if token == "are" || token == "is" {
			t.PosTag[i] = "VBZ"
		} else if token == "were" || token == "was" {
			t.PosTag[i] = "VBD"
		}
	}
	return t
}

func NounCheck(t tag.Tag) tag.Tag {
	var knownProperNouns = map[string]bool{
		// Companies and Organizations
		"Apple": true, "Google": true, "Microsoft": true, "Amazon": true, "Facebook": true,
		"Netflix": true, "Adobe": true, "Tesla": true, "SpaceX": true, "NASA": true,
		"United Nations": true, "European Union": true, "Red Cross": true, "World Bank": true,
		"International Monetary Fund": true, "IBM": true, "Samsung": true, "Sony": true,
		"Boeing": true, "Airbus": true, "Toyota": true, "Ford": true, "General Motors": true,
		"ExxonMobil": true, "Shell": true, "BP": true, "Walmart": true, "Disney": true,

		// Cities and Countries
		"London": true, "Paris": true, "New York": true, "Tokyo": true, "Beijing": true,
		"Los Angeles": true, "Berlin": true, "Rome": true, "Moscow": true, "Sydney": true,
		"United States": true, "United Kingdom": true, "Canada": true, "France": true,
		"Germany": true, "Japan": true, "China": true, "India": true, "Brazil": true,
		"Australia": true, "Spain": true, "Italy": true, "Mexico": true, "Russia": true,

		// People (Common Names - Be careful with these!)
		"John": true, "Jane": true, "David": true, "Michael": true, "Sarah": true,
		"Emily": true, "Jessica": true, "Ashley": true, "Brian": true, "Christopher": true,
		"Daniel": true, "Matthew": true, "Andrew": true, "James": true, "Justin": true,
		"Amanda": true,
		"Nicole": true, "Melissa": true, "Stephanie": true, "Rebecca": true,

		// Historical Figures and Characters
		"Shakespeare": true, "Einstein": true, "Gandhi": true, "Napoleon": true,
		"Cleopatra": true, "Sherlock Holmes": true, "Harry Potter": true, "Leonardo da Vinci": true,
		"Nelson Mandela": true, "Martin Luther King Jr.": true, "Queen Elizabeth II": true,
		"Julius Caesar": true, "Albert Einstein": true, "Marie Curie": true,

		// Landmarks and Monuments
		"Eiffel Tower": true, "Great Wall of China": true, "Statue of Liberty": true,
		"Taj Mahal": true, "Pyramids of Giza": true, "Big Ben": true, "Golden Gate Bridge": true,
		"Colosseum": true, "Sydney Opera House": true, "Mount Everest": true,

		// Brands and Products
		"iPhone": true, "Windows": true, "Coca-Cola": true, "Nike": true, "Adidas": true,
		"Android": true, "PlayStation": true, "Xbox": true, "Amazon Prime": true,

		// Days of the Week and Months of the Year
		"Monday": true, "Tuesday": true, "Wednesday": true, "Thursday": true, "Friday": true,
		"Saturday": true, "Sunday": true, "January": true, "February": true, "March": true,
		"April": true, "May": true, "June": true, "July": true, "August": true, "September": true,
		"October": true, "November": true, "December": true,

		// Programming Languages
		"Python": true, "JavaScript": true, "Java": true, "C++": true, "C#": true,
		"PHP": true, "Swift": true, "Go": true, "Ruby": true, "Kotlin": true,

		// ... (Add more names, places, organizations, etc. relevant to your domain) ...
	}
	var commonNouns = map[string]bool{
		// Basic Concepts
		"cat": true, "person": true, "thing": true, "place": true, "time": true, "way": true,
		"year": true, "day": true, "man": true, "woman": true, "child": true,
		// Objects
		"book": true, "house": true, "car": true, "tree": true, "door": true,
		"table": true, "chair": true, "window": true, "computer": true, "phone": true,
		"pen": true, "pencil": true, "paper": true, "bag": true, "box": true,
		// Living Things
		"animal": true, "plant": true, "dog": true, "bird": true,
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
		// Abstract Concepts
		"happiness": true, "freedom": true, "democracy": true, "love": true, "peace": true,
		"justice": true, "equality": true, "knowledge": true, "beauty": true, "truth": true,
		// Actions (Gerunds)
		"walking": true, "reading": true, "eating": true, "sleeping": true, "working": true,
		"playing": true, "talking": true, "thinking": true, "running": true, "swimming": true,
		// ... (Add more common nouns relevant to your domain) ...
	}
	// Initialize t.NerTag
	t.NerTag = make([]string, len(t.Tokens)) // or len(tokens) if that's where length comes from

	for i, token := range t.Tokens {
		isProper := knownProperNouns[token]
		if !isProper && commonNouns[token] {
			isProper = false
			t.PosTag[i] = "NN" // Explicitly tag as NN if in commonNouns
		}

		if isProper && i > 1 && i < len(t.Tokens) {
			t.PosTag[i] = "NNP"
		} else if t.Tokens[i] == "the" {
			t.PosTag[i] = "DET"

		} else if t.Tokens[i] == "webserver" {
			t.NerTag[i] = "WEBSERVER"
			t.PosTag[i] = "NN"
		} else if t.Tokens[i] == "with" {
			t.PosTag[i] = "IN"

		} else if isProper && i > 2 && i < len(t.Tokens) && (t.PosTag[i-1] == "VB" || t.Tokens[i-1] == "webserver" || t.PosTag[i-1] == "IN") {
			t.PosTag[i] = "NNP"
		} else if t.PosTag[i] == "NN" && token != "with" && token != "of" {
			if i > 0 && (t.Tokens[i-1] == "named" || t.Tokens[i-1] == "handler" || t.Tokens[i-1] == "structure") {
				t.PosTag[i] = "NNP"
			} else if token == "webserver" {
				t.NerTag[i] = "WEBSERVER"
				t.PosTag[i] = "NN"
			} else if token == "handler" {
				t.NerTag[i] = "HANDLER"
			} // ... more special cases ...

		} else if t.PosTag[i] == "NNP" {
			if i < len(t.Tokens)-1 && token != "the" {
				nextToken := t.Tokens[i+1]
				if nextToken == "of" || nextToken == "in" || nextToken == "'s" || nextToken == "his" || nextToken == "her" {
					// Handle possessive cases or further analysis
				}
			}
			if t.PosTag[i] == "NN" && strings.HasSuffix(token, "ing") {
				if i > 0 && (t.Tokens[i-1] == "is" || t.Tokens[i-1] == "are" || t.Tokens[i-1] == "was" || t.Tokens[i-1] == "were") {
					t.PosTag[i] = "VBG"
				}
			}
		} else if isProper && i > 2 && i < len(t.Tokens) && i > 0 && (t.Tokens[i-1] == "a" || t.Tokens[i-1] == "an" || t.Tokens[i-1] == "the") {
			if i < len(t.Tokens)-1 && t.Tokens[i+1] == "of" {
				t.NerTag[i] = "OBJECT_TYPE"
			} else if t.PosTag[i] == "DET" && commonNouns[token] { // Check if tagged as DET but is a common noun
				t.NerTag[i] = "OBJECT_NAME"
				t.PosTag[i] = "NN" // Correct the POS tag
			} else {
				t.NerTag[i] = "OBJECT_NAME"
			}
		}
	}
	return t
}

func BasicTokenChecking(token string) string {
	var postag string
	// Handle special characters and numbers
	if strings.ContainsAny(token, ".,;!?") {
		postag = "PUNCT"
	} else if _, err := strconv.Atoi(token); err == nil {
		postag = "NUM"
	} else {
		return postag

	}
	return postag
}

func addMissingValues(slice1, slice2 []string) []string {
	// Create a map to store the elements of slice2 for faster lookup.
	slice2Map := make(map[string]bool)
	for _, val := range slice2 {
		slice2Map[val] = true
	}
	// Iterate over slice1 and add missing elements to slice2.
	for _, val := range slice1 {
		if _, ok := slice2Map[val]; !ok { // If the element is not in slice2
			slice2 = append(slice2, val) // Add it to slice2
		}
	}
	return slice2
}

// findTokensWithoutPosTags identifies tokens in 'tokens' that lack corresponding POS tags in 'posTags'
// and adds "NN" as their POS tag.

func padMissingvaluesWithNN(target, source []string) []string {
	if len(target) >= len(source) {
		return target
	}
	diff := len(source) - len(target)
	padding := make([]string, diff)
	for i := range padding {
		padding[i] = "NN" // Or any other desired value
	}
	return append(target, padding...)
}
