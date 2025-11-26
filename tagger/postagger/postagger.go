package postagger

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/zendrulat/nlptagger/tagger/stem"
	"github.com/zendrulat/nlptagger/tagger/tag"
)

var posTags = map[string]string{
	`\b(?:create)\b`:   "VB",
	`\b(?:function)\b`: "NN",
	`\b(?:the)\b`:      "DET",
	`\bin\b`:           "IN",
	`\bcalled\b`:       "VBN",
	`\bthat\b`:         "WDT",
	`\btakes\b`:        "VBZ",
	`\b(?:add|new|implement|generate|define|create)\b`:                                          "CODE_VB",
	`\b(?:tl|tls|certificate|project|deploy|model|user|database|schema|products|JWT|handler)\b`: "CODE_NN",

	`\b(?:what(?:ever)?|which(?:ever)?|whose|that)\b`: "WDT", //Wh-determiner
	`\b(?:containing|holding|including)\b`:            "VBG", // Present Participle Verbs
	`\b(?:and|or|but|nor|for|yet|so)\b`:               "CC",
	`\b(?:with|of|in|on|at|by|for|from|to|through|over|under|around|between|among|across|near|behind|beside|against|along|up|down|inside|outside)\b`: "IN", // Prepositions
	`\b(?:[Aa]n?|[Tt]he)\b`:              "DET", // Article
	`\b(?:a|an|the)\b`:                   "DET", // Determiners
	`\bwebserver\b`:                      "NN",
	`\b(?:[Ii]|you|he|she|it|we|they)\b`: "PRP", // Personal pronoun
	//` \b(?:[A-Za-z]{3,}(?:ness|ment|tion|ance|ence|ship|ity|ism)\b|[A-Za-z]{3,}[A-Za-z]+)`: "NN",   // Abstract nouns
	`\b(?:[A-Za-z]+(?:er|or|ist))\b`:                     "NN",  // Agent nouns
	`\b(?:[A-Za-z]+s)\b`:                                 "NNS", // Plural nouns (regular)
	`\b(?:[A-Za-z]+(?:es|ies))\b`:                        "NNS", // Plural nouns (special cases)
	`\b(?:[0-9]+)\b`:                                     "CD",  // Cardinal number
	`\b(?:(?:[A-Za-z]+ly)|[Vv]ery|[Ee]xtremely|[Ss]o)\b`: "RB",  // Adverb
	`\b(?:when|where|why|how)\b`:                         "WP",  //Wh-pronoun
	`\b(?:(?:[A-Za-z]+est)|[Mm]ost|[Ll]east)\b`:          "RBS", // Superlative adverb
	`\b(?:[Bb]e|[Hh]ave|[Dd]o|[Ss]ay|[Gg]o|[Gg]et|[Mm]ake|[Kk]now|[Tt]hink|[Ss]ee|[Ff]eel|[Ww]ant|[Cc]reate)\b`: "VB", // Base form verb
	`\b(?:am|is|are|was|were|been)\b`: "VBZ", // Verb (to be)
	`\b(?:[Hh]as|[Hh]ad)\b`:           "VBD", // Verb (to have)
	// Verb phrases (basic)
	`\b(?:(?:[Hh]as|have|[Hh]ad) been \w+ing)\b`: "VBG", // Present perfect continuous
	`\b(?:(?:[Ww]as|were) \w+ing)\b`:             "VBG", // Past continuous
	`\b(?:\w+ed)\b`:                              "VBN", // Verb (past participle)
	`\b(?:\w+s)\b`:                               "VBZ", // Verb (present tense, 3rd person singular)
	`\b(?:could|would|should|might|must)\b`:      "MD",  // Modal verb
	`\b(?:[Aa]fter|[Aa]s|[Bb]ecause|[Bb]efore|[Ii]f|[Ss]ince|[Uu]ntil|[Ww]hile|[Aa]lthough|[Tt]hough)\b`:             "IN", // Subordinating conjunction
	`\b(?:to|from|in|on|at|by|with|about|against|between|into|through|during|before|after|above|below|under|over)\b`: "IN", // Preposition
	`\b(?:[Oo]h|[Aa]h|[Ww]ow|[Uu]h|[Hh]uh|[Yy]eah|[Nn]o|[Pp]lease|[Tt]hanks)\b`:                                      "UH", // Interjection
	`\b\w+\.\w+\b`: ".",
	`[\(\)]`:   "-LRB-", // Left/Right Round Bracket
	`[\{\}]`:   "-LCB-", // Left/Right Curly Bracket
	`[\[\]]`:   "-LSB-", // Left/Right Square Bracket
	// Dates (very basic)
	`\b(?:\d{1,2}/\d{1,2}/\d{2,4})\b`: "DATE",
	// Times (basic)
	`\b(?:\d{1,2}:\d{2}(?::\d{2})?(?:\s?(?:AM|PM))?)\b`: "TIME",
	// Emails (simplified)
	`\b(?:\w+@\w+\.\w+)\b`: "EMAIL",
}

var posTagToID map[string]int

func init() {
	posTagToID = make(map[string]int)
	id := 0
	for _, tagValue := range posTags {
		if _, ok := posTagToID[tagValue]; !ok {
			posTagToID[tagValue] = id
			id++
		}
	}
}

func PosTagMap() map[string]string {
	return posTags
}

func PosTagToIDMap() map[string]int {
	return posTagToID
}

func PosTags() int {
	return len(posTagToID)
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
	var stemtoken string
	for _, token := range tokens { // Iterate over each token
		// Split the sentence into words using space as the delimiter

		if token != "tls" {
			stemtoken = stem.Stem(token)
		}

		//	spacetext := addSpaces(stemtoken)
		for pattern, tag := range posTags {
			re = regexp.MustCompile(pattern)

			if re.MatchString(stemtoken) {
				// if token == "the" || token == " th " {
				// 	fmt.Println(token)
				// }
				t.PosTag = append(t.PosTag, tag)
				break // Move to the next token once a match is found
			} else {
				match, err := regexp.MatchString(pattern, stemtoken)
				if err != nil {
					fmt.Println(err)
				}
				if match {
					t.PosTag = append(t.PosTag, tag)
					break
				}
			}
		}

	}
	t.Tokens = addMissingValues(t.Tokens, tokens)
	//t.PosTag = padMissingvaluesWithNN(t.PosTag, tokens)
	t.PosTag = padMissingvalues(t)
	t.Tokens = tokens

	return t
}

func TagTokens(tokens []string) []string {
	t := tag.Tag{
		Tokens: tokens,
	}
	var stemtoken string
	for _, token := range tokens { // Iterate over each token
		if token != "tls" {
			stemtoken = stem.Stem(token)
		}

		found := false
		for pattern, tag := range posTags {
			re := regexp.MustCompile(pattern)
			if re.MatchString(stemtoken) {
				t.PosTag = append(t.PosTag, tag)
				found = true
				break // Move to the next token once a match is found
			}
		}
		if !found {
			t.PosTag = append(t.PosTag, "NN") // Default to NN
		}
	}
	t = VerbCheck(t)
	t = NounCheck(t)

	if len(t.PosTag) > len(tokens) {
			t.PosTag = t.PosTag[:len(tokens)]
	}

	return t.PosTag
}

func addSpaces(word string) string {
	return " " + word + " "
}
func VerbCheck(t tag.Tag) tag.Tag {
	commandVerbs := map[string]bool{
		"make": true, "build": true, "generate": true, "execute": true, "launch": true, "initiate": true, "set": true, "define": true, "establish": true, "form": true, "compose": true, "construct": true, "design": true, "develop": true, "engineer": true, "fabricate": true, "manufacture": true, "produce": true, "serve": true, "host": true, "deploy": true, "run": true, "start": true, "stop": true, "restart": true, "configure": true, "manage": true, "monitor": true,
	}
	pastTenseVerbs := map[string]bool{
		"create": true,

		"went": true, "gone": true, "began": true, "begun": true, "blew": true, "broke": true, "brought": true, "built": true, "bought": true, "caught": true, "chose": true, "came": true, "come": true, "cut": true, "did": true, "drew": true, "drank": true, "drove": true, "ate": true, "fell": true, "felt": true, "fought": true, "found": true, "flew": true, "forgot": true, "forgave": true, "got": true, "gave": true, "grew": true, "had": true, "heard": true, "hid": true, "hit": true, "held": true, "hurt": true, "kept": true, "knew": true, "laid": true, "led": true, "left": true, "lent": true, "let": true, "lay": true, "lit": true, "lost": true, "made": true, "meant": true, "met": true, "paid": true, "put": true, "quit": true, "read": true, "rode": true, "rang": true, "rose": true, "ran": true, "said": true, "saw": true, "sold": true, "sent": true, "set": true, "shook": true, "shone": true, "shot": true, "showed": true, "shut": true, "sang": true, "sat": true, "slept": true, "slid": true, "spoke": true, "spent": true, "stood": true, "stole": true, "stuck": true, "stung": true, "stank": true, "struck": true, "swore": true, "swept": true, "swam": true, "took": true, "taught": true, "tore": true, "told": true, "thought": true, "threw": true, "understood": true, "woke": true, "wore": true, "won": true, "wrote": true,
	}
	pastParticipleVerbs := map[string]bool{
		"done": true, "bitten": true, "blown": true, "broken": true, "chosen": true, "drawn": true, "drunk": true, "driven": true, "eaten": true, "fallen": true, "flown": true, "forgotten": true, "forgiven": true, "gotten": true, "given": true, "gone": true, "grown": true, "hidden": true, "known": true, "lain": true, "lighted": true, "ridden": true, "rung": true, "risen": true, "run": true, "seen": true, "shaken": true, "shined": true, "shown": true, "sung": true, "spoken": true, "stolen": true, "stunk": true, "sworn": true, "swum": true, "taken": true, "torn": true, "thrown": true, "woken": true, "worn": true, "written": true,
	}

	for i, token := range t.Tokens {

		switch {
		case t.PosTag[i] == "VB" && strings.HasSuffix(token, "ed"):
			t.PosTag[i] = "VBD"
		case t.PosTag[i] == "VB" && strings.HasSuffix(token, "ing"):
			t.PosTag[i] = "VBG"
		case pastTenseVerbs[token]:
			t.PosTag[i] = "VBD"
		case commandVerbs[token]:
			t.PosTag[i] = "VB"
		case pastParticipleVerbs[token]:
			t.PosTag[i] = "VBN"
		case token == "be" || token == "am" || token == "is" || token == "are" || token == "was" || token == "were" || token == "been":
			t.PosTag[i] = "VB"
		case token == "has" || token == "have" || token == "had":
			t.PosTag[i] = "VBG"
		case token == "will" || token == "would" || token == "should" || token == "could" || token == "might" || token == "must":
			t.PosTag[i] = "MD"
		case token == "can" || token == "may":
			t.PosTag[i] = "MD"
		case token == "do" || token == "did":
			t.PosTag[i] = "VBD"
		case token == "does":
			t.PosTag[i] = "VBZ"
		case token == "are" || token == "is":
			t.PosTag[i] = "VBZ"
		case token == "were" || token == "was":
			t.PosTag[i] = "VBD"
		}
	}
	return t
}

func NounCheck(t tag.Tag) tag.Tag {
	var knownProperNouns = map[string]bool{}

	// Initialize t.NerTag
	t.NerTag = make([]string, len(t.Tokens)) // or len(tokens) if that's where length comes from

	var commonNouns = map[string]bool{
		"Inventory": true, "Amazon": true, "Apple": true, "Google": true, "Microsoft": true, "Facebook": true, "Netflix": true, "Adobe": true, "Tesla": true, "SpaceX": true, "NASA": true, "United Nations": true, "European Union": true, "Red Cross": true, "World Bank": true, "International Monetary Fund": true, "IBM": true, "Samsung": true, "Sony": true, "Boeing": true, "Airbus": true, "Toyota": true, "Ford": true, "General Motors": true, "ExxonMobil": true, "Shell": true, "BP": true, "Walmart": true, "Disney": true, "London": true, "Paris": true, "New York": true, "Tokyo": true, "Beijing": true, "Los Angeles": true, "Berlin": true, "Rome": true, "Moscow": true, "Sydney": true, "United States": true, "United Kingdom": true, "Canada": true, "France": true, "Germany": true, "Japan": true, "China": true, "India": true, "Brazil": true, "Australia": true, "Spain": true, "Italy": true, "Mexico": true, "Russia": true, "John": true, "Jane": true, "David": true, "Michael": true, "Sarah": true, "Emily": true, "Jessica": true, "Ashley": true, "Brian": true, "Christopher": true, "Daniel": true, "Matthew": true, "Andrew": true, "James": true, "Justin": true, "Amanda": true, "Nicole": true, "Melissa": true, "Stephanie": true, "Rebecca": true, "Shakespeare": true, "Einstein": true, "Gandhi": true, "Napoleon": true, "Cleopatra": true, "Sherlock Holmes": true, "Harry Potter": true, "Leonardo da Vinci": true, "Nelson Mandela": true, "Martin Luther King Jr.": true, "Queen Elizabeth II": true, "Julius Caesar": true, "Albert Einstein": true, "Marie Curie": true, "Eiffel Tower": true, "Great Wall of China": true, "Statue of Liberty": true, "Taj Mahal": true, "Pyramids of Giza": true, "Big Ben": true, "Golden Gate Bridge": true, "Colosseum": true, "Sydney Opera House": true, "Mount Everest": true, "iPhone": true, "Windows": true, "Coca-Cola": true, "Nike": true, "Adidas": true, "Android": true, "PlayStation": true, "Xbox": true, "Amazon Prime": true, "Monday": true, "Tuesday": true, "Wednesday": true, "Thursday": true, "Friday": true, "Saturday": true, "Sunday": true, "January": true, "February": true, "March": true, "April": true, "May": true, "June": true, "July": true, "August": true, "September": true, "October": true, "November": true, "December": true, "Christmas": true, "Easter": true, "Thanksgiving": true, "Halloween": true, "New Year's Day": true, "Valentine's Day": true, "Mount Rushmore": true, "Grand Canyon": true, "Yosemite National Park": true, "Yellowstone National Park": true, "Statue of David": true, "Mona Lisa": true, "The Starry Night": true, "The Last Supper": true, "Sistine Chapel ceiling": true, "Hamlet": true, "Romeo and Juliet": true, "The Great Gatsby": true, "To Kill a Mockingbird": true, "1984": true, "The Lord of the Rings": true, "The Hobbit": true, "Pride and Prejudice": true, "The Catcher in the Rye": true, "The Adventures of Huckleberry Finn": true, "The Bible": true, "The Quran": true, "The Torah": true, "The Bhagavad Gita": true, "The Iliad": true, "The Odyssey": true, "The Republic": true, "The Art of War": true, "The Communist Manifesto": true, "The Origin of Species": true, "A Brief History of Time": true, "Cosmos": true, "Sapiens: A Brief History of Humankind": true, "The Diary of a Young Girl": true, "I Am Malala": true, "Becoming": true, "The Autobiography of Malcolm X": true, "Long Walk to Freedom": true,
		"cat": true, "person": true, "thing": true, "place": true, "time": true, "way": true, "year": true, "day": true, "man": true, "woman": true, "child": true, "book": true, "house": true, "car": true, "tree": true, "door": true, "table": true, "chair": true, "window": true, "computer": true, "phone": true, "pen": true, "pencil": true, "paper": true, "bag": true, "box": true, "animal": true, "plant": true, "dog": true, "bird": true, "fish": true, "insect": true, "flower": true, "idea": true, "thought": true, "feeling": true, "opinion": true, "problem": true, "solution": true, "story": true, "question": true, "answer": true, "reason": true, "action": true, "event": true, "meeting": true, "party": true, "trip": true, "game": true, "movie": true, "song": true, "dance": true, "group": true, "family": true, "company": true, "government": true, "country": true, "city": true, "school": true, "hospital": true, "world": true, "life": true, "work": true, "money": true, "food": true, "water": true, "air": true, "happiness": true, "freedom": true, "democracy": true, "love": true, "peace": true, "justice": true, "equality": true, "knowledge": true, "beauty": true, "truth": true, "walking": true, "reading": true, "eating": true, "sleeping": true, "working": true, "playing": true, "talking": true, "thinking": true, "running": true, "swimming": true,
	}

	for i, token := range t.Tokens {
		if token == "create" && i == 0 {
			t.PosTag[i] = "COMMAND_VB"

		}
	}

	for i, token := range t.Tokens {
		isProper := knownProperNouns[token]
		switch {
		case token == "the":
			t.PosTag[i] = "DET" // Tag as proper noun
			break
		case token == "in":
			t.PosTag[i] = "IN" // Tag as proper noun
			break
		case t.Tokens[i] == "the" || t.Tokens[i] == "a" || t.Tokens[i] == "an" || t.Tokens[i] == "The":
			t.PosTag[i] = "DET"
			break

		case t.Tokens[i] == "inventory":
			t.PosTag[i] = "NNP"
			break

		case t.Tokens[i] == "and":
			t.PosTag[i] = "CC"
			break

		case t.Tokens[i] == "with" || t.Tokens[i] == "of" || t.Tokens[i] == "in" || t.Tokens[i] == "on" || t.Tokens[i] == "at" || t.Tokens[i] == "by" || t.Tokens[i] == "for" || t.Tokens[i] == "from" || t.Tokens[i] == "to" || t.Tokens[i] == "through" || t.Tokens[i] == "over" || t.Tokens[i] == "under" || t.Tokens[i] == "around" || t.Tokens[i] == "between" || t.Tokens[i] == "among" || t.Tokens[i] == "across" || t.Tokens[i] == "near" || t.Tokens[i] == "behind" || t.Tokens[i] == "beside" || t.Tokens[i] == "against" || t.Tokens[i] == "along" || t.Tokens[i] == "up" || t.Tokens[i] == "down" || t.Tokens[i] == "inside" || t.Tokens[i] == "outside" || t.Tokens[i] == "before" || t.Tokens[i] == "after" || t.Tokens[i] == "during" || t.Tokens[i] == "while" || t.Tokens[i] == "although" || t.Tokens[i] == "though" || t.Tokens[i] == "if":
			t.PosTag[i] = "IN"
			break

		case t.Tokens[i] == "that":
			t.PosTag[i] = "WDT"
			break

		case i > 2 && i < len(t.Tokens) && token == "Inventory" && t.PosTag[i-1] == "VBN" && t.PosTag[i] != "CODE_NN":
			t.PosTag[i] = "NNP"
			break

		case token == "string" || token == "integer":
			t.PosTag[i] = "NN"
			break

		case strings.HasSuffix(token, "ing") && !strings.ContainsAny(token, ".,;!?"):
			t.PosTag[i] = "VBG" // Handle present participles like "containing"
			break

		case !isProper && commonNouns[token]:
			isProper = false
			t.PosTag[i] = "NN" // Explicitly tag as NN if in commonNouns
			break

		case t.Tokens[0] == "Create":
			t.PosTag[0] = "COMMAND_VB"
			break

		case isProper && i > 1 && i < len(t.Tokens):
			t.PosTag[i] = "NNP"
			break

		case t.NerTag[i] == "WEBSERVER":
			t.PosTag[i] = "CODE_NN"
			break

		case isProper && i > 2 && i < len(t.Tokens) && (t.PosTag[i-1] == "VB" || t.Tokens[i-1] == "webserver" || t.PosTag[i-1] == "IN"):
			t.PosTag[i] = "NNP"
			break

		case t.PosTag[i] == "NN" && token != "with" && token != "of":
			if i > 0 && (t.Tokens[i-1] == "named" || t.Tokens[i-1] == "handler" || t.Tokens[i-1] == "structure") {
				t.PosTag[i] = "NNP"
			}
			break

		case token == "handler":
			t.PosTag[i] = "NNP"
			t.NerTag[i] = "HANDLER"
			break

		case t.PosTag[i] == "NN" && t.Tokens[i] == "named":
			t.PosTag[i] = "VBN"
			break

		case i < len(t.Tokens) && i > 0 && t.PosTag[i-1] == "VBN" && t.Tokens[i] == "Inventory" && t.PosTag[i] != "CODE_NN":
			t.PosTag[i] = "NNP"
			break

		case t.Tokens[i] == "named":
			t.PosTag[i] = "VBN"
			break

		case t.PosTag[i] == "NNP" && t.PosTag[i] != "CODE_NN":
			if i < len(t.Tokens)-1 && token != "the" {
				nextToken := t.Tokens[i+1]
				if nextToken == "of" || nextToken == "in" || nextToken == "'s" || nextToken == "his" || nextToken == "her" {
					// Handle possessive cases or further analysis
					t.PosTag[i] = "NNP"
				}

			}

			if t.PosTag[i] == "NN" && strings.HasSuffix(token, "ing") && t.PosTag[i] != "CODE_NN" {
				if i > 0 && (t.Tokens[i-1] == "is" || t.Tokens[i-1] == "are" || t.Tokens[i-1] == "was" || t.Tokens[i-1] == "were") {
					t.PosTag[i] = "VBG"
				}
			}
			break

		case token == "Inventory" && i > 0 && t.PosTag[i-1] == "VBN":
			t.PosTag[i] = "NNP"
			break

		case isProper && i > 2 && i < len(t.Tokens) && i > 0 && (t.Tokens[i-1] == "a" || t.Tokens[i-1] == "an" || t.Tokens[i-1] == "the") && t.PosTag[i] != "CODE_NN":
			if i < len(t.Tokens)-1 && t.Tokens[i+1] == "of" {
				t.NerTag[i] = "OBJECT_TYPE"
			} else if t.PosTag[i] == "DET" && commonNouns[token] { // Check if tagged as DET but is a common noun
				t.NerTag[i] = "OBJECT_NAME"
				t.PosTag[i] = "NN" // Correct the POS tag
			} else {
				t.NerTag[i] = "OBJECT_NAME"
			}
			break

		}

		if _, err := strconv.Atoi(token); err == nil {
			t.PosTag[i] = "CD"

		}
	}
	return t
}
func isNumber(token string) bool {
	_, err := strconv.Atoi(token)
	return err == nil
}

func BasicTokenChecking(token string) string {
	var postag string
	// Handle special characters and numbers
	switch {
	case strings.ContainsAny(token, ".,;!?"):
		postag = "PUNCT"
	case isNumber(token):
		postag = "NUM"
	case isNumber(token): // You likely want a different condition here for "CD"
		postag = "CD"
	default:
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

func padMissingvalues(t tag.Tag) []string {
	diff := len(t.Tokens) - len(t.PosTag) // Calculate the difference in length

	if diff <= 0 {
		return t.PosTag // No need to pad if PosTag is equal or longer than Tokens
	}
	padding := make([]string, diff)
	for i := 0; i < diff; i++ {
		// Try to match token against patterns
		for pattern, tag := range posTags { // Iterate over POS tag patterns
			if regexp.MustCompile(pattern).MatchString(t.Tokens[len(t.PosTag)+i]) {
				padding[i] = tag // Assign found tag to padding
				break            // Exit inner loop as we've found a match
			} else {
				padding[i] = "NN" // Default to NN if no pattern matches
			}
		}
	}
	return append(t.PosTag, padding...) // Concatenate the original POS tags with the padding
}