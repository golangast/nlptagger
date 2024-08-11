package tagging

import "regexp"

// checking and returning the tags
func Tagging(text string) ([]string, map[string]string, map[string]string, map[string]string, map[string]bool) {

	// Regular expression to match words, numbers, and punctuation.
	re := regexp.MustCompile(`[\w]+|[^\s\w]`)
	tokens := re.FindAllString(text, -1)
	contractions := map[string]string{
		"don't":     "do not",
		"can't":     "cannot",
		"aren't":    "are not",
		"couldn't":  "could not",
		"didn't":    "did not",
		"doesn't":   "does not",
		"hadn't":    "had not",
		"hasn't":    "has not",
		"haven't":   "have not",
		"he'd":      "he would", // Or "he had" depending on context
		"he'll":     "he will",
		"he's":      "he is",   // Or "he has" depending on context
		"how'd":     "how did", // Or "how would" depending on context
		"how'll":    "how will",
		"how's":     "how is",  // Or "how has" depending on context
		"I'd":       "I would", // Or "I had" depending on context
		"I'll":      "I will",
		"I'm":       "I am",
		"I've":      "I have",
		"isn't":     "is not",
		"it'd":      "it would", // Or "it had" depending on context
		"it'll":     "it will",
		"it's":      "it is", // Or "it has" depending on context
		"mightn't":  "might not",
		"mustn't":   "must not",
		"shan't":    "shall not",
		"she'd":     "she would", // Or "she had" depending on context
		"she'll":    "she will",
		"she's":     "she is", // Or "she has" depending on context
		"shouldn't": "should not",
		"that's":    "that is",
		"there'd":   "there would", // Or "there had" depending on context
		"there'll":  "there will",
		"there's":   "there is",   // Or "there has" depending on context
		"they'd":    "they would", // Or "they had" depending on context
		"they'll":   "they will",
		"they're":   "they are",
		"they've":   "they have",
		"wasn't":    "was not",
		"we'd":      "we would", // Or "we had" depending on context
		"we'll":     "we will",
		"we're":     "we are",
		"we've":     "we have",
		"weren't":   "were not",
		"what'll":   "what will",
		"what're":   "what are",
		"what's":    "what is", // Or "what has" depending on context
		"what've":   "what have",
		"where'd":   "where did",
		"where's":   "where is",
		"who'd":     "who would", // Or "who had" depending on context
		"who'll":    "who will",
		"who's":     "who is", // Or "who has" depending on context
		"who've":    "who have",
		"why'd":     "why did",
		"why's":     "why is",
		"won't":     "will not",
		"wouldn't":  "would not",
		"you'd":     "you would", // Or "you had" depending on context
		"you'll":    "you will",
		"you're":    "you are",
		"you've":    "you have",
	}
	posTags := map[string]string{
		`\b(?:[Aa]n?|[Tt]he)\b`: "DET", // Article
		`\b(?:[Mm]y|[Yy]our|[Hh]is|[Hh]er|[Ii]ts|[Oo]ur|[Tt]heir)\b`: "PRP$", // Possessive pronoun
		`\b(?:[Ii]|you|he|she|it|we|they)\b`:                         "PRP",  // Personal pronoun
		`\b(?:[A-Za-z]+(?:ness|ment|tion|ance|ence|ship|ity|ism))\b`: "NN",   // Abstract nouns
		`\b(?:[A-Za-z]+(?:er|or|ist))\b`:                             "NN",   // Agent nouns
		`\b(?:[A-Za-z]+s)\b`:                                         "NNS",  // Plural nouns (regular)
		`\b(?:[A-Za-z]+(?:es|ies))\b`:                                "NNS",  // Plural nouns (special cases)
		`\b(?:[0-9]+)\b`:                                             "CD",   // Cardinal number
		`\b(?:(?:[A-Za-z]+ly)|[Vv]ery|[Ee]xtremely|[Ss]o)\b`:         "RB",   // Adverb
		`\b(?:(?:[A-Za-z]+er)|[Mm]ore|[Ll]ess)\b`:                    "RBR",  // Comparative adverb
		`\b(?:(?:[A-Za-z]+est)|[Mm]ost|[Ll]east)\b`:                  "RBS",  // Superlative adverb
		`\b(?:[Bb]e|[Hh]ave|[Dd]o|[Ss]ay|[Gg]o|[Gg]et|[Mm]ake|[Kk]now|[Tt]hink|[Ss]ee|[Ff]eel|[Ww]ant)\b`: "VB", // Base form verb
		`\b(?:am|is|are|was|were|been)\b`: "VBZ", // Verb (to be)
		`\b(?:[Hh]as|[Hh]ad)\b`:           "VBD", // Verb (to have)
		// Verb phrases (basic)
		`\b(?:(?:[Hh]as|have|[Hh]ad) been \w+ing)\b`: "VBG", // Present perfect continuous
		`\b(?:(?:[Ww]as|were) \w+ing)\b`:             "VBG", // Past continuous
		`\b(?:\w+ed)\b`:                              "VBN", // Verb (past participle)
		`\b(?:\w+s)\b`:                               "VBZ", // Verb (present tense, 3rd person singular)
		`\b(?:could|would|should|might|must)\b`:      "MD",  // Modal verb
		`\b(?:[A-Za-z]+(?:y|ful|less|able|ible))\b`:  "JJ",  // Adjective
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
		`\b[A-Za-z]\w+\b`: "NNP", // Proper noun (singular)
		// More complex verb phrases (be careful with overmatching)
		`\b(?:(?:[Hh]as|have|[Hh]ad) \w+)\b`: "VBN", // Present perfect
		`\b(?:(?:[Ww]as|were) \w+)\b`:        "VBN", // Past perfect

	}

	nerTags := map[string]string{
		`\b(?:(?:[Dd]r\.|[Mm]r\.|[Mm]rs\.|[Mm]s\.)\s+)?[A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*(?:\s+(?:[Jj]r\.|[Ss]r\.|III|IV))?\b`: "PERSON", // Names with titles, hyphens, suffixes
		`\b(?:[A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*\s+(?:&|and)\s+[A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*)\b`:                            "ORG",    // Partnerships (e.g., Smith & Jones)
		`\b(?:[Tt]he\s+)?[A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*\s+(?:[Oo]f|[Ii]n)\s+[A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*\b`:            "ORG",    // Organizations with "of" or "in" (e.g., The University of California)
		`\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s+[A-Z]{2})\b`:                                                                 "GPE",    // US locations with city and state
		`\b(?:[A-Z]{2,})\b`: "GPE", // Country codes (e.g., US, UK)
		`\b(?:[0-9]{1,3}\s+[A-Za-z]+(?:\s+(?:[Ss]t\.|[Aa]ve\.|[Rr]d\.|[Bb]lvd\.|[Dd]r\.|[Ll]n\.))?\s+(?:[A-Z][a-z]+)?(?:[\s-][A-Z][a-z]+)*)\b`:                                          "LOC",     // More detailed addresses
		`\b(?:[Mm]ount|[Ll]ake)\s+[A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*\b`:                                                                                                                   "LOC",     // Mountains and lakes
		`\b(?:[A-Z][a-z]{3,}\s+[A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*)\b`:                                                                                                                     "PRODUCT", // Potential product names (e.g., Coca Cola)
		`\b(?:[0-9]{1,2}(?:st|nd|rd|th)?\s+(?:[Jj]anuary|[Ff]ebruary|[Mm]arch|[Aa]pril|[Mm]ay|[Jj]une|[Jj]uly|[Aa]ugust|[Ss]eptember|[Oo]ctober|[Nn]ovember|[Dd]ecember)\s+[0-9]{4})\b`: "DATE",    // Dates with ordinal suffixes
		`\b(?:[0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4})\b`:                                                                                                                                      "DATE",    // Dates in different formats
		`\b(?:[0-9]{1,2}:\d{2}(?::\d{2})?(?:\s?(?:AM|PM))?)\b`:                                                                                                                          "TIME",    // Times
		`\bcreate\b`:                "ACTION",
		`\bdata\s+structure\s+of\b`: "DATA_STRUCTURE",
		`\b[A-Z][a-z]+\b`:           "OBJECT_NAME",
		`\bnamed\s+([a-zA-Z]+)\b`:   "NAME", // Capture the name
		`\b(?:create|delete|update)\s+(?:a\s+)?(webserver|database|handler)\s+named\s+([a-zA-Z]+)\b`: "COMMAND",
		`\bwith\s+(?:the\s+)?handlers?\s+named\s+([a-zA-Z]+(?:\s*,\s*[a-zA-Z]+)*)\b`:                 "HANDLERS",
		`\bdata\s+structure\s+of\s+([a-zA-Z]+)\s+with\s+(\d+)\s+fields\b`:                            "DATA_STRUCTURE",
		`\bset\s+the\s+port\s+to\s+(\d+)\b`:                                                          "PORT",
		`\bwith\s+(\d+)\s+fields\b`:                                                                  "FIELDS", // Capture the number of fields
		`\b(?:[Tt]he\s+)?[A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*\s+(?:[Ii]nc\.|[Ll]td\.|[Cc]orp\.)\b`:       "ORG",
	}

	// List of known action verbs
	var actionVerbs = map[string]bool{
		"create":       true,
		"delete":       true,
		"update":       true,
		"build":        true,
		"set":          true,
		"add":          true,
		"remove":       true,
		"modify":       true,
		"configure":    true,
		"deploy":       true,
		"run":          true,
		"execute":      true,
		"start":        true,
		"stop":         true,
		"generate":     true,
		"write":        true,
		"read":         true,
		"send":         true,
		"receive":      true,
		"connect":      true,
		"disconnect":   true,
		"attach":       true,
		"detach":       true,
		"upload":       true,
		"download":     true,
		"install":      true,
		"uninstall":    true,
		"copy":         true,
		"move":         true,
		"rename":       true,
		"open":         true,
		"close":        true,
		"find":         true,
		"search":       true,
		"get":          true,
		"put":          true,
		"post":         true,
		"list":         true,
		"show":         true,
		"display":      true,
		"view":         true,
		"edit":         true,
		"save":         true,
		"load":         true,
		"import":       true,
		"export":       true,
		"test":         true,
		"debug":        true,
		"validate":     true,
		"compile":      true,
		"link":         true,
		"publish":      true,
		"subscribe":    true,
		"authorize":    true,
		"authenticate": true,
		"encrypt":      true,
		"decrypt":      true,
		"compress":     true,
		"decompress":   true,
		"backup":       true,
		"restore":      true,
		"rollback":     true,
		"commit":       true,
		"push":         true,
		"pull":         true,
		"clone":        true,
		"fork":         true,
		"merge":        true,
		"revert":       true,
		"branch":       true,
	}
	return tokens, contractions, posTags, nerTags, actionVerbs
}
