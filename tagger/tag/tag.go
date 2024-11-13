package tag

type Features struct {
	WebServerKeyword  float64
	PreviousWord      float64
	NextWord          float64
	PreviousArticle   float64
	NextPreposition   float64
	NextOfIn          float64
	SpecialCharacters float64
	NameSuffix        float64
	PreviousTag       float64
	NextTag           float64
	FollowedByNumber  float64
	IsNoun            float64
}
type Tag struct {
	PosTag          []string
	NerTag          []string
	PhraseTag       []string
	Tokens          []string
	DepRelationsTag []string
	Features        []Features
	Dependencies    []Dependency `json:"dependencies"` // Or "dependencyTag" if that's what the key is in your json
	Epoch           int
	IsName          bool
	Token           string
	Sentence        string
	Tags            []Tag
}
type Dependency struct {
	Head int    `json:"head"`
	Dep  string `json:"dep"` // or Dependent if that's what your JSON has

}
