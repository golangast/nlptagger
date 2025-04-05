package crf_model

// WordExample represents a single word in a sentence with its features and label.
type WordExample struct {
	Features map[string]string // Map of feature names to feature values (e.g., "word": "system", "pos": "NN")
	Label    string            // The correct semantic role label for this word (e.g., "Arg0")
}

// TrainingExample represents a single training sentence with labels.
type TrainingExample struct {
	WordExamples []WordExample // Slice of WordExamples for each word in the sentence.
}

// CRFModel represents the CRF model
type CRFModel struct {
	Features []string                      // List of features (e.g., "pos=VB", "dep=dobj")
	Labels   []string                      // List of labels (e.g., "Arg0", "Arg1")
	Weights  map[string]map[string]float64 // Weights for each feature and transition
	//Transition weights, example: Weight["Arg0"]["Arg1"] = 0.5 //Transition from Arg0 to Arg1
	//Feature weights, example: Weight["pos=VB"]["Arg1"] = 0.5 //If pos is VB, Arg1 is likely
}

// ViterbiOutput represents the result of the Viterbi algorithm
type ViterbiOutput struct {
	Labels []string
	Score  float64
}

// NewCRFModel creates a new CRF model with initialized weights
func NewCRFModel(features []string, labels []string) *CRFModel {
	// Initialize the CRF model with features, labels, and empty weights.
	model := &CRFModel{
		Features: features,
		Labels:   labels,
		Weights:  make(map[string]map[string]float64),
	}

	// Initialize the weights map for each label.
	for _, label := range labels {
		model.Weights[label] = make(map[string]float64)
	}

	return model
}
