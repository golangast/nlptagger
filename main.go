package main

import (
	"fmt"
	"log"
	"strings"

	"tagging/tokenize"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func main() {
	text := "create a webserver named doggy with the handler kitty that has the data structure moose with 4 string fields"
	entities := tokenize.Tokenize(text)
	fmt.Println("Entities:", entities)

	// 1. Prepare Training Data (Example)
	trainingData := []struct {
		features []float64
		isName   bool
	}{
		{[]float64{1, 0, 1, 1, 1, 1, 1}, true},  // Example features for a name
		{[]float64{0, 1, 0, 1, 0, 0, 0}, false}, // Example features for not a name
	}

	// 2. Create and Train the Model
	inputSize := len(trainingData[0].features) // Number of features
	hiddenSize := 4                            // Number of neurons in the hidden layer
	outputSize := 1                            // Number of output neurons (for binary classification)
	// Number of neurons in the hidden layer

	// Model creation
	model := NewNameFinder(inputSize, outputSize, hiddenSize)
	// Define a cost function (using basic operations)
	y := gorgonia.NewMatrix(model.g, gorgonia.Float64, gorgonia.WithShape(1, 1), gorgonia.WithName("y"))
	// ... in your main function ...
	cost := gorgonia.Must(gorgonia.Mean(gorgonia.Must(gorgonia.Abs(gorgonia.Must(gorgonia.Sub(model.output, y))))))
	// Create a solver (Stochastic Gradient Descent)
	solver := gorgonia.NewAdamSolver(
		gorgonia.WithLearnRate(0.01),
		gorgonia.WithBeta1(0.8),   // Adjust the first moment decay rate (default: 0.9)
		gorgonia.WithBeta2(0.999), // Adjust the second moment decay rate (default: 0.999)
	)
	// Create the TapeMachine
	vm := gorgonia.NewTapeMachine(model.g, gorgonia.BindDualValues(model.Learnables()...))

	defer vm.Close()
	// Training loop
	epochs := 1000
	for i := 0; i < epochs; i++ {
		for _, data := range trainingData {
			inputData := tensor.New(tensor.WithShape(1, inputSize), tensor.WithBacking(data.features))
			outputData := tensor.New(tensor.WithShape(1, 1), tensor.WithBacking([]float64{boolToFloat(data.isName)}))
			gorgonia.Let(model.input, inputData)
			gorgonia.Let(y, outputData)
			if err := vm.RunAll(); err != nil {
				log.Fatal(err)
			}

			if grads, err := gorgonia.Grad(cost, model.Learnables()...); err != nil {

			} else {
				var valueGrads []gorgonia.ValueGrad
				for _, grad := range grads {
					valueGrads = append(valueGrads, grad)
				}
				if err := solver.Step(valueGrads); err != nil {
					//some char wont have grads
					if strings.Contains(err.Error(), "No Grad found for") {
					} else {
						// Handle error
						fmt.Println(err)
					}

				}
			}
			if err := vm.RunAll(); err != nil {
				log.Fatal(cost)
			}
			for i, ent := range entities {
				fmt.Printf("Epoch %d, Cost: %f isName: %t Token: %s Type: %s \n", i, cost.Value(), data.isName, ent.Text, ent.Type)
			}

			vm.Reset()

		}

	}
}

type Entity struct {
	Text       string
	Type       string
	Tag        string
	StartIndex int
	EndIndex   int
}

type MyLSTM struct {
	g *gorgonia.ExprGraph
	// Weights and biases (gorgonia.Nodes)
	Wi, Wf, Wo, Wc *gorgonia.Node
	Bi, Bf, Bo, Bc *gorgonia.Node
	hiddenSize     int
}

func NewMyLSTM(g *gorgonia.ExprGraph, inputSize, hiddenSize int) *MyLSTM {
	// Initialize weights and biases
	Wi := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(hiddenSize, inputSize), gorgonia.WithName("Wi"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	Wf := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(hiddenSize, inputSize), gorgonia.WithName("Wf"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	Wo := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(hiddenSize, inputSize), gorgonia.WithName("Wo"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	Wc := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(hiddenSize, inputSize), gorgonia.WithName("Wc"), gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	Bi := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(hiddenSize, 1), gorgonia.WithName("Bi"), gorgonia.WithInit(gorgonia.Zeroes()))
	Bf := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(hiddenSize, 1), gorgonia.WithName("Bf"), gorgonia.WithInit(gorgonia.Zeroes()))
	Bo := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(hiddenSize, 1), gorgonia.WithName("Bo"), gorgonia.WithInit(gorgonia.Zeroes()))
	Bc := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(hiddenSize, 1), gorgonia.WithName("Bc"), gorgonia.WithInit(gorgonia.Zeroes()))
	return &MyLSTM{
		g:          g,
		Wi:         Wi,
		Wf:         Wf,
		Wo:         Wo,
		Wc:         Wc,
		Bi:         Bi,
		Bf:         Bf,
		Bo:         Bo,
		Bc:         Bc,
		hiddenSize: hiddenSize,
	}
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Helper function to convert bool to float64
func boolToFloat(b bool) float64 {
	if b {
		return 1.0
	}
	return 0.0
}

// Define the model architecture
type NameFinder struct {
	g           *gorgonia.ExprGraph
	input       *gorgonia.Node
	hiddenLayer *Layer // Example: One hidden layer
	outputLayer *Layer
	output      *gorgonia.Node // Add this field to store the output
}

// Learnables returns the learnable parameters of the model
func (nf *NameFinder) Learnables() gorgonia.Nodes {
	return gorgonia.Nodes{nf.hiddenLayer.Weights, nf.hiddenLayer.Biases, nf.outputLayer.Weights, nf.outputLayer.Biases}
}
func NewNameFinder(inputSize, outputSize, hiddenSize int) *NameFinder {
	g := gorgonia.NewGraph()
	input := gorgonia.NewMatrix(g, gorgonia.Float64, gorgonia.WithShape(1, inputSize), gorgonia.WithName("input"))
	// Create layers
	hiddenLayer := NewLayer(g, inputSize, hiddenSize, rectify)
	outputLayer := NewLayer(g, hiddenSize, outputSize, sigmoid)
	// Initialize output node (important!)
	var output *gorgonia.Node
	hiddenOutput := hiddenLayer.Forward(input)
	output = outputLayer.Forward(hiddenOutput)
	return &NameFinder{
		g:           g,
		input:       input,
		hiddenLayer: hiddenLayer,
		outputLayer: outputLayer,
		output:      output,
	}
}
func rectify(x *gorgonia.Node) *gorgonia.Node {
	retVal, err := gorgonia.Rectify(x)
	if err != nil {
		panic(err) // Or handle the error more gracefully
	}
	return retVal
}
func sigmoid(x *gorgonia.Node) *gorgonia.Node {
	retVal, err := gorgonia.Sigmoid(x)
	if err != nil {
		panic(err) // Or handle the error more gracefully
	}
	return retVal
}

// Forward pass
func (nf *NameFinder) Forward(input tensor.Tensor) (tensor.Tensor, error) {
	if input.Shape()[0] != 1 {
		err := input.Reshape(1, input.Shape()[0])
		if err != nil {
			return nil, err
		}
	}
	gorgonia.Let(nf.input, input)
	// Forward pass through layers
	hiddenOutput := nf.hiddenLayer.Forward(nf.input)
	nf.output = nf.outputLayer.Forward(hiddenOutput) // Store the output in the struct field
	fmt.Println("Input shape:", nf.input.Shape(), "type:", nf.input.Type())
	fmt.Println(nf.output.Value())

	outputValue := nf.output.Value()
	return outputValue.(tensor.Tensor), nil

}

type Token struct {
	Text  string // The actual text of the token
	Type  string // Part-of-speech tag or other type information (optional)
	Index int    // Position of the token in the sentence
	// ... other fields as needed ...
}

// extractFeaturesForNoun extracts features for a noun entity.
func extractFeaturesForNoun(entity Entity, tokens []Token) []float64 {
	features := make([]float64, 0)
	// Feature 1: Is the first letter capitalized?
	if isCapitalized(entity.Text) {
		features = append(features, 1.0)
	} else {
		features = append(features, 0.0)
	}
	// Feature 2: Does the word contain a number?
	if containsNumber(entity.Text) {
		features = append(features, 1.0)
	} else {
		features = append(features, 0.0)
	}
	// Feature 3: Is the word in a list of known titles?
	titles := []string{"Mr.", "Ms.", "Dr.", "Prof."}
	if contains(titles, entity.Text) {
		features = append(features, 1.0)
	} else {
		features = append(features, 0.0)
	}
	// Feature 4: Is the previous word a title?
	if entity.StartIndex > 0 {
		prevToken := tokens[entity.StartIndex-1]
		if contains(titles, prevToken.Text) {
			features = append(features, 1.0)
		} else {
			features = append(features, 0.0)
		}
	} else {
		features = append(features, 0.0)
	}
	// Feature 5: Word length
	wordLength := float64(len(entity.Text))
	features = append(features, wordLength)
	// Feature 6: Is the previous word an article ("a", "an", "the")?
	if entity.StartIndex > 0 {
		prevToken := tokens[entity.StartIndex-1]
		if prevToken.Text == "a" || prevToken.Text == "an" || prevToken.Text == "the" {
			features = append(features, 1.0)
		} else {
			features = append(features, 0.0)
		}
	} else {
		features = append(features, 0.0)
	}
	// Feature 7: Is the next word a preposition ("of", "in", etc.)?
	if entity.EndIndex < len(tokens) {
		nextToken := tokens[entity.EndIndex]
		if nextToken.Text == "of" || nextToken.Text == "in" {
			features = append(features, 1.0)
		} else {
			features = append(features, 0.0)
		}
	} else {
		features = append(features, 0.0)
	}
	return features
}

// Helper functions for feature extraction
func isCapitalized(text string) bool {
	return strings.ToUpper(text[:1]) == text[:1]
}
func containsNumber(text string) bool {
	for _, char := range text {
		if char >= '0' && char <= '9' {
			return true
		}
	}
	return false
}
func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

/*
````````````````````````````````````````````````````````````
````````````````````````````````````````````````````````````
*/
type Layer struct {
	g          *gorgonia.ExprGraph
	Weights    *gorgonia.Node
	Biases     *gorgonia.Node
	activation func(*gorgonia.Node) *gorgonia.Node // Activation function
}

func NewLayer(g *gorgonia.ExprGraph, inputSize, outputSize int, activation func(*gorgonia.Node) *gorgonia.Node) *Layer {
	// Initialize weights and biases using Gorgonia functions (e.g., gorgonia.NewMatrix)
	weights := gorgonia.NewMatrix(g, gorgonia.Float64,
		gorgonia.WithShape(outputSize, inputSize),
		gorgonia.WithName("Weights"),
		gorgonia.WithInit(gorgonia.GlorotN(1.0))) // Glorot initialization
	biases := gorgonia.NewMatrix(g, gorgonia.Float64,
		gorgonia.WithShape(outputSize, 1),
		gorgonia.WithName("Biases"),
		gorgonia.WithInit(gorgonia.Zeroes()))
	return &Layer{
		g:          g,
		Weights:    weights,
		Biases:     biases,
		activation: activation,
	}
}
func (l *Layer) Forward(input *gorgonia.Node) *gorgonia.Node {
	// Perform matrix multiplication, add bias, and apply activation function
	weightedSum := gorgonia.Must(gorgonia.Add(gorgonia.Must(gorgonia.Mul(input, gorgonia.Must(gorgonia.Transpose(l.Weights)))), gorgonia.Must(gorgonia.Transpose(l.Biases))))

	return l.activation(weightedSum)
}
