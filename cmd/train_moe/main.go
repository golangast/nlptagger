package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"os/signal"
	"runtime"
	"runtime/pprof" // Added for profiling
	"strings"
	"syscall"

	"nlptagger/neural/moe"
	"nlptagger/neural/nn"
	. "nlptagger/neural/nn"
	mainvocab "nlptagger/neural/nnu/vocab"
	"nlptagger/neural/nnu/word2vec"
	"nlptagger/neural/semantic"
	. "nlptagger/neural/tensor"
	"nlptagger/neural/tokenizer"
)

// IntentTrainingExample represents a single training example with a query and its intents.
type IntentTrainingExample struct {
	Query          string                  `json:"query"`
	SemanticOutput semantic.SemanticOutput `json:"semantic_output"`
}

// IntentTrainingData represents the structure of the intent training data JSON.
type IntentTrainingData []IntentTrainingExample

// LoadIntentTrainingData loads the intent training data from a JSON file.
func LoadIntentTrainingData(filePath string) (*IntentTrainingData, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open training data file %s: %w", filePath, err)
	}
	defer file.Close()

	bytes, err := io.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("failed to read training data file %s: %w", filePath, err)
	}

	var data IntentTrainingData
	err = json.Unmarshal(bytes, &data)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal training data JSON from %s: %w", filePath, err)
	}

	return &data, nil
}

// TokenizeAndConvertToIDs tokenizes a text and converts tokens to their corresponding IDs, handling padding/truncation.
func TokenizeAndConvertToIDs(text string, tokenizer *tokenizer.Tokenizer, vocabulary *mainvocab.Vocabulary, maxLen int) ([]int, error) {

	tokenIDs, err := tokenizer.Encode(text)
	if err != nil {
		return nil, fmt.Errorf("failed to encode text: %w", err)
	}

	// Handle empty input strings by returning a slice with only the PaddingTokenID
	if len(tokenIDs) == 0 && maxLen > 0 {
		tokenIDs = make([]int, maxLen)
		for i := range tokenIDs {
			tokenIDs[i] = vocabulary.PaddingTokenID
		}
		return tokenIDs, nil
	} else if len(tokenIDs) == 0 {
		return []int{vocabulary.PaddingTokenID}, nil
	}

	if maxLen > 0 {
		if len(tokenIDs) > maxLen {
			tokenIDs = tokenIDs[:maxLen]
		} else if len(tokenIDs) < maxLen {
			paddingSize := maxLen - len(tokenIDs)
			padding := make([]int, paddingSize)
			for i := range padding {
				padding[i] = vocabulary.PaddingTokenID
			}
			tokenIDs = append(tokenIDs, padding...)
		}
	}
	return tokenIDs, nil
}

// TrainIntentMoEModel trains the MoEClassificationModel.
func TrainIntentMoEModel(model *moe.IntentMoE, data *IntentTrainingData, epochs int, learningRate float64, batchSize int, queryVocab, semanticOutputVocab *mainvocab.Vocabulary, queryTokenizer, semanticOutputTokenizer *tokenizer.Tokenizer, maxSequenceLength int, cpuProfileFile *os.File) error {

	if model == nil {
		return errors.New("cannot train a nil model")
	}
	if data == nil || len(*data) == 0 {
		return errors.New("no training data provided")
	}

	optimizer := NewOptimizer(model.Parameters(), learningRate, 1.0) // Using a clip value of 1.0

	// Learning rate scheduling parameters
	baseLR := learningRate
	minLR := learningRate / 10.0 // 0.00001
	totalBatches := (len(*data) + batchSize - 1) / batchSize
	totalSteps := epochs * totalBatches
	warmupSteps := totalBatches * 2 // Warmup for first 2 epochs
	currentStep := 0

	for epoch := 0; epoch < epochs; epoch++ {
		// Calculate scheduled sampling probability for logging
		scheduledSamplingProb := math.Min(0.5, float64(epoch)/float64(epochs*2))
		log.Printf("Epoch %d/%d (Scheduled Sampling: %.1f%%)", epoch+1, epochs, scheduledSamplingProb*100)
		totalLoss := 0.0
		numBatches := 0
		// Create batches for training
		for i := 0; i < len(*data); i += batchSize {
			end := i + batchSize
			if end > len(*data) {
				end = len(*data)
			}
			batch := (*data)[i:end]

			// Update learning rate with scheduling
			currentLR := calculateLearningRate(currentStep, totalSteps, warmupSteps, baseLR, minLR)
			if adamOpt, ok := optimizer.(*Adam); ok {
				adamOpt.SetLearningRate(currentLR)
			}
			currentStep++

			loss, err := trainIntentMoEBatch(model, optimizer, batch, queryVocab, semanticOutputVocab, queryTokenizer, semanticOutputTokenizer, maxSequenceLength, epoch, epochs)
			if err != nil {
				log.Printf("Error training batch: %v", err)
				continue // Or handle error more strictly
			}
			totalLoss += loss
			numBatches++

			// Log gradient norms every 5 batches for debugging
			if numBatches%5 == 0 {
				gradNorm := computeGradientNorm(model.Parameters())
				log.Printf("Batch %d: Loss=%.2f, GradNorm=%.4f, LR=%.6f", numBatches, loss, gradNorm, currentLR)
			}

			// if numBatches == 1 && memProfileFile != nil {
			// 	if err := pprof.WriteHeapProfile(memProfileFile); err != nil {
			// 		log.Printf("could not write memory profile for batch 1: %v", err)
			// 	}
			// }
		}
		if numBatches > 0 {
			log.Printf("Epoch %d, Average Loss: %f", epoch+1, totalLoss/float64(numBatches))
		}
	}

	return nil
}

// trainIntentMoEBatch performs a single training step on a batch of data.
func trainIntentMoEBatch(intentMoEModel *moe.IntentMoE, optimizer Optimizer, batch IntentTrainingData, queryVocab, semanticOutputVocab *mainvocab.Vocabulary, queryTokenizer, semanticOutputTokenizer *tokenizer.Tokenizer, maxSequenceLength int, epoch, totalEpochs int) (float64, error) {
	optimizer.ZeroGrad()

	batchSize := len(batch)

	inputIDsBatch := make([]int, batchSize*maxSequenceLength)
	semanticOutputIDsBatch := make([]int, batchSize*maxSequenceLength)

	for i, example := range batch {
		// Tokenize and pad query
		queryTokens, err := TokenizeAndConvertToIDs(example.Query, queryTokenizer, queryVocab, maxSequenceLength)
		if err != nil {
			return 0, fmt.Errorf("query tokenization failed for item %d: %w", i, err)
		}
		copy(inputIDsBatch[i*maxSequenceLength:(i+1)*maxSequenceLength], queryTokens)

		// Serialize and tokenize semantic output
		semanticOutputJSON, err := json.Marshal(example.SemanticOutput)
		if err != nil {
			return 0, fmt.Errorf("failed to marshal semantic output: %w", err)
		}

		trainingSemanticOutput := "<s> " + string(semanticOutputJSON) + " </s>"
		semanticOutputTokens, err := TokenizeAndConvertToIDs(trainingSemanticOutput, semanticOutputTokenizer, semanticOutputVocab, maxSequenceLength)
		if err != nil {
			return 0, fmt.Errorf("semantic output tokenization failed for item %d: %w", i, err)
		}
		// Verbose logging disabled for faster training
		// log.Printf("Query: %s", example.Query)
		// log.Printf("Tokenized Query: %v", queryTokens)
		// log.Printf("Semantic Output: %s", trainingSemanticOutput)
		// log.Printf("Tokenized Semantic Output: %v", semanticOutputTokens)
		copy(semanticOutputIDsBatch[i*maxSequenceLength:(i+1)*maxSequenceLength], semanticOutputTokens)
	}

	// Convert input IDs to a Tensor (embeddings will be handled by the model)
	inputTensor := NewTensor([]int{batchSize, maxSequenceLength}, convertIntsToFloat64s(inputIDsBatch), false)
	semanticOutputTensor := NewTensor([]int{batchSize, maxSequenceLength}, convertIntsToFloat64s(semanticOutputIDsBatch), false)

	// Calculate scheduled sampling probability: gradually increase from 0% to 50%
	// Formula: min(0.5, epoch / (totalEpochs * 2))
	scheduledSamplingProb := math.Min(0.5, float64(epoch)/float64(totalEpochs*2))

	// Forward pass through the IntentMoE model with scheduled sampling
	semanticOutputLogits, _, err := intentMoEModel.Forward(scheduledSamplingProb, inputTensor, semanticOutputTensor)
	if err != nil {
		return 0, fmt.Errorf("IntentMoE model forward pass failed: %w", err)
	}

	// Calculate loss for the semantic output
	semanticOutputLoss := 0.0
	// The decoder now produces maxSequenceLength-1 outputs
	semanticOutputGrads := make([]*Tensor, maxSequenceLength-1)
	entropyLoss := 0.0 // Entropy regularization term

	for t := 0; t < maxSequenceLength-1; t++ {
		targets := make([]int, batchSize)
		for i := 0; i < batchSize; i++ {
			// Target for step t (input t) is token at t+1
			targets[i] = semanticOutputIDsBatch[i*maxSequenceLength+t+1]
		}
		loss, grad := CrossEntropyLoss(semanticOutputLogits[t], targets, semanticOutputVocab.PaddingTokenID, 0.1)
		semanticOutputLoss += loss
		semanticOutputGrads[t] = grad

		// Entropy calculation commented out for training speed
		// Calculate entropy for diversity regularization
		// logits := semanticOutputLogits[t]
		// vocabSize := logits.Shape[1]
		// for i := 0; i < batchSize; i++ {
		// 	if targets[i] == semanticOutputVocab.PaddingTokenID {
		// 		continue
		// 	}
		// 	maxLogit := logits.Data[i*vocabSize]
		// 	for j := 1; j < vocabSize; j++ {
		// 		if logits.Data[i*vocabSize+j] > maxLogit {
		// 			maxLogit = logits.Data[i*vocabSize+j]
		// 		}
		// 	}
		// 	expSum := 0.0
		// 	for j := 0; j < vocabSize; j++ {
		// 		expSum += math.Exp(logits.Data[i*vocabSize+j] - maxLogit)
		// 	}
		// 	entropy := 0.0
		// 	for j := 0; j < vocabSize; j++ {
		// 		prob := math.Exp(logits.Data[i*vocabSize+j]-maxLogit) / expSum
		// 		if prob > 1e-10 {
		// 			entropy -= prob * math.Log(prob)
		// 		}
		// 	}
		// 	entropyLoss -= entropy
		// }
	}

	// Combine losses with entropy regularization weight
	entropyWeight := 0.01 // Small weight to not dominate main loss
	totalLoss := semanticOutputLoss + entropyWeight*entropyLoss

	// Backward pass
	err = intentMoEModel.Backward(semanticOutputGrads...)
	if err != nil {
		return 0, fmt.Errorf("IntentMoE model backward pass failed: %w", err)
	}

	optimizer.Step()

	// Per-batch example logging commented out for speed
	// Only log loss, not decoded examples
	// predictedIDs, err := intentMoEModel.GreedySearchDecode(contextVector, 20, semanticOutputVocab.GetTokenID("<s>"), semanticOutputVocab.GetTokenID("</s>"), 1.0)
	// if err != nil {
	// 	log.Printf("Error decoding guessed sentence: %v", err)
	// } else {
	// 	guessedSentence, err := semanticOutputTokenizer.Decode(predictedIDs)
	// 	if err != nil {
	// 		log.Printf("Error decoding guessed sentence: %v", err)
	// 	} else {
	// 		log.Printf("Guessed semantic output: %s", guessedSentence)
	// 	}
	// 	targetJSON, _ := json.Marshal(batch[0].SemanticOutput)
	// 	log.Printf("Target semantic output: %s", string(targetJSON))
	// }

	return totalLoss, nil
}

// computeGradientNorm calculates the L2 norm of all parameter gradients
func computeGradientNorm(params []*Tensor) float64 {
	totalNorm := 0.0
	for _, param := range params {
		if param.Grad != nil {
			for _, g := range param.Grad.Data {
				totalNorm += g * g
			}
		}
	}
	return math.Sqrt(totalNorm)
}

// calculateLearningRate computes the learning rate with warmup and cosine decay
func calculateLearningRate(step, totalSteps, warmupSteps int, baseLR, minLR float64) float64 {
	if step < warmupSteps {
		// Linear warmup
		return baseLR * float64(step) / float64(warmupSteps)
	}
	// Cosine decay after warmup
	progress := float64(step-warmupSteps) / float64(totalSteps-warmupSteps)
	return minLR + (baseLR-minLR)*0.5*(1+math.Cos(math.Pi*progress))
}

func convertIntsToFloat64s(input []int) []float64 {
	output := make([]float64, len(input))
	for i, v := range input {
		output[i] = float64(v)
	}
	return output
}

func convertW2VVocab(w2vVocab map[string]int) *mainvocab.Vocabulary {
	vocab := mainvocab.NewVocabulary()
	vocab.WordToToken = w2vVocab
	maxID := 0
	for _, id := range w2vVocab {
		if id > maxID {
			maxID = id
		}
	}
	vocab.TokenToWord = make([]string, maxID+1)
	for token, id := range w2vVocab {
		vocab.TokenToWord[id] = token
	}
	return vocab
}

func BuildVocabularies(dataPath string) (*mainvocab.Vocabulary, *mainvocab.Vocabulary, error) {
	queryVocabulary := mainvocab.NewVocabulary()
	semanticOutputVocabulary := mainvocab.NewVocabulary()

	semanticTrainingData, err := LoadIntentTrainingData(dataPath)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to load semantic training data from %s: %w", dataPath, err)
	}

	for _, pair := range *semanticTrainingData {
		// Use the same tokenizer logic as during inference to build the vocabulary
		tokenizedQuery := tokenizer.Tokenize(strings.ToLower(pair.Query))
		for _, word := range tokenizedQuery {
			queryVocabulary.AddToken(word)
		}

		semanticOutputJSON, err := json.Marshal(pair.SemanticOutput)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to marshal semantic output: %w", err)
		}

		// Add BOS and EOS tokens to the sentence when building the vocabulary
		trainingSemanticOutput := "<s> " + string(semanticOutputJSON) + " </s>"
		tokenizedSemanticOutput := tokenizer.Tokenize(trainingSemanticOutput)
		for _, word := range tokenizedSemanticOutput {
			semanticOutputVocabulary.AddToken(word)
		}
	}

	// Explicitly add BOS and EOS tokens to the sentence vocabulary
	semanticOutputVocabulary.BosID = semanticOutputVocabulary.GetTokenID("<s>")
	semanticOutputVocabulary.EosID = semanticOutputVocabulary.GetTokenID("</s>")

	return queryVocabulary, semanticOutputVocabulary, nil
}

func main() {
	const trainingDataPath = "./trainingdata/intent_data.json"
	const semanticTrainingDataPath = "./trainingdata/semantic_output_data.json"
	const word2vecModelPath = "gob_models/word2vec_model.gob"

	// Start CPU profiling
	cpuProfileFile, err := os.Create("cpu.prof")
	if err != nil {
		log.Fatal("could not create CPU profile: ", err)
	}
	if err := pprof.StartCPUProfile(cpuProfileFile); err != nil {
		log.Fatal("could not start CPU profile: ", err)
	}

	// Set up a channel to listen for interrupt signals
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	// Goroutine to handle graceful shutdown on signal
	go func() {
		<-sigChan // Block until a signal is received
		log.Println("Received interrupt signal. Stopping CPU and Memory profiles and closing files.")
		pprof.StopCPUProfile()
		cpuProfileFile.Close()

		// Write heap profile on exit
		memProfFile, err := os.Create("mem.prof")
		if err != nil {
			log.Fatal("could not create memory profile: ", err)
		}
		if err := pprof.WriteHeapProfile(memProfFile); err != nil {
			log.Fatal("could not write memory profile: ", err)
		}
		memProfFile.Close()

		os.Exit(0) // Exit gracefully
	}()

	// Define training parameters
	epochs := 20           // Increased from 3 for better model quality (Step 1: Gradual Scaling)
	learningRate := 0.0001 // Reduced from 0.001 for more stable training
	batchSize := 16        // Drastically reduced from 128 to reduce memory and computation
	semanticOutputVocabularySavePath := "gob_models/semantic_output_vocabulary.gob"

	// Load Word2Vec model
	word2vecModel, err := word2vec.LoadModel(word2vecModelPath)
	if err != nil {
		log.Fatalf("Failed to load Word2Vec model: %v", err)
	}

	// Create query vocabulary from word2vec model
	queryVocabulary := convertW2VVocab(word2vecModel.Vocabulary)

	// Try to load other vocabularies first
	semanticOutputVocabulary, err := mainvocab.LoadVocabulary(semanticOutputVocabularySavePath)
	if err != nil {
		log.Println("Failed to load semantic output vocabulary, creating a new one.")
	}

	if semanticOutputVocabulary == nil {
		log.Println("Building vocabularies from scratch...")
		_, semanticOutputVocabulary, err = BuildVocabularies(semanticTrainingDataPath)
		if err != nil {
			log.Fatalf("Failed to build vocabularies: %v", err)
		}
	}

	log.Printf("Query Vocabulary (after load/create): Size=%d", len(queryVocabulary.WordToToken))
	log.Printf("Semantic Output Vocabulary (after load/create): Size=%d", len(semanticOutputVocabulary.WordToToken))

	// Load Intent training data
	semanticTrainingData, err := LoadIntentTrainingData(semanticTrainingDataPath)
	if err != nil {
		log.Fatalf("Failed to load semantic training data from %s: %v", semanticTrainingDataPath, err)
	}
	log.Printf("Loaded %d training examples from %s.", len(*semanticTrainingData), semanticTrainingDataPath)

	// Load WikiQA training data
	const wikiQATrainingDataPath = "./trainingdata/generated_wikiqa_intents.json"
	wikiQATrainingData, err := LoadIntentTrainingData(wikiQATrainingDataPath)
	if err != nil {
		log.Printf("Warning: Failed to load WikiQA training data from %s: %v. Proceeding without it.", wikiQATrainingDataPath, err)
	} else {
		log.Printf("Loaded %d training examples from %s.", len(*wikiQATrainingData), wikiQATrainingDataPath)
		// Sample WikiQA data to improve training speed
		const maxWikiQASamples = 1000 // Increased from 500 for Step 2: More training data
		if len(*wikiQATrainingData) > maxWikiQASamples {
			*wikiQATrainingData = (*wikiQATrainingData)[:maxWikiQASamples]
			log.Printf("Sampled WikiQA data to %d examples for faster training.", maxWikiQASamples)
		}
		// Merge datasets
		*semanticTrainingData = append(*semanticTrainingData, *wikiQATrainingData...)
		log.Printf("Total training examples after merging: %d", len(*semanticTrainingData))
	}

	// After vocabularies are fully populated, determine vocab sizes and create/load model
	inputVocabSize := len(queryVocabulary.WordToToken)
	semanticOutputVocabSize := len(semanticOutputVocabulary.WordToToken)
	embeddingDim := 64 // Increased from 8 to 64 for better representation
	numExperts := 1
	maxSequenceLength := 32 // Max length for input query and output description (reduced for memory)
	maxAttentionHeads := 1

	log.Printf("Query Vocabulary Size: %d", inputVocabSize)
	log.Printf("Semantic Output Vocabulary Size: %d", semanticOutputVocabSize)
	log.Printf("Embedding Dimension: %d", embeddingDim)
	log.Printf("Word2Vec Model Vocab Size: %d", word2vecModel.VocabSize)
	log.Printf("Word2Vec Model Vector Size: %d", word2vecModel.VectorSize)
	log.Printf("Number of Experts: %d", numExperts)

	var intentMoEModel *moe.IntentMoE // Declare intentMoEModel here

	modelSavePath := "gob_models/moe_classification_model.gob"

	// Always create a new IntentMoE model for now to debug gob loading
	log.Printf("Creating a new IntentMoE model.")
	embeddingDim = 64      // Reduced from 128 to speed up computation
	hiddenSize := 64       // Reduced from 128
	expertHiddenDim := 128 // Reduced from 256
	numExperts = 2         // Reduced from 4
	k := 1                 // Reduced from 2 - use only 1 expert at a time
	// epochs is set above - don't override here
	maxAttentionHeads = 2 // Reduced from 4
	numLayers := 1        // Reduced from 2 - single layer LSTM
	dropoutRate := 0.2    // Reduced from 0.3

	// 1. Embedding
	embedding := nn.NewEmbedding(inputVocabSize, embeddingDim)
	if word2vecModel != nil {
		embedding.LoadPretrainedWeights(word2vecModel.WordVectors)
	}

	// 2. MoE Encoder
	expertBuilder := func(expertIdx int) (moe.Expert, error) {
		return moe.NewFeedForwardExpert(embeddingDim, expertHiddenDim, embeddingDim)
	}
	moeLayer, err := moe.NewMoELayer(embeddingDim, numExperts, k, expertBuilder)
	if err != nil {
		log.Fatalf("Failed to create MoE layer: %v", err)
	}

	// 3. RNN Decoder with increased capacity and dropout
	decoder, err := moe.NewRNNDecoder(embeddingDim, semanticOutputVocabSize, hiddenSize, maxAttentionHeads, numLayers, dropoutRate)
	if err != nil {
		log.Fatalf("Failed to create decoder: %v", err)
	}

	// 4. Create IntentMoE model
	intentMoEModel = &moe.IntentMoE{
		Embedding:         embedding,
		Encoder:           moeLayer,
		Decoder:           decoder,
		SentenceVocabSize: semanticOutputVocabSize,
	}

	// Training Loop
	// epochs = 5 // Removed redundant assignment

	// Create tokenizers once after vocabularies are loaded/created
	queryTokenizer, err := tokenizer.NewTokenizer(queryVocabulary)
	if err != nil {
		log.Fatalf("Failed to create query tokenizer: %v", err)
	}
	semanticOutputTokenizer, err := tokenizer.NewTokenizer(semanticOutputVocabulary)
	if err != nil {
		log.Fatalf("Failed to create semantic output tokenizer: %v", err)
	}

	// Create a dedicated file for memory profiling after the first batch
	// memBatch1ProfileFile, err := os.Create("mem_batch_1.prof")
	// if err != nil {
	// 	log.Fatal("could not create memory profile for batch 1: ", err)
	// }

	// Train the model
	err = TrainIntentMoEModel(intentMoEModel, semanticTrainingData, epochs, learningRate, batchSize, queryVocabulary, semanticOutputVocabulary, queryTokenizer, semanticOutputTokenizer, maxSequenceLength, cpuProfileFile)
	if err != nil {
		log.Fatalf("Failed to train IntentMoE model: %v", err)
	}

	// Detach the model from the computation graph to allow for clean serialization
	log.Println("Detaching model from computation graph...")
	DetachModel(intentMoEModel)

	// Save the trained model
	fmt.Printf("Saving IntentMoE model to %s\n", modelSavePath)
	memSaveProfileFile, err := os.Create("mem_save.prof")
	if err != nil {
		log.Fatal("could not create memory profile for saving: ", err)
	}
	defer memSaveProfileFile.Close()
	if err := pprof.WriteHeapProfile(memSaveProfileFile); err != nil {
		log.Fatal("could not write memory profile for saving: ", err)
	}
	modelFile, err := os.Create(modelSavePath)
	if err != nil {
		log.Fatalf("Failed to create model file: %v", err)
	}
	defer modelFile.Close()
	err = moe.SaveIntentMoEModelToGOB(intentMoEModel, modelFile)
	if err != nil {
		log.Fatalf("Failed to save IntentMoE model: %v", err)
	}

	// Save the vocabularies
	queryVocabularySavePath := "gob_models/query_vocabulary.gob"
	err = queryVocabulary.Save(queryVocabularySavePath)
	if err != nil {
		log.Fatalf("Failed to save query vocabulary: %v", err)
	}
	err = semanticOutputVocabulary.Save(semanticOutputVocabularySavePath)
	if err != nil {
		log.Fatalf("Failed to save semantic output vocabulary: %v", err)
	}
}

// DetachModel removes the computation graph (gradients and creators) from the model parameters
// to ensure that only the weights are saved. This prevents serialization issues and reduces file size.
func DetachModel(model *moe.IntentMoE) {
	params := model.Parameters()
	for _, param := range params {
		param.Grad = nil
		param.Creator = nil
		param.Mask = nil
		param.Operation = nil
		// We keep RequiresGrad as is, or set it to false if we want to freeze the model.
		// For saving, it doesn't strictly matter for gob if we don't save Creator,
		// but setting Creator to nil is the key.
	}

	// Clear decoder state which might hold references to the computation graph
	if model.Decoder != nil {
		model.Decoder.InitialHiddenState = nil
		model.Decoder.InitialCellState = nil

		// Clear LSTM cells state
		if model.Decoder.LSTM != nil {
			for _, layer := range model.Decoder.LSTM.Cells {
				for _, cell := range layer {
					cell.InputTensor = nil
					cell.PrevHidden = nil
					cell.PrevCell = nil
				}
			}
		}
	}

	log.Println("Model detached from computation graph.")
	runtime.GC() // Force garbage collection to free up memory before saving
}
