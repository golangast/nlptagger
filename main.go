package main

import (
	"encoding/json"
	"flag"
	"log"
	"os"

	"github.com/golangast/nlptagger/neural/nnu/bartsimple"
	"github.com/golangast/nlptagger/neural/nnu/bert"
	"github.com/golangast/nlptagger/neural/nnu/bert/cli"
	"github.com/golangast/nlptagger/neural/nnu/word2vec"
	"github.com/golangast/nlptagger/tagger/nertagger"
	"github.com/golangast/nlptagger/tagger/postagger"
	vocabbert "github.com/golangast/nlptagger/neural/nnu/bert/vocab"
)

var (
	trainBart    = flag.Bool("train-bart", false, "Enable BART model training")	
	trainBert    = flag.Bool("train-bert", false, "Enable BERT model training")

	epochs       = flag.Int("epochs", 10, "Number of training epochs")
	learningRate = flag.Float64("lr", 0.001, "Learning rate for training")
	bartDataPath = flag.String("bart-data", "trainingdata/bartdata/bartdata.json", "Path to BART training data")
	bertDataPath = flag.String("bert-data", "trainingdata/bertdata/bert.json", "Path to BERT training data")
	dimModel     = flag.Int("dim", 64, "Dimension of the model")
	numHeads     = flag.Int("heads", 4, "Number of attention heads")
	maxSeqLength = flag.Int("maxlen", 64, "Maximum sequence length")
	batchSize    = flag.Int("batchsize", 4, "Batch size for training")
)

func main() {
	flag.Parse()

	// Define paths
	const bartModelPath = "gob_models/simplified_bart_model.gob"
	const trainingDataPath = "trainingdata/tagdata/nlp_training_data.json"
	const vocabPath = "gob_models/vocabulary.gob"

	vocabulary, err := vocabbert.SetupVocabulary(vocabPath, trainingDataPath)
	if err != nil {
		log.Fatalf("Failed to set up vocabulary: %v", err)
	}

	var bartModel *bartsimple.SimplifiedBARTModel // Declare here

	bartModel, err = bert.SetupModel(bartModelPath, vocabulary, *dimModel, *numHeads, *maxSeqLength)
	if err != nil {
		log.Fatalf("Failed to set up BART model: %v", err)
	}

	if *trainBart {
		bert.RunTraining(bartModel, *bartDataPath, bartModelPath, *epochs, *learningRate, *batchSize)
		return // Exit after training
	}

	// BERT model setup and training
	bertConfig := bert.BertConfig{
		VocabSize:             len(vocabulary.WordToToken),
		HiddenSize:            *dimModel,
		NumHiddenLayers:       2, // Example value
		NumAttentionHeads:     *numHeads,
		IntermediateSize:      *dimModel * 4, // Example value
		MaxPositionEmbeddings: *maxSeqLength,
		TypeVocabSize:         2, // Example value
		LayerNormEps:          1e-12,
		HiddenDropoutProb:     0.1,
		Vocabulary:            vocabulary,
	}

	// Pre-train Word2Vec embeddings
	w2v := &word2vec.SimpleWord2Vec{
		VectorSize: *dimModel,
		HiddenSize: *dimModel,
		Window: 5,
		Epochs: 100,
		LearningRate: 0.025,
		MinWordFrequency: 1,
	}
	bertTrainingData, err := bert.LoadTrainingData(*bertDataPath)
	if err != nil {
		log.Fatalf("Error loading BERT training data: %v", err)
	}
	var sentences []string
	for _, example := range bertTrainingData {
		sentences = append(sentences, example.Text)
	}
	w2v.Train(sentences)
	word2vecEmbeddings := word2vec.ConvertToMap(w2v.WordVectors, w2v.Vocabulary)

	var bertModel *bert.BertModel
	
		bertModel, err = bert.Train(bertConfig, bertTrainingData, *epochs, *learningRate, vocabulary, word2vecEmbeddings)
		if err != nil {
			log.Fatalf("BERT model training failed: %v", err)
		}
		bertModel.TrainingData = bertTrainingData 
		tokenizer, err := bartsimple.NewTokenizer(
			vocabulary,
			vocabulary.BeginningOfSentenceID,
			vocabulary.EndOfSentenceID,
			vocabulary.PaddingTokenID,
			vocabulary.UnknownTokenID,
		)
		if err != nil {
			log.Fatalf("Failed to create tokenizer: %v", err)
		}

		// Make sure tokenizer is initialized before this block!
		for i := range bertModel.TrainingData {
			ex := &bertModel.TrainingData[i]
			if ex.Embedding != nil && len(ex.Embedding) == bertConfig.HiddenSize {
				continue // Already has a valid embedding
			}
			tokenIDs, _ := tokenizer.Encode(ex.Text)
			inputTensor := bert.NewTensor(nil, []int{1, len(tokenIDs)}, false)
			for j, id := range tokenIDs {
				inputTensor.Data[j] = float64(id)
			}
			tokenTypeIDs := bert.NewTensor(make([]float64, len(tokenIDs)), []int{1, len(tokenIDs)}, false)

			// Generate POS and NER tags
			taggedText := postagger.Postagger(ex.Text)
			nerTaggedText := nertagger.Nertagger(taggedText)

			posTagIDsData := make([]float64, len(nerTaggedText.PosTag))
			for j, tag := range nerTaggedText.PosTag {
				posTagIDsData[j] = float64(postagger.PosTagToIDMap()[tag])
			}
			posTagIDs := bert.NewTensor(posTagIDsData, []int{1, len(posTagIDsData)}, false)

			nerTagIDsData := make([]float64, len(nerTaggedText.NerTag))
			for j, tag := range nerTaggedText.NerTag {
				nerTagIDsData[j] = float64(nertagger.NerTagToIDMap()[tag])
			}
			nerTagIDs := bert.NewTensor(nerTagIDsData, []int{1, len(nerTagIDsData)}, false)

			embeddingOutput := bertModel.Embeddings.Forward(inputTensor, tokenTypeIDs, posTagIDs, nerTagIDs)
			sequenceOutput, _ := bertModel.Encoder.Forward(embeddingOutput)
			pooledOutput, _ := bertModel.Pooler.Forward(sequenceOutput)
			ex.Embedding = make([]float64, len(pooledOutput.Data))
			copy(ex.Embedding, pooledOutput.Data)
		}
	
		// After precomputing embeddings for all examples:
		file, err := os.Create(*bertDataPath)
		if err != nil {
			log.Fatalf("Could not save training data with embeddings: %v", err)
		}
		defer file.Close()
		if err := json.NewEncoder(file).Encode(bertModel.TrainingData); err != nil {
			log.Fatalf("Could not encode training data with embeddings: %v", err)
		}

	cli.RunInference(bartModel, bertModel, bertConfig, tokenizer)
}



//go run main.go -train-bert -bert-data=trainingdata/bertdata/bert.json -epochs=10 -lr=0.001  