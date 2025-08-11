package main

import (
	"encoding/json"
	"flag"
	"log"
	"os"

	"github.com/golangast/nlptagger/neural/nnu/bartsimple"
	"github.com/golangast/nlptagger/neural/nnu/bert"
	"github.com/golangast/nlptagger/neural/nnu/bert/cli"
	vocabbert "github.com/golangast/nlptagger/neural/nnu/bert/vocab"
	"github.com/golangast/nlptagger/neural/nnu/word2vec"
	"github.com/golangast/nlptagger/tagger/nertagger"
	"github.com/golangast/nlptagger/tagger/postagger"
)

var (
	trainBart    = flag.Bool("train-bart", false, "Enable BART model training")	
	trainBert    = flag.Bool("train-bert", false, "Enable BERT model training")

	epochs       = flag.Int("epochs", 10, "Number of training epochs")
	learningRate = flag.Float64("lr", 0.001, "Learning rate for training")
	bartDataPath = flag.String("bart-data", "trainingdata/bartdata/bartdata.json", "Path to BART training data")
	bertDataPath = flag.String("bert-data", "trainingdata/bertdata/bert.json", "Path to BERT training data")
	dimModel     = flag.Int("dim", 100, "Dimension of the model")
	numHeads     = flag.Int("heads", 4, "Number of attention heads")
	maxSeqLength = flag.Int("maxlen", 64, "Maximum sequence length")
	batchSize    = flag.Int("batchsize", 4, "Batch size for training")
)

func main() {
	flag.Parse()

	// Define paths
	const bartModelPath = "gob_models/simplified_bart_model.gob"
	const trainingDataPath = "trainingdata/bartdata/explanation_data.json"
	const vocabPath = "gob_models/vocabulary.gob"

	vocabulary, err := vocabbert.SetupVocabulary(vocabPath, trainingDataPath)
	if err != nil {
		log.Fatalf("Failed to set up vocabulary: %v", err)
	}

	// Train Word2Vec model first
	word2vecTrainingDataPath := "trainingdata/bartdata/explanation_data.json"
	word2vecModelSavePath := "gob_models/word2vec_model.gob"
	word2vecVectorSize := *dimModel
	word2vecEpochs := 50
	word2vecWindow := 5
	word2vecNegativeSamples := 5
	word2vecMinWordFrequency := 1
	word2vecUseCBOW := true

	word2vecModel, err := word2vec.LoadModel(word2vecModelSavePath)
	if err != nil {
		log.Printf("Word2Vec model not found, training a new one...")
		word2vecModel, err = word2vec.TrainWord2VecModel(
			word2vecTrainingDataPath,
			word2vecModelSavePath,
			word2vecVectorSize,
			word2vecEpochs,
			word2vecWindow,
			word2vecNegativeSamples,
			word2vecMinWordFrequency,
			word2vecUseCBOW,
		)
		if err != nil {
			log.Fatalf("Error training Word2Vec model: %v", err)
		}
	} else {
		log.Println("Loaded existing Word2Vec model.")
	}

	// Load word2vec model for BART and BERT
	word2vecModel, err = word2vec.LoadModel(word2vecModelSavePath)
	var pretrainedEmbeddings map[string][]float64
	if err != nil {
		log.Printf("Warning: Could not load word2vec model: %v. BART and BERT embeddings will be initialized randomly.", err)
		pretrainedEmbeddings = nil // Or an empty map
	} else {
		log.Println("Word2vec model loaded successfully.")
		pretrainedEmbeddings = word2vec.ConvertToMap(word2vecModel.WordVectors, word2vecModel.Vocabulary)
	}

	var bartModel *bartsimple.SimplifiedBARTModel // Declare here

	bartModel, err = bartsimple.LoadSimplifiedBARTModelFromGOB(bartModelPath)
	if err != nil {
		log.Printf("BART model not found, creating a new one...")
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
		bartModel, err = bartsimple.NewSimplifiedBARTModel(tokenizer, vocabulary, *dimModel, *numHeads, *maxSeqLength, pretrainedEmbeddings)
		if err != nil {
			log.Fatalf("Failed to create a new BART model: %v", err)
		}
	}

	if _, err := os.Stat(bartModelPath); os.IsNotExist(err) || *trainBart {
		log.Printf("BART model not found or training is forced, training a new one...")
		bartData, err := bartsimple.LoadBARTTrainingData(*bartDataPath)
		if err != nil {
			log.Fatalf("Failed to load BART training data: %v", err)
		}
		bartsimple.TrainBARTModel(bartModel, bartData, *epochs, *learningRate, *batchSize)
		if err := bartsimple.SaveSimplifiedBARTModelToGOB(bartModel, bartModelPath); err != nil {
			log.Fatalf("Failed to save BART model: %v", err)
		}
		if *trainBart {
			return // Exit after training if the flag is set
		}
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

	bertTrainingData, err := bert.LoadTrainingData(*bertDataPath)
	if err != nil {
		log.Fatalf("Error loading BERT training data: %v", err)
	}

	var bertModel *bert.BertModel
	
		bertModel, err = bert.Train(bertConfig, bertTrainingData, *epochs, *learningRate, vocabulary, pretrainedEmbeddings)
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