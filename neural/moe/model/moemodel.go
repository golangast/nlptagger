package model

import (
	"encoding/gob"
	"fmt"
	"log"
	"os"

	"nlptagger/neural/moe"
	"nlptagger/neural/nn"
	"nlptagger/neural/nnu/bert"
	"nlptagger/neural/tensor"
)

func init() {
	gob.Register(bert.BertConfig{})
}

// MoEClassificationModel is a wrapper around the MoELayer to make it a trainable model.
type MoEClassificationModel struct {
	Embedding        *nn.Embedding
	BertModel        *bert.BertModel
	MoELayer         *moe.MoELayer
	ParentClassifier *nn.Linear
	ChildClassifier  *nn.Linear
	BertConfig       bert.BertConfig
	pooledOutput     *tensor.Tensor // Add this field
}

func NewMoEClassificationModel(vocabSize, embeddingDim, numParentClasses, numChildClasses, numExperts, k, maxSeqLength int) (*MoEClassificationModel, error) {
	embedding := nn.NewEmbedding(vocabSize, embeddingDim)

	bertConfig := bert.BertConfig{
		VocabSize:             vocabSize,
		HiddenSize:            embeddingDim,
		MaxPositionEmbeddings: maxSeqLength,
		NumHiddenLayers:       4,
		NumAttentionHeads:     4,
		IntermediateSize:      embeddingDim * 4,
		TypeVocabSize:         2,
		InitializerRange:      0.02,
	}
	bertModel := bert.NewBertModel(bertConfig, nil)

	expertBuilder := func(expertIdx int) (moe.Expert, error) {
		return moe.NewFeedForwardExpert(embeddingDim, embeddingDim*2, embeddingDim)
	}

	moeLayer, err := moe.NewMoELayer(embeddingDim, numExperts, k, expertBuilder)
	if err != nil {
		return nil, err
	}

	parentClassifier, err := nn.NewLinear(embeddingDim, numParentClasses)
	if err != nil {
		return nil, err
	}
	childClassifier, err := nn.NewLinear(embeddingDim, numChildClasses)
	if err != nil {
		return nil, err
	}

	return &MoEClassificationModel{
		Embedding:        embedding,
		BertModel:        bertModel,
		MoELayer:         moeLayer,
		ParentClassifier: parentClassifier,
		ChildClassifier:  childClassifier,
		BertConfig:       bertConfig,
	}, nil
}

func (m *MoEClassificationModel) SetMode(training bool) {
	m.MoELayer.SetMode(training)
}

func (m *MoEClassificationModel) Forward(inputs ...*tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, error) {
	inputIDs := inputs[0]

	batchSize, seqLength := inputIDs.Shape[0], inputIDs.Shape[1]

	// Create dummy tokenTypeIDs, posTagIDs, nerTagIDs for BertModel

	tokenTypeIDs := tensor.NewTensor([]int{batchSize, seqLength}, make([]float64, batchSize*seqLength), false) // All zeros for now
	posTagIDs := tensor.NewTensor([]int{batchSize, seqLength}, make([]float64, batchSize*seqLength), false)    // All zeros for now
	nerTagIDs := tensor.NewTensor([]int{batchSize, seqLength}, make([]float64, batchSize*seqLength), false)    // All zeros for now

	// Pass inputIDs through BertModel to get contextualized embeddings
	bertOutput, err := m.BertModel.Forward(inputIDs, tokenTypeIDs, posTagIDs, nerTagIDs)
	if err != nil {
		return nil, nil, fmt.Errorf("bert model forward failed: %w", err)
	}

	moeOutput, err := m.MoELayer.Forward(bertOutput)
	if err != nil {
		return nil, nil, fmt.Errorf("moe forward failed: %w", err)
	}

	// Extract the output of the first token (CLS token) for classification
	clsTokenOutput, err := moeOutput.Slice(1, 0, 1) // Slice along the sequence length dimension
	if err != nil {
		return nil, nil, fmt.Errorf("failed to slice CLS token output: %w", err)
	}

	clsTokenOutputReshaped, err := clsTokenOutput.Reshape([]int{batchSize, m.BertConfig.HiddenSize})
	if err != nil {
		return nil, nil, fmt.Errorf("failed to reshape CLS token output: %w", err)
	}

	parentLogits, err := m.ParentClassifier.Forward(clsTokenOutputReshaped)
	if err != nil {
		return nil, nil, fmt.Errorf("parent classifier forward failed: %w", err)
	}

	childLogits, err := m.ChildClassifier.Forward(clsTokenOutputReshaped)
	if err != nil {
		return nil, nil, fmt.Errorf("child classifier forward failed: %w", err)
	}

	return parentLogits, childLogits, nil
}

func (m *MoEClassificationModel) Backward(parentGrad, childGrad *tensor.Tensor) error {
	err := m.ParentClassifier.Backward(parentGrad)
	if err != nil {
		return fmt.Errorf("parent classifier backward failed: %w", err)
	}

	err = m.ChildClassifier.Backward(childGrad)
	if err != nil {
		return fmt.Errorf("child classifier backward failed: %w", err)
	}

	// Since ParentClassifier and ChildClassifier share the same input tensor,
	// and Linear.Backward accumulates gradients, the gradient is already summed
	// in the input tensor's Grad field.
	clsTokenGrad := m.ParentClassifier.Input().Grad

	moeLayerGrad := tensor.NewTensor(m.MoELayer.GetOutputShape(), make([]float64, m.MoELayer.GetOutputShape()[0]*m.MoELayer.GetOutputShape()[1]*m.MoELayer.GetOutputShape()[2]), false)

	batchSize := clsTokenGrad.Shape[0]
	hiddenSize := clsTokenGrad.Shape[1]
	seqLength := moeLayerGrad.Shape[1]

	for i := 0; i < batchSize; i++ {
		copy(moeLayerGrad.Data[i*seqLength*hiddenSize:i*seqLength*hiddenSize+hiddenSize], clsTokenGrad.Data[i*hiddenSize:(i+1)*hiddenSize])
	}

	err = m.MoELayer.Backward(moeLayerGrad)
	if err != nil {
		return fmt.Errorf("moe layer backward failed: %w", err)
	}

	bertGrad := m.MoELayer.Inputs()[0].Grad

	err = m.BertModel.Backward(bertGrad)
	if err != nil {
		return fmt.Errorf("bert model backward failed: %w", err)
	}

	return nil
}

func (m *MoEClassificationModel) Parameters() []*tensor.Tensor {
	var params []*tensor.Tensor
	// Only include BertModel parameters if it's not nil
	if m.BertModel != nil {
		params = append(params, m.BertModel.Parameters()...)
	}
	params = append(params, m.MoELayer.Parameters()...)
	params = append(params, m.ParentClassifier.Parameters()...)
	params = append(params, m.ChildClassifier.Parameters()...)
	return params
}

// Description prints a summary of the MoEClassificationModel's architecture.
func (m *MoEClassificationModel) Description() {
}

// SaveMoEClassificationModelToGOB saves the MoEClassificationModel to a file in Gob format.
func SaveMoEClassificationModelToGOB(model *MoEClassificationModel, filePath string) error {
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to create file for saving MoE model: %w", err)
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)
	if err := encoder.Encode(model); err != nil {
		return fmt.Errorf("failed to encode MoE model to Gob: %w", err)
	}

	return nil
}

// LoadMoEClassificationModelFromGOB loads a MoEClassificationModel from a file in Gob format.
func LoadMoEClassificationModelFromGOB(filePath string, vocabSize, numParentClasses, numChildClasses, maxSeqLength int) (*MoEClassificationModel, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("error opening MoE model gob file: %w", err)
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	var loadedModel MoEClassificationModel
	err = decoder.Decode(&loadedModel)
	if err != nil {
		return nil, fmt.Errorf("error decoding MoE model from gob: %w", err)
	}

	// --- Start of explicit BertModel re-creation and weight copying ---
	// Update BertConfig's VocabSize with the provided vocabSize
	loadedModel.BertConfig.VocabSize = vocabSize
	loadedModel.BertConfig.MaxPositionEmbeddings = maxSeqLength // Add this line
	// Create a new BertModel using the loaded BertConfig
	newBertModel := bert.NewBertModel(loadedModel.BertConfig, nil)

	// Copy WordEmbeddings weights
	if loadedModel.BertModel != nil && loadedModel.BertModel.BertEmbeddings != nil && loadedModel.BertModel.BertEmbeddings.WordEmbeddings != nil && loadedModel.BertModel.BertEmbeddings.WordEmbeddings.Weight != nil {
		// Determine the minimum length to copy to avoid out-of-bounds access
		minLen := len(loadedModel.BertModel.BertEmbeddings.WordEmbeddings.Weight.Data)
		if len(newBertModel.BertEmbeddings.WordEmbeddings.Weight.Data) < minLen {
			minLen = len(newBertModel.BertEmbeddings.WordEmbeddings.Weight.Data)
		}
		copy(newBertModel.BertEmbeddings.WordEmbeddings.Weight.Data[:minLen], loadedModel.BertModel.BertEmbeddings.WordEmbeddings.Weight.Data[:minLen])
		if len(newBertModel.BertEmbeddings.WordEmbeddings.Weight.Data) > minLen {
			log.Printf("WARNING: New WordEmbeddings size (%d) is larger than loaded size (%d). New embeddings will be randomly initialized for new tokens.", len(newBertModel.BertEmbeddings.WordEmbeddings.Weight.Data), minLen)
		}
	} else {
		log.Printf("WARNING: WordEmbeddings not found in loaded model, using newly initialized weights.")
	}

	// Copy PositionEmbeddings weights
	if loadedModel.BertModel != nil && loadedModel.BertModel.BertEmbeddings != nil && loadedModel.BertModel.BertEmbeddings.PositionEmbeddings != nil && loadedModel.BertModel.BertEmbeddings.PositionEmbeddings.Weight != nil {
		minLen := len(loadedModel.BertModel.BertEmbeddings.PositionEmbeddings.Weight.Data)
		if len(newBertModel.BertEmbeddings.PositionEmbeddings.Weight.Data) < minLen {
			minLen = len(newBertModel.BertEmbeddings.PositionEmbeddings.Weight.Data)
		}
		copy(newBertModel.BertEmbeddings.PositionEmbeddings.Weight.Data[:minLen], loadedModel.BertModel.BertEmbeddings.PositionEmbeddings.Weight.Data[:minLen])
	} else {
		log.Printf("WARNING: PositionEmbeddings not found in loaded model, using newly initialized weights.")
	}

	// Copy TokenTypeEmbeddings weights
	if loadedModel.BertModel != nil && loadedModel.BertModel.BertEmbeddings != nil && loadedModel.BertModel.BertEmbeddings.TokenTypeEmbeddings != nil && loadedModel.BertModel.BertEmbeddings.TokenTypeEmbeddings.Weight != nil {
		minLen := len(loadedModel.BertModel.BertEmbeddings.TokenTypeEmbeddings.Weight.Data)
		if len(newBertModel.BertEmbeddings.TokenTypeEmbeddings.Weight.Data) < minLen {
			minLen = len(newBertModel.BertEmbeddings.TokenTypeEmbeddings.Weight.Data)
		}
		copy(newBertModel.BertEmbeddings.TokenTypeEmbeddings.Weight.Data[:minLen], loadedModel.BertModel.BertEmbeddings.TokenTypeEmbeddings.Weight.Data[:minLen])
		if len(newBertModel.BertEmbeddings.TokenTypeEmbeddings.Weight.Data) > minLen {
			log.Printf("WARNING: New TokenTypeEmbeddings size (%d) is larger than loaded size (%d). New embeddings will be randomly initialized for new token types.", len(newBertModel.BertEmbeddings.TokenTypeEmbeddings.Weight.Data), minLen)
		}
	} else {
		log.Printf("WARNING: TokenTypeEmbeddings not found in loaded model, using newly initialized weights.")
	}

	// Assign the newly created and weight-copied BertModel
	loadedModel.BertModel = newBertModel
	// --- End of explicit BertModel re-creation and weight copying ---

	// Re-initialize the MoELayer (this part seems necessary due to interface serialization issues)
	if loadedModel.MoELayer == nil || loadedModel.MoELayer.GatingNetwork == nil || len(loadedModel.MoELayer.Experts) == 0 {
		// Use default values for numExperts and k if not available in the loaded model
		// These values should ideally be part of the saved model configuration.
		// For now, we'll use hardcoded defaults based on NewMoEClassificationModel.
		numExperts := 8                                   // Default value
		k := 2                                            // Default value
		embeddingDim := loadedModel.BertConfig.HiddenSize // Use BertConfig.HiddenSize for embeddingDim

		expertBuilder := func(expertIdx int) (moe.Expert, error) {
			return moe.NewFeedForwardExpert(embeddingDim, embeddingDim*2, embeddingDim)
		}

		moeLayer, err := moe.NewMoELayer(embeddingDim, numExperts, k, expertBuilder)
		if err != nil {
			return nil, fmt.Errorf("failed to re-initialize MoE layer: %w", err)
		}
		loadedModel.MoELayer = moeLayer
	}

	// Re-initialize the Activation function in the Pooler after loading,
	// as gob cannot serialize/deserialize functions.
	if loadedModel.BertModel != nil && loadedModel.BertModel.Pooler != nil {
		loadedModel.BertModel.Pooler.Activation = func(x *tensor.Tensor) *tensor.Tensor {
			t, _ := x.Tanh()
			return t
		}
	}

	// Re-initialize ParentClassifier and ChildClassifier after loading,
	// as gob might not correctly deserialize their internal *tensor.Tensor fields.
	if loadedModel.ParentClassifier == nil {
		parentClassifier, err := nn.NewLinear(loadedModel.BertConfig.HiddenSize, numParentClasses)
		if err != nil {
			return nil, fmt.Errorf("failed to re-initialize parent classifier: %w", err)
		}
		loadedModel.ParentClassifier = parentClassifier
	} else if loadedModel.ParentClassifier != nil {
		// Store deserialized weights and biases
		parentWeightsData := loadedModel.ParentClassifier.Weights.Data
		parentBiasesData := loadedModel.ParentClassifier.Biases.Data
		parentInputDim := loadedModel.ParentClassifier.Weights.Shape[0]

		// Create new Linear layer
		newParentClassifier, err := nn.NewLinear(parentInputDim, numParentClasses) // Use numParentClasses
		if err != nil {
			return nil, fmt.Errorf("failed to re-initialize parent classifier: %%w", err)
		}

		// Copy deserialized data to new Linear layer
		var minLen int
		minLen = len(parentWeightsData)
		if len(newParentClassifier.Weights.Data) < minLen {
			minLen = len(newParentClassifier.Weights.Data)
		}
		copy(newParentClassifier.Weights.Data[:minLen], parentWeightsData[:minLen])

		minLen = len(parentBiasesData)
		if len(newParentClassifier.Biases.Data) < minLen {
			minLen = len(newParentClassifier.Biases.Data)
		}
		copy(newParentClassifier.Biases.Data[:minLen], parentBiasesData[:minLen])

		loadedModel.ParentClassifier = newParentClassifier
	}

	if loadedModel.ChildClassifier == nil {
		childClassifier, err := nn.NewLinear(loadedModel.BertConfig.HiddenSize, numChildClasses)
		if err != nil {
			return nil, fmt.Errorf("failed to re-initialize child classifier: %%w", err)
		}
		loadedModel.ChildClassifier = childClassifier
	} else if loadedModel.ChildClassifier != nil {
		// Store deserialized weights and biases
		childWeightsData := loadedModel.ChildClassifier.Weights.Data
		childBiasesData := loadedModel.ChildClassifier.Biases.Data
		childInputDim := loadedModel.ChildClassifier.Weights.Shape[0]

		// Create new Linear layer
		newChildClassifier, err := nn.NewLinear(childInputDim, numChildClasses) // Use numChildClasses
		if err != nil {
			return nil, fmt.Errorf("failed to re-initialize child classifier: %%w", err)
		}

		// Copy deserialized data to new Linear layer
		var minLen int
		minLen = len(childWeightsData)
		if len(newChildClassifier.Weights.Data) < minLen {
			minLen = len(newChildClassifier.Weights.Data)
		}
		copy(newChildClassifier.Weights.Data[:minLen], childWeightsData[:minLen])

		minLen = len(childBiasesData)
		if len(newChildClassifier.Biases.Data) < minLen {
			minLen = len(newChildClassifier.Biases.Data)
		}
		copy(newChildClassifier.Biases.Data[:minLen], childBiasesData[:minLen])

		loadedModel.ChildClassifier = newChildClassifier
	}

	return &loadedModel, nil
}
