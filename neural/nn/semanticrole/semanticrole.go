package semanticrole

import (
	"encoding/gob"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/golangast/nlptagger/neural/nn/semanticrole/bilstm_model"
	"github.com/golangast/nlptagger/neural/nnu/word2vec"
)

// CreateRoleMap creates a role map from training data
func CreateRoleMap(data []SentenceRoleData) map[string]int {
	roleMap := make(map[string]int)
	nextID := 0
	for _, sentenceData := range data {
		for _, roleData := range sentenceData.Tokens {
			if _, ok := roleMap[roleData.Role]; !ok {
				roleMap[roleData.Role] = nextID
				nextID++
			}
		}
	}
	return roleMap
}

// RoleData represents the data structure for a token and its role
type RoleData struct {
	Token string `json:"token"`
	Role  string `json:"role"`
}

type SentenceRoleData struct {
	Sentence string     `json:"sentence"`
	Tokens   []RoleData `json:"tokens"`
}

// SemanticRoleModel represents the semantic role model
type SemanticRoleModel struct {
	EmbeddingModel *word2vec.SimpleWord2Vec
	BiLSTMModel    *bilstm_model.BiLSTMModel
	RoleMap        map[int]string
}

func (m *SemanticRoleModel) PredictRoles(tokens []string) ([]string, error) {
	tokenIDs := make([]int, len(tokens))
	for i, token := range tokens {
		index, ok := m.EmbeddingModel.Vocabulary[token]
		if !ok { // if the word is not in the vocabulary
			index = 0 // Use the index of "UNK"
		}

		tokenIDs[i] = index
	}

	// Check if there are any tokens to process

	if len(tokens) == 0 {
		return []string{}, nil // Return empty slice if no tokens

	}

	hiddenStates := m.BiLSTMModel.Forward(tokenIDs)
	predictedIndices := m.BiLSTMModel.Predict(hiddenStates)

	roleNames := make([]string, len(predictedIndices))
	for i, index := range predictedIndices {
		roleNames[i] = m.RoleMap[index]
	}
	return roleNames, nil
}

func NewSemanticRoleModel(embeddingModelPath string, bilstmModelPath string, roleMapPath string) (*SemanticRoleModel, error) {
	embeddingModel, err := word2vec.LoadModel(embeddingModelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load embedding model: %w", err)
	}

	bilstmModel, err := loadBiLSTMModel(bilstmModelPath) //loadBiLSTMModel(bilstmModelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load BiLSTM model: %w", err)
	}

	intToStringRoleMap, err := LoadRoleMap(roleMapPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load role map: %w", err)
	}

	return &SemanticRoleModel{
		EmbeddingModel: embeddingModel,
		BiLSTMModel:    bilstmModel,
		RoleMap:        intToStringRoleMap,
	}, nil
}

func loadBiLSTMModel(path string) (*bilstm_model.BiLSTMModel, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	model := &bilstm_model.BiLSTMModel{}
	err = decoder.Decode(model)
	if err != nil {
		return nil, err
	}
	return model, nil
}

func LoadRoleData(filePath string) ([]SentenceRoleData, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	data, err := ioutil.ReadAll(file)
	if err != nil {
		return nil, err
	}

	var sentenceRoleData []SentenceRoleData
	err = json.Unmarshal(data, &sentenceRoleData)
	if err != nil {
		return nil, err
	}
	return sentenceRoleData, nil
}

// LoadRoleMap loads the role map from a file
func LoadRoleMap(path string) (map[int]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)
	stringToIntRoleMap := make(map[string]int)
	err = decoder.Decode(&stringToIntRoleMap)
	if err != nil {
		return nil, err
	}
	intToStringRoleMap := make(map[int]string)
	for role, id := range stringToIntRoleMap {
		intToStringRoleMap[id] = role
	}
	return intToStringRoleMap, nil
}
