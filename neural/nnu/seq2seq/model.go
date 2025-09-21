package seq2seq

import (
	"encoding/gob"
	"fmt"
	"log"
	"os"

	"nlptagger/neural/nn"
	"nlptagger/neural/tensor"
	"nlptagger/neural/tokenizer"
	"nlptagger/neural/nnu/vocab"
)

// SerializableSeq2Seq is a struct for saving and loading the model.
type SerializableSeq2Seq struct {
	Encoder     *Encoder
	Decoder     *Decoder
	OutputVocab *vocab.Vocabulary
	HiddenDim   int
}

// Encoder represents the encoder part of the Seq2Seq model.
type Encoder struct {
	Embedding *nn.Embedding
	LSTM      *nn.LSTM
	// Add other layers as needed
}

// NewEncoder creates a new Encoder.
func NewEncoder(inputVocabSize, embeddingDim, hiddenDim int) *Encoder {
	return &Encoder{
		Embedding: nn.NewEmbedding(inputVocabSize, embeddingDim),
		LSTM:      nn.NewLSTM(embeddingDim, hiddenDim),
	}
}

// Forward performs a forward pass through the encoder.
func (e *Encoder) Forward(inputIDs *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, error) {
	// inputIDs: [batch_size, sequence_length]
	embedded, err := e.Embedding.Forward(inputIDs)
	if err != nil {
		return nil, nil, fmt.Errorf("encoder embedding forward failed: %w", err)
	}

	// embedded: [batch_size, sequence_length, embedding_dim]
	// LSTM.Forward expects input [batch_size, sequence_length, input_size]
	_, hidden, cell, err := e.LSTM.Forward(embedded, nil, nil) // Initial hidden/cell states are nil
	if err != nil {
		return nil, nil, fmt.Errorf("encoder LSTM forward failed: %w", err)
	}
	// hidden, cell: [batch_size, hidden_dim] (last hidden/cell state)
	return hidden, cell, nil
}

// Parameters returns all learnable parameters of the Encoder.
func (e *Encoder) Parameters() []*tensor.Tensor { // Changed here
	params := []*tensor.Tensor{} // Changed here
	params = append(params, e.Embedding.Parameters()...)
	params = append(params, e.LSTM.Parameters()...)
	return params
}

// Decoder represents the decoder part of the Seq2Seq model.
type Decoder struct {
	Embedding *nn.Embedding
	LSTM      *nn.LSTM
	Output    *nn.Linear
	// Add other layers as needed, e.g., attention
}

// NewDecoder creates a new Decoder.
func NewDecoder(outputVocabSize, embeddingDim, hiddenDim int) *Decoder {
	outputLayer, err := nn.NewLinear(hiddenDim, outputVocabSize)
	if err != nil {
		log.Fatalf("Failed to create output linear layer for decoder: %v", err)
	}
	return &Decoder{
		Embedding: nn.NewEmbedding(outputVocabSize, embeddingDim),
		LSTM:      nn.NewLSTM(embeddingDim, hiddenDim), // Input to LSTM will be embedded token + context
		Output:    outputLayer,
	}
}

// Forward performs a forward pass through the decoder.
// It takes the previous output token, and the encoder's final hidden and cell states.
func (d *Decoder) Forward(inputTokenID *tensor.Tensor, hidden, cell *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor, error) {
	// inputTokenID: [batch_size, 1] (single token ID)
	embedded, err := d.Embedding.Forward(inputTokenID)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("decoder embedding forward failed: %w", err)
	}
	// embedded: [batch_size, 1, embedding_dim]

	// LSTM expects input [batch_size, sequence_length, input_size]
	// For a single token, sequence_length is 1
	output, hidden, cell, err := d.LSTM.Forward(embedded, hidden, cell)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("decoder LSTM forward failed: %w", err)
	}
	// output: [batch_size, 1, hidden_dim]

	// Apply linear layer to the output of the LSTM
	// Reshape output to [batch_size, hidden_dim] for linear layer
	outputReshaped, err := output.Reshape([]int{output.Shape[0], output.Shape[2]})
	if err != nil {
		return nil, nil, nil, fmt.Errorf("decoder output reshape failed: %w", err)
	}
	prediction, err := d.Output.Forward(outputReshaped)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("decoder output linear layer failed: %w", err)
	}
	// prediction: [batch_size, output_vocab_size] (logits for next token)

	return prediction, hidden, cell, nil
}

// Parameters returns all learnable parameters of the Decoder.
func (d *Decoder) Parameters() []*tensor.Tensor { // Changed here
	params := []*tensor.Tensor{} // Changed here
	params = append(params, d.Embedding.Parameters()...)
	params = append(params, d.LSTM.Parameters()...)
	params = append(params, d.Output.Parameters()...)
	return params
}

// Seq2Seq represents the complete Encoder-Decoder model.
type Seq2Seq struct {
	Encoder *Encoder
	Decoder *Decoder
	Tokenizer *tokenizer.Tokenizer // For encoding/decoding text
	OutputVocab *vocab.Vocabulary // Vocabulary for the output descriptions
	HiddenDim int
}

// NewSeq2Seq creates a new Seq2Seq model.
func NewSeq2Seq(inputVocabSize, outputVocabSize, embeddingDim, hiddenDim int, tok *tokenizer.Tokenizer, outVocab *vocab.Vocabulary) *Seq2Seq {
	return &Seq2Seq{
		Encoder: NewEncoder(inputVocabSize, embeddingDim, hiddenDim),
		Decoder: NewDecoder(outputVocabSize, embeddingDim, hiddenDim),
		Tokenizer: tok,
		OutputVocab: outVocab,
		HiddenDim: hiddenDim,
	}
}

// Parameters returns all learnable parameters of the Seq2Seq model.
func (m *Seq2Seq) Parameters() []*tensor.Tensor { // Changed here
	params := []*tensor.Tensor{} // Changed here
	params = append(params, m.Encoder.Parameters()...)
	params = append(params, m.Decoder.Parameters()...)
	return params
}

// Forward performs a forward pass through the Seq2Seq model for training.
// It takes input sequence IDs and target output sequence IDs.
func (m *Seq2Seq) Forward(inputIDs, targetIDs *tensor.Tensor) (*tensor.Tensor, error) {
	// inputIDs: [batch_size, input_seq_len]
	// targetIDs: [batch_size, target_seq_len] (includes <SOS> and <EOS>)

	batchSize := inputIDs.Shape[0]
	targetSeqLen := targetIDs.Shape[1]
	outputVocabSize := m.OutputVocab.Size()

	// Encoder forward pass
	encoderHidden, encoderCell, err := m.Encoder.Forward(inputIDs)
	if err != nil {
		return nil, fmt.Errorf("seq2seq encoder forward failed: %w", err)
	}

	// Prepare tensor to store decoder outputs (logits for each token in the target sequence)
	decoderOutputs := tensor.NewTensor([]int{batchSize, targetSeqLen, outputVocabSize}, nil, true)

	// Initialize decoder hidden and cell states with encoder's final states
	decoderHidden := encoderHidden
	decoderCell := encoderCell

	// Teacher forcing: feed target token as next input
	for t := 0; t < targetSeqLen; t++ {
		// Get current input token for decoder (targetIDs[:, t])
		decoderInputData := make([]float64, batchSize)
		for b := 0; b < batchSize; b++ {
			// Assuming targetIDs is [batch_size, seq_len]
			decoderInputData[b] = targetIDs.Data[b*targetSeqLen + t]
		}
		decoderInput := tensor.NewTensor([]int{batchSize, 1}, decoderInputData, true)

		// Decoder forward pass
		prediction, hidden, cell, err := m.Decoder.Forward(decoderInput, decoderHidden, decoderCell)
		if err != nil {
			return nil, fmt.Errorf("seq2seq decoder forward failed at step %d: %w", t, err)
		}

		// Store prediction
		// prediction is [batch_size, output_vocab_size]
		// decoderOutputs is [batch_size, target_seq_len, output_vocab_size]
		for b := 0; b < batchSize; b++ {
			for v_idx := 0; v_idx < outputVocabSize; v_idx++ {
				decoderOutputs.Set([]int{b, t, v_idx}, prediction.Get([]int{b, v_idx}))
			}
		}

		decoderHidden = hidden
		decoderCell = cell
	}

	return decoderOutputs, nil
}

// Predict generates a description given an input query.
func (m *Seq2Seq) Predict(query string, maxLen int) (string, error) {
	// Encode the input query
	inputTokenIDs, err := m.Tokenizer.Encode(query)
	if err != nil {
		return "", fmt.Errorf("failed to tokenize query: %w", err)
	}

	// Convert token IDs to tensor
	inputTensorData := make([]float64, len(inputTokenIDs))
	for i, id := range inputTokenIDs {
		inputTensorData[i] = float64(id)
	}
	inputTensor := tensor.NewTensor([]int{1, len(inputTokenIDs)}, inputTensorData, true)

	// Encoder forward pass
	encoderHidden, encoderCell, err := m.Encoder.Forward(inputTensor)
	if err != nil {
		return "", fmt.Errorf("prediction encoder forward failed: %w", err)
	}

	decoderHidden := encoderHidden
	decoderCell := encoderCell

	// Start with <SOS> token
	outputTokens := []int{}
	currentInputTokenID := float64(m.OutputVocab.BosID)

	for t := 0; t < maxLen; t++ {
		decoderInput := tensor.NewTensor([]int{1, 1}, []float64{currentInputTokenID}, true)

		prediction, hidden, cell, err := m.Decoder.Forward(decoderInput, decoderHidden, decoderCell)
		if err != nil {
			return "", fmt.Errorf("prediction decoder forward failed at step %d: %w", t, err)
		}

		// Get the token with the highest probability (greedy decoding)
		predictedTokenID := 0
		maxProb := prediction.Data[0]
		for i := 1; i < prediction.Shape[1]; i++ {
			if prediction.Data[i] > maxProb {
				maxProb = prediction.Data[i]
				predictedTokenID = i
			}
		}

		outputTokens = append(outputTokens, predictedTokenID)
		if predictedTokenID == m.OutputVocab.EosID {
			break
		}

		// Use predicted token as next input
		currentInputTokenID = float64(predictedTokenID)
		decoderHidden = hidden
		decoderCell = cell
	}

	// Decode output tokens to string
	decodedDescription := m.OutputVocab.Decode(outputTokens)

	return decodedDescription, nil
}

// Save saves the Seq2Seq model to a file.
func (m *Seq2Seq) Save(filePath string) error {
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("failed to create file for saving model: %w", err)
	}
	defer file.Close()

	encoder := gob.NewEncoder(file)

	// Create a serializable version of the model
	serializableModel := &SerializableSeq2Seq{
		Encoder:     m.Encoder,
		Decoder:     m.Decoder,
		OutputVocab: m.OutputVocab,
		HiddenDim:   m.HiddenDim,
	}

	if err := encoder.Encode(serializableModel); err != nil {
		return fmt.Errorf("failed to encode model: %w", err)
	}

	log.Printf("Seq2Seq model saved to %s", filePath)
	return nil
}

// Load loads the Seq2Seq model from a file.
func Load(filePath string, tok *tokenizer.Tokenizer) (*Seq2Seq, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open file for loading model: %w", err)
	}
	defer file.Close()

	decoder := gob.NewDecoder(file)

	var serializableModel SerializableSeq2Seq
	if err := decoder.Decode(&serializableModel); err != nil {
		return nil, fmt.Errorf("failed to decode model: %w", err)
	}

	// Create a new Seq2Seq model from the loaded data
	model := &Seq2Seq{
		Encoder:     serializableModel.Encoder,
		Decoder:     serializableModel.Decoder,
		Tokenizer:   tok, // Tokenizer is not saved, it's passed in
		OutputVocab: serializableModel.OutputVocab,
		HiddenDim:   serializableModel.HiddenDim,
	}

	log.Printf("Seq2Seq model loaded from %s", filePath)
	return model, nil
}