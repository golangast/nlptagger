package cli

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/golangast/nlptagger/neural/nnu/bartsimple"
	"github.com/golangast/nlptagger/neural/nnu/bert"
)

func RunInference(
	bartModel *bartsimple.SimplifiedBARTModel,
	bertModel *bert.BertModel,
	bertConfig bert.BertConfig,
	tokenizer *bartsimple.Tokenizer,
) {
	// Load the command map from the JSON file
	commands := GetCommands()
	intentMap := make(map[string]func([]string))

	// Map commands to their corresponding functions
	for _, cmd := range commands {
		intentMap[cmd] = func(args []string) {
			fmt.Printf("Executing command: %s with args: %v\n", cmd, args)
		}
	}

	for {
		command := InputScanDirections("Enter a command (or 'quit', exit, 'stop', 'q', 'close', 'help', 'list' to list available commands):")
		upperCmd := strings.ToUpper(command)
		switch upperCmd {
		case "LIST", "COMMANDS", "HELP":
			fmt.Println("Available commands are:")
			for i, cmd := range commands {
				fmt.Printf("- %d. %s\n", i, cmd)
			}
			continue
		case "":
			fmt.Println("No command entered. Please try again.")
			continue
		case "EXIT", "STOP", "Q", "QUIT", "CLOSE":
			fmt.Println("Exiting.")
			return
		default:
			// 1. Use the BERT model to predict the intent
			intent, bartReply, err := bertModel.BertProcessCommand(command, bertConfig, tokenizer, bartModel, bertModel.TrainingData)
			
			fmt.Printf("Intent: %s\n", intent)

			if err != nil {
				if strings.Contains(err.Error(), "user rejected the BART reply") {
					fmt.Println("Please provide the correct BART reply:")
					reader := bufio.NewReader(os.Stdin)
					correctReply, _ := reader.ReadString('\n')
					correctReply = strings.TrimSpace(correctReply)

					// Update BART training data
					bartDataPath := "trainingdata/bartdata/bartdata.json"
					bartTrainingData, _ := bartsimple.LoadBARTTrainingData(bartDataPath)
					newSentence := struct {
						Input  string `json:"input"`
						Output string `json:"output"`
					}{Input: command, Output: correctReply}
					bartTrainingData.Sentences = append(bartTrainingData.Sentences, newSentence)
					file, _ := json.MarshalIndent(bartTrainingData, "", " ")
					_ = os.WriteFile(bartDataPath, file, 0644)
					fmt.Println("BART training data updated. Retraining BART model...")
					fmt.Println("BART training data sources:")
					for _, s := range bartTrainingData.Sentences {
						fmt.Printf("  Input: %s, Output: %s\n", s.Input, s.Output)
					}
					bartsimple.TrainBARTModel(bartModel, bartTrainingData, 10, 0.001, 4)
					fmt.Println("BART model retrained.")
				} else {
					log.Printf("Error processing command with BERT model: %v", err)
				}
			}

			fmt.Println("Is this correct? (y/n)")
            reader := bufio.NewReader(os.Stdin)
            answer, _ := reader.ReadString('\n')
            if strings.TrimSpace(strings.ToLower(answer)) != "y" {
                fmt.Println("Please provide the correct intent:")
                correctIntent, _ := reader.ReadString('\n')
                correctIntent = strings.TrimSpace(correctIntent)

                // Update BERT training data
                bertDataPath := "trainingdata/bertdata/bert.json"
                bertTrainingData, _ := bert.LoadTrainingData(bertDataPath)
                newLabel := len(bertTrainingData)
                bertTrainingData = append(bertTrainingData, bert.TrainingExample{Text: command, Label: newLabel, Embedding: nil})
                file, _ := json.MarshalIndent(bertTrainingData, "", " ")
                _ = os.WriteFile(bertDataPath, file, 0644)
                fmt.Println("BERT training data updated.")
                intent = correctIntent
            }

            fmt.Printf("BART reply: %s\n", bartReply)
            fmt.Println("Is the BART reply correct? (y/n)")
            bartAnswer, _ := reader.ReadString('\n')
            if strings.TrimSpace(strings.ToLower(bartAnswer)) != "y" {
                fmt.Println("Please provide the correct BART reply:")
                correctBartReply, _ := reader.ReadString('\n')
                correctBartReply = strings.TrimSpace(correctBartReply)

                // Update BART training data
                bartDataPath := "trainingdata/bartdata/bartdata.json"
                bartTrainingData, _ := bartsimple.LoadBARTTrainingData(bartDataPath)
                newSentence := struct {
                    Input  string `json:"input"`
                    Output string `json:"output"`
                }{Input: command, Output: correctBartReply}
                bartTrainingData.Sentences = append(bartTrainingData.Sentences, newSentence)
                file, _ := json.MarshalIndent(bartTrainingData, "", " ")
                _ = os.WriteFile(bartDataPath, file, 0644)
                fmt.Println("BART training data updated. Retraining BART model...")
                bartsimple.TrainBARTModel(bartModel, bartTrainingData, 10, 0.001, 4)
                fmt.Println("BART model retrained.")
            }

			// 2. Execute the action based on the intent
			if action, ok := intentMap[intent]; ok {
				args := strings.Fields(command)
				action(args)
			} else {
				args := strings.Fields(command)
				if len(args) > 0 {
					commandName := strings.ToUpper(args[0])
					isNewCommand := true
					for _, cmd := range commands {
						if cmd == commandName {
							isNewCommand = false
							break
						}
					}

					if isNewCommand {
						commands = append(commands, commandName)
						// Update the commands.json file
						file, err := os.Create("neural/nnu/bert/commands/commands.json")
						if err != nil {
							log.Printf("Error updating commands file: %v", err)
						} else {
							encoder := json.NewEncoder(file)
							encoder.SetIndent("", "  ")
							if err := encoder.Encode(commands); err != nil {
								log.Printf("Error encoding updated commands: %v", err)
							}
							file.Close()
							fmt.Printf("Command '%s' added to commands list.\n", commandName)
						}
					}

				}
			}
		}
	}
}

func GetCommands() []string {

	// Load the command map from the JSON file
	commandsFile, err := os.Open("neural/nnu/bert/commands/commands.json")
	if err != nil {
		log.Fatalf("Error opening commands file: %v", err)
	}
	defer commandsFile.Close()
	var commands struct {
		Commands []string `json:"commands"`
	}

	if err := json.NewDecoder(commandsFile).Decode(&commands); err != nil {
		fmt.Println(commands)
		log.Fatalf("Error decoding commands file: %v", err)

	}

	// Create a map of intents to actions
	return commands.Commands

}

func InputScanDirections(directions string) string {
	fmt.Println(directions)

	scannerdesc := bufio.NewScanner(os.Stdin)
	if scannerdesc.Scan() {
		dir := scannerdesc.Text()
		return strings.TrimSpace(dir)
	}
	if err := scannerdesc.Err(); err != nil {
		log.Printf("Error reading input: %v", err)
	}
	return ""
}