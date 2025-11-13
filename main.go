package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
)

func main() {
	trainWord2Vec := flag.Bool("train-word2vec", false, "Train the Word2Vec model")
	trainMoE := flag.Bool("train-moe", false, "Train the MoE model")
	trainIntentClassifier := flag.Bool("train-intent-classifier", false, "Train the intent classification model")
	moeInferenceQuery := flag.String("moe_inference", "", "Run MoE inference with the given query")

	flag.Parse()

	if *trainWord2Vec {
		runModule("cmd/train_word2vec")
	} else if *trainMoE {
		runModule("cmd/train_moe")
	} else if *trainIntentClassifier {
		runModule("cmd/train_intent_classifier")
	} else if *moeInferenceQuery != "" {
		runMoeInference(*moeInferenceQuery)
	} else {
		log.Println("No action specified. Use -train-word2vec, -train-moe, -train-intent-classifier, -moe_inference <query>, or -jill.")
	}
}

func runMoeInference(query string) {
	cmd := exec.Command("go", "run", "./cmd/moe_inference", "-query", query)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err := cmd.Run()
	if err != nil {
		log.Fatalf("Failed to run MoE inference: %v", err)
	}
}

func runModule(path string) {
	cmd := exec.Command("go", "run", "./"+path)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err := cmd.Run()
	if err != nil {
		fmt.Println(err, cmd.Stderr, cmd.Stdout)
		log.Fatalf("Failed to run module %s: %v", path, err)
	}
}
