# nlptagger

When you need a program to understand context of commands.


![GitHub repo file count](https://img.shields.io/github/directory-file-count/golangast/nlptagger) 
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/golangast/nlptagger)
![GitHub repo size](https://img.shields.io/github/repo-size/golangast/nlptagger)
![GitHub](https://img.shields.io/github/license/golangast/nlptagger)
![GitHub commit activity](https://img.shields.io/github/commit-activity/w/golangast/nlptagger)
![Go 100%](https://img.shields.io/badge/Go-100%25-blue)
![status beta](https://img.shields.io/badge/Status-Beta-red)
<img src="https://img.shields.io/github/license/golangast/nlptagger.svg"><img src="https://img.shields.io/github/stars/golangast/nlptagger.svg">[![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](http://endrulats.com)[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)[![GitHub go.mod Go version of a Go module](https://img.shields.io/github/go-mod/go-version/gomods/athens.svg)](https://github.com/golangast/nlptagger)[![GoDoc reference example](https://img.shields.io/badge/godoc-reference-blue.svg)](https://pkg.go.dev/github.com/golangast/nlptagger/nlptaggerer)[![GoReportCard example](https://goreportcard.com/badge/github.com/golangast/nlptagger)](https://goreportcard.com/report/github.com/golangast/nlptagger)[![saythanks](https://img.shields.io/badge/say-thanks-ff69b4.svg)](https://saythanks.io/to/golangast)



  - [nlptagger](#nlptagger)
  - [General info](#general-info)
  - [Why build this?](#why-build-this)
  - [What does it do?](#what-does-it-do)
  - [Technologies](#technologies)
  - [Requirements](#requirements)
  - [Repository overview](#repository-overview)
  - [Overview of the code.](#Overview-of-the-code.)
  - [Things to remember](#things-to-remember)
  - [Reference Commands](#reference-commands)
  - [Special thanks](#special-thanks)
  - [Why Go?](#why-go)
  - [Just added](#just-added)


## General info
*This project is used for tagging cli commands.  It is not a LLM or trying to be.  I am using it to generate go code but I made this completely separate so others can enjoy it.
*I will keep working on it and hopefully improving the phrase tagging and hopefully adding neural networks in the future.


## Why build this?
* Go never changes
* It is nice to not have terminal drop downs


## What does it do?
* It tags words for commands.
*I made an overview video on this project.
[video](https://www.youtube.com/watch?v=QuY5tZj0CXI)



## Technologies
*Just Go.


## Requirements
* go 1.23 for gonew

## How to run as is?
```go
//import tagger.Tagging
t := tagger.Tagging("please create handler named jim")

//which returns a tag struct that you can pull apart.
/*
type Tag struct {
	PosTag    []string
	NerTag    []string
	PhraseTag []string
	Tokens    []string
	Features  []Features
	Epoch     int
	Cost      gorgonia.Value
	IsName    bool
	Token     string
	Tags      []Tag
}
*/
	for i := range t.Tokens {
		fmt.Printf("Tokens: %s\t\t\t Ner: %s\t\t Pos: %s\t\t PhraseTag: %s\t\t  \n", t.Tokens[i], t.NerTag[i], t.PosTag[i], t.PhraseTag[i])
	}

```

*- clone it
```bash
git clone https://github.com/golangast/nlptagger
```
* - or
* - install gonew to pull down project quickly
```bash
go install golang.org/x/tools/cmd/gonew@latest
```
* - run gonew
```bash
gonew github.com/golangast/nlptagger example.com/nlptagger
```

* - cd into nlptagger
=======
* - cd into switchterm
```bash
cd nlptagger
```

* - run the project
```bash
go run main.go
```

## Repository overview
```bash
├── nertagger #ner tagging which is kinda more general then pos tagging
├── phrasetagger  #phrase tagging
├── postagger   #pos tagging
├── tag #data structure for tag
├── tagger.go #where it is processed

```

## Overview of the code.
*All this does is tag sentences and it is not a LLM but only is an attempt at tagging commands.
```
## Things to remember
* it is not a LLM or trying to be
* it is only for cli commands

 ```



## Just added
*the project



## Special thanks
* [Go Team because they are gods](https://github.com/golang/go/graphs/contributors)


## Why Go?
* The language is done since 1.0.https://youtu.be/rFejpH_tAHM there are little features that get added after 10 years but whatever you learn now will forever be useful.
* It also has a compatibility promise https://go.dev/doc/go1compat
* It was also built by great people. https://hackernoon.com/why-go-ef8850dc5f3c
* 14th used language https://insights.stackoverflow.com/survey/2021
* Highest starred language https://github.com/golang/go
* It is also number 1 language to go to and not from https://www.jetbrains.com/lp/devecosystem-2021/#Do-you-plan-to-adopt--migrate-to-other-languages-in-the-next--months-If-so-to-which-ones
* Go is growing in all measures https://madnight.github.io/githut/#/stars/2023/3
* Jobs are almost doubling every year. https://stacktrends.dev/technologies/programming-languages/golang/
* Companies that use go. https://go.dev/wiki/GoUsers
* Why I picked Go https://youtu.be/fD005g07cU4
