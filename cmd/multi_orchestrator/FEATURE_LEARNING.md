# Feature: Learning from Files

## Problem

The user wanted the orchestrator to "learn" code from existing files instead of just generating boilerplate or empty files. Specifically, they wanted to be able to point the system to a `learning` folder containing code snippets and have the system ask which one to use.

## Solution

I implemented a **Knowledge Base** system.

### How It Works

1.  **`learning/` Directory**: The system now looks for a `learning` directory in the project root. You can put any code files (templates, snippets) in there.
2.  **Interactive Lookup**: When generating a file (e.g., `handler.go`), if the content is not fully specified by the template, the system searches the `learning` directory for files matching the name or component (e.g., "handler").
3.  **User Selection**: If matches are found, the system **prompts you**:
    ```
    📚 Found learned content for 'handler.go'. Use one of these?
       1. handler_template.go
       0. No, use default/generated
       > 
    ```
    Selecting a file will use its content for the new file.

## Files Modified

- `cmd/multi_orchestrator/knowledge.go`: New file implementing `KnowledgeBase` logic.
- `cmd/multi_orchestrator/main.go`:
  - Initialized `KnowledgeBase`.
  - Updated `generateFromSemantic` to use the KB and prompt the user.

## Try It

1.  Create a file in `learning/`:
    ```bash
    mkdir -p learning
    echo 'package main; func MyHandler() {}' > learning/my_handler.go
    ```
2.  Run the orchestrator:
    ```bash
    ./cmd/multi_orchestrator/multi_orchestrator
    > add handler code to handler.go
    ```
3.  The system should ask if you want to use `my_handler.go`.
