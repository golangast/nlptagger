# Enhancement: Interactive Intent Clarification

## Problem

When the semantic parser fails to understand an intent (e.g., "delete the server.go" returned `unknown` intent), the system would fall back to basic keyword matching, which often failed to perform the desired action (like deleting a specific file).

## Solution

I implemented an **interactive clarification system**. Now, if the intent is unknown:

1. The system **pauses and asks the user** what they want to do.
2. It presents a list of likely intents (create_handler, delete_file, etc.).
3. Based on the user's choice, it **updates the intent** and proceeds.
4. I also added specific handling for the `delete_file` intent, which was previously missing from the main loop.

## How It Works

### 1. Interactive Prompt
```
❓ I couldn't understand your intent.
   I found these entities: map[file:server.go]
   What would you like to do?
   1. create_handler
   2. create_database
   3. delete_file
   4. delete_project
   > 3
```

### 2. Intent Correction
The system updates the parsed goal:
```
   ✅ Intent set to: delete_file
```

### 3. Execution
The system executes the logic for the corrected intent:
```
Deleting file: server.go
File deleted successfully.
```

## Files Modified

- `cmd/multi_orchestrator/main.go`:
  - Modified `parseGoalWithSemantics` to return partial results even on error.
  - Added interactive prompt loop in `main`.
  - Added logic to handle `delete_file` intent.

## Try It

```bash
# Build
go build -o cmd/multi_orchestrator/multi_orchestrator ./cmd/multi_orchestrator

# Run
./cmd/multi_orchestrator/multi_orchestrator

# Try a command it might not understand perfectly
> delete the server.go
```
