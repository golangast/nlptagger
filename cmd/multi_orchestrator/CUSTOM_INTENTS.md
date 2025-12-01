# Enhancement: Custom Intents and Aliasing

## Problem

The user wanted more flexibility than just selecting from a predefined list of intents when the system fails to understand a command. They wanted to be able to define their own intents and entities manually, and map (alias) them to known actions if needed.

## Solution

I added a **"Define Custom / Alias"** option to the interactive clarification menu.

### How It Works

1.  **Select Option 5**: When prompted, choose "5. Define Custom / Alias".
2.  **Enter Intent**: Type any intent name you want (e.g., "deploy_app", "nuke_it").
3.  **Enter Entities**: Provide key-value pairs for entities (e.g., `file=server.go, env=prod`).
4.  **Aliasing**:
    *   If the intent you entered is **known** (e.g., you manually typed `delete_file`), it executes immediately.
    *   If the intent is **unknown** (e.g., `nuke_it`), the system asks if you want to **map it** to a known action.
    *   You can then select a target action (e.g., `delete_file`), and the system will execute that action using the entities you provided.

### Example Scenario

**User**: "nuke the server"
**System**: "Unknown intent."
**User**: Selects **5. Define Custom / Alias**
**System**: "Enter Intent Name"
**User**: `nuke_it`
**System**: "Enter Entities"
**User**: `file=server.go`
**System**: "Intent 'nuke_it' is not natively supported. Map it to a known action?"
**User**: Selects **3. delete_file**
**System**: Executes `delete_file` on `server.go`.

## Files Modified

- `cmd/multi_orchestrator/main.go`: Added the custom input and aliasing logic to the interactive switch statement.

## Try It

```bash
./cmd/multi_orchestrator/multi_orchestrator
> nuke the server
```
Select option 5 and follow the prompts!
