# Fix: Intent Classification for File Creation

## Problem

The user tried to "create a handler file handler.go", but the system classified it as `unknown` because the intent classifier was too strict. It saw the word "handler" (a component keyword) and assumed it wasn't a simple file creation, but failed to match it to "add feature" because of missing keywords.

## Solution

1.  **Relaxed Intent Classifier**: I updated `intent_classifier.go` to allow `IntentCreateFile` even if component keywords (like "handler") are present, as long as "file" or a file extension is detected.
2.  **Updated Interactive Prompt**: I added `create_file` as an explicit option in the interactive clarification menu.

## Result

Now, "create a handler file handler.go" should be correctly classified as `create_file`. Even if it isn't, you can now select **4. create_file** from the menu, and the system will generate the file using the `generateFromSemantic` function I added earlier.

## Files Modified

- `neural/semantic/intent_classifier.go`: Relaxed `IntentCreateFile` rules.
- `cmd/multi_orchestrator/main.go`: Added `create_file` to the interactive prompt.

## Try It

```bash
./cmd/multi_orchestrator/multi_orchestrator
> create a handler file handler.go
```
This should now work automatically!
