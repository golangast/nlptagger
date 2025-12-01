# Fix: Generating Handler Content

## Problem

When the user asked to "add handler code to handler.go", the system:
1.  Failed to identify `handler.go` as the target file (it saw it as a destination file due to the word "to").
2.  Created an empty file `new_file.txt` instead.
3.  Didn't generate any actual handler code.

## Solution

1.  **Smarter File Extraction**: I updated `fillCreateFile` to check `destination_file` if the primary `file` entity is missing. This handles phrasing like "add code to [file]".
2.  **Content Generation**: I updated `fillCreateFile` to generate actual Go handler code if the component is "handler". It now generates a valid `func Handler(...)` structure.

## Result

Now, "add handler code to handler.go" will:
1.  Correctly identify `handler.go` as the file to create.
2.  Generate a file with valid Go HTTP handler code inside it.

## Files Modified

- `neural/semantic/intent_templates.go`: Updated `fillCreateFile` logic.

## Try It

```bash
./cmd/multi_orchestrator/multi_orchestrator
> add handler code to the handler.go
```
This should now create `handler.go` with actual code!
