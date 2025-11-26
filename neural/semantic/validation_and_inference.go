package semantic

// ValidateAndInferProperties orchestrates the validation and inference process
func ValidateAndInferProperties(output *SemanticOutput) error {
	if output == nil {
		return nil
	}

	// First, infer properties
	InferProperties(output.TargetResource)

	// Then, validate the resource with inferred properties
	if err := ValidateResource(output.TargetResource, output.Context); err != nil {
		return err
	}

	return nil
}
