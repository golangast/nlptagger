package neural

type SemanticRolePredictor interface {
	PredictRoles(tokens []string) ([]string, error)
}