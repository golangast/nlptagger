package bartsimple

// BartProcessCommand generates a response to a command using the BART model.
func (m *SimplifiedBARTModel) BartProcessCommand(command string) (string, error) {
	return m.Reply(command)
}

