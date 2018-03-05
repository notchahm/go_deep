package go_deep

type HyperbolicTangentActivation struct {
}

func (this HyperbolicTangentActivation) GetNumOutputs() int {
	// Activation operations occur in place, so they have no size on stack
	return 0
}

func (this HyperbolicTangentActivation) GetStackSize() int {
	// Activation operations occur in place, so they have no size on stack
	return 0
}

func (this HyperbolicTangentActivation) CalculateOutput(input Vector, output Vector) Vector {
	for index := 0; index < input.GetSize(); index++ {
		output.SetValue(index, tanh(input.GetValue(index)))
	}
	return output
}

func (this HyperbolicTangentActivation) EvaluateDelta(output Vector, input_delta Vector, delta Vector, step_size ValueType) {
	for index := 0; index < input_delta.GetSize(); index++ {
		input_delta.SetValue(index, delta.GetValue(index) * (1 - output.GetValue(index) * output.GetValue(index)))
	}
}

