package go_deep

type LogisticActivation struct {
}

func (this LogisticActivation) GetNumOutputs() int {
	// Activation operations occur in place, so they have no size on stack
	return 0
}

func (this LogisticActivation) GetStackSize() int {
	// Activation operations occur in place, so they have no size on stack
	return 0
}

func (this LogisticActivation) CalculateOutput(input Vector, output Vector) Vector {
	for index := 0; index < input.GetSize(); index++ {
		output.SetValue(index, 1.0/(1.0+exp(-1.0*input.GetValue(index))))
	}
	return output
}

func (this LogisticActivation) EvaluateDelta(output Vector, input_delta Vector, delta Vector, step_size ValueType) {
	for index := 0; index < input_delta.GetSize(); index++ {
		input_delta.SetValue(index, delta.GetValue(index) * output.GetValue(index) * (1 - output.GetValue(index)))
	}
}

