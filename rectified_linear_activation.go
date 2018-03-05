package go_deep

type RectifiedLinearActivation struct {
}

func (this RectifiedLinearActivation) GetNumOutputs() int {
	// Activation operations occur in place, so they have no size on stack
	return 0
}

func (this RectifiedLinearActivation) GetStackSize() int {
	// Activation operations occur in place, so they have no size on stack
	return 0
}

func (this RectifiedLinearActivation) CalculateOutput(input Vector, output Vector) Vector {
	for index := 0; index < input.GetSize(); index++ {
		if input.GetValue(index) < 0.0 {
			output.SetValue(index, 0.0)
		}
	}
	return output
}

func (this RectifiedLinearActivation) EvaluateDelta(output Vector, input_delta Vector, delta Vector, step_size ValueType) {
	for index := 0; index < input_delta.GetSize(); index++ {
		if output.GetValue(index) == 0.0 {
			input_delta.SetValue(index, 0.0)
		} else if input_delta.GetValue(index) > 10.0 {
			// Gradient clipping to mitigate exploding gradient
			input_delta.SetValue(index, 10.0)
		}
	}
}

