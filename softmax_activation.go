package go_deep

type SoftmaxActivation struct {
}

func (this SoftmaxActivation) GetNumOutputs() int {
	// Activation operations occur in place, so they have no size on stack
	return 0
}

func (this SoftmaxActivation) GetStackSize() int {
	// Activation operations occur in place, so they have no size on stack
	return 0
}

func (this SoftmaxActivation) CalculateOutput(input Vector, output Vector) Vector {
	var sum_across_classes ValueType = 1.0
	max_index, max_value := input.GetMax()
	
	for index := 0; index < input.GetSize(); index++ {
		if index != max_index {
			norm_exponent := input.GetValue(index) - max_value
			if norm_exponent < -100.0 {
				output.SetValue(index, 0.0)
			} else {
				output.SetValue(index, exp(norm_exponent))
				sum_across_classes += output.GetValue(index)
			}
		} else {
			output.SetValue(index, 1.0)
		}
	}
	output.Scale(1.0/sum_across_classes);
	return output
}

func (this SoftmaxActivation) EvaluateDelta(output Vector, input_delta Vector, delta Vector, step_size ValueType) {
	for index := 0; index < input_delta.GetSize(); index++ {
		input_delta.SetValue(index, delta.GetValue(index) * output.GetValue(index) * (1 - output.GetValue(index)))
	}
}
