package go_deep

import "math/rand"

type Dropout struct {
	dropout_rate float32
}

func (this Dropout) GetNumOutputs() int {
	// Activation operations occur in place, so they have no size on stack
	return 0
}

func (this Dropout) GetStackSize() int {
	// Activation operations occur in place, so they have no size on stack
	return 0
}

func (this Dropout) CalculateOutput(input Vector, output Vector) Vector {
	for index := 0; index < input.GetSize(); index++ {
		if rand.Float32() < this.dropout_rate {
			output.SetValue(index, 0.0)
		}
	}
	return output
}

func (this Dropout) EvaluateDelta(output Vector, input_delta Vector, delta Vector, step_size ValueType) {
	for index := 0; index < input_delta.GetSize(); index++ {
		if output.GetValue(index) == 0.0 {
			input_delta.SetValue(index, 0.0)
		}
	}
}

