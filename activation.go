package go_deep

type Activation interface {
	GetNumOutputs() int
	GetStackSize() int
	CalculateOutput(input Vector, output Vector) Vector
	EvaluateDelta(output Vector, input_delta Vector, delta Vector, step_size ValueType)
}

