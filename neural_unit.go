package go_deep

type NeuralUnit interface {
	CalculateOutput(input Vector, output Vector) Vector
	EvaluateDelta(input Vector, delta Vector, output Vector, step_size ValueType)
	ApplyRegularization(regularization_rate ValueType) ValueType
	ApplyMomentum(momentum_rate ValueType)
	GetOutputHeight() int
	GetOutputWidth() int
	GetNumOutputs() int
	GetStackSize() int
	GetAsString() string
	WriteToFile(filename string)
}

