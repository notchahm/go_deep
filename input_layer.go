package go_deep

type InputLayer struct {
	size int
}

func NewInputLayer(size int) *InputLayer {
	return &InputLayer{size: size}
}

func (this *InputLayer) GetOutputHeight() int {
	return 1
}

func (this *InputLayer) GetOutputWidth() int {
	return 1
}

func (this *InputLayer) GetNumOutputs() int {
	return this.size
}

func (this *InputLayer) GetStackSize() int {
	return 0
}

func (this *InputLayer) CalculateOutput(data Vector, output_stack Vector) Vector {
	return data
}

func (this *InputLayer) EvaluateDelta(data Vector, delta_stack Vector, output_stack Vector, step_size ValueType) {
	return
}

func (this *InputLayer) ApplyRegularization(regularization_rate ValueType) ValueType {
	return 0.0
}

func (this *InputLayer) ApplyMomentum(momentum_rate ValueType) {
}

func (this *InputLayer) GetAsString() string {
    return ""
}
func (this *InputLayer) WriteToFile(filename string) {
}
