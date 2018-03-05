package go_deep

type ClassificationOutputLayer struct {
	net *LinearTransform
	velocity Matrix
	activation Activation
	regularization_rate ValueType
	previous NeuralUnit
}

func NewClassificationOutputLayer(prev NeuralUnit, size int, regularization_rate ValueType) *ClassificationOutputLayer {
	this := ClassificationOutputLayer{}
	this.net = new_linear_transform(prev.GetNumOutputs(), size)
	this.velocity = NewMatrix(size, prev.GetNumOutputs())
	this.activation = SoftmaxActivation{}
	this.regularization_rate = regularization_rate
	this.previous = prev
	return &this
}

func (this *ClassificationOutputLayer) GetOutputHeight() int {
	return 1
}

func (this *ClassificationOutputLayer) GetOutputWidth() int {
	return 1
}

func (this *ClassificationOutputLayer) GetNumOutputs() int {
	return this.net.weights.GetNumRows()
}

func (this *ClassificationOutputLayer) GetStackSize() int {
	stack_size := this.net.GetStackSize() + this.activation.GetStackSize()
	if this.previous != nil {
		stack_size += this.previous.GetStackSize()
	}
	return stack_size
}

func (this *ClassificationOutputLayer) CalculateOutput(data Vector, output_stack Vector) Vector {
	num_inputs, num_outputs := this.net.weights.GetDimensions()
	output_vector := output_stack.GetSlice(0, num_outputs)
	input_vector := output_stack.GetSlice(num_outputs,num_outputs+num_inputs)
	input_vector = this.previous.CalculateOutput(data, input_vector)
	this.net.CalculateOutput(input_vector, output_vector)
	this.activation.CalculateOutput(output_vector, output_vector)
	return output_vector
}

func (this *ClassificationOutputLayer) EvaluateDelta(data Vector, delta_stack Vector, output_stack Vector, step_size ValueType) {
	num_inputs, num_outputs := this.net.weights.GetDimensions()
	//output_vector := output_stack.GetSlice(0, num_outputs)
	input_vector := output_stack.GetSlice(num_outputs,num_outputs+num_inputs)
	if input_vector.GetSize() == 0 {
		input_vector = data
	}
	delta_output_vector := delta_stack.GetSlice(0, num_outputs)
	delta_input_vector := delta_stack.GetSlice(num_outputs,num_outputs+num_inputs)

	// Skip chaining softmax activation -- assume cancel with denominator of cross entropy loss
	//this.activation.EvaluateDelta(output_vector, delta_output_vector, delta_output_vector, step_size)

	this.net.EvaluateDelta(input_vector, delta_input_vector, delta_output_vector, step_size)
	if step_size > 0.0 {
		this.velocity.AddScaledCrossProduct(step_size, delta_output_vector, input_vector)
	}
	this.previous.EvaluateDelta(data, delta_input_vector, input_vector, step_size)
}

func (this *ClassificationOutputLayer) ApplyRegularization(regularization_rate ValueType) ValueType {
	this.previous.ApplyRegularization(regularization_rate)
	return this.net.ApplyRegularization(regularization_rate*this.regularization_rate)
}

func (this *ClassificationOutputLayer) ApplyMomentum(momentum_rate ValueType) {
	this.previous.ApplyMomentum(momentum_rate)
	this.velocity.Scale(momentum_rate)
	this.net.weights.AddMatrix(this.velocity)
}

func (this *ClassificationOutputLayer) GetAsString() string {
	string_representation := "{net: " + this.net.GetAsString() + "\n"
	string_representation += "  velocity: " + this.velocity.GetAsString() + "\n"
	string_representation += "}\n"

	return string_representation
}

func (this *ClassificationOutputLayer) WriteToFile(filename string) {
}
