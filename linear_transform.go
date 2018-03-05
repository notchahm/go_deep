package go_deep

type LinearTransform struct {
	weights Matrix
	bias Vector
}

func new_linear_transform(num_inputs int, num_outputs int) *LinearTransform {
	this := LinearTransform{}
	this.weights = NewMatrix(num_outputs, num_inputs)
	this.weights.Randomize(sqrt(ValueType(1.0)/ValueType(num_inputs)))
	this.bias = NewVector(num_outputs)
	return &this
}

func (this *LinearTransform) GetNumOutputs() int {
	return this.weights.GetNumRows()
}

func (this *LinearTransform) GetStackSize() int {
	return this.weights.GetNumRows()
}

func (this *LinearTransform) GetAsString() string {
	string_representation := "{\n"
	string_representation += "  weights: " + this.weights.GetAsString() + ",\n"
	string_representation += "  bias: " + this.bias.GetAsString() + "\n"
	string_representation += "}"
	return string_representation
}

func (this *LinearTransform) CalculateOutput(input Vector, output Vector) Vector{
	// Calculate output vector as matrix vector product plus bias (y = Wx + b)
	output.ProductOf(this.weights, input)	// Wx
	output.AddVector(this.bias)				// + b
	return output
}

func (this *LinearTransform) EvaluateDelta(input Vector, delta_stack Vector, output_delta Vector, step_size ValueType) {
	// If there's a non-zero step size, update weights directly by apply the gradient as the cross product of output_delta and layer inputs
	if step_size > 0.0 {
		this.weights.AddScaledCrossProduct(step_size, output_delta, input)
		this.bias.AddScaledVector(step_size, output_delta)
	}
	// Backpropagate error by calculating delta vector through chain rule (input_delta = W_T output_delta)
	if delta_stack.GetSize() > 0 {
		input_delta := delta_stack.GetSlice(0,this.weights.GetNumColumns())
		input_delta.ProductOfTransposed(this.weights, output_delta)
	}
}

func (this *LinearTransform) ApplyRegularization(regularization_rate ValueType) ValueType {
	// L2 regularization update implemented as weight decay
	this.weights.Scale(1.0-regularization_rate)
	// Also decay bias term
	this.bias.Scale(1.0-regularization_rate)
	// return L2 objective as weights norm
	return this.weights.GetNorm()
}
