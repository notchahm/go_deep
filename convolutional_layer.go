package go_deep

type ConvolutionalLayer struct {
	net *LinearTransform
	velocity Matrix
	activation Activation
	regularization_rate ValueType
	dropout Activation
	previous NeuralUnit
	filter_height int
	filter_width int
	num_channels int
	stride int
	num_inputs int
	input_height int
	input_width int
	input_depth int
	output_height int
	output_width int
	filter_row_size int
}

type ConvolutionFunction func (*ConvolutionalLayer, Vector, Vector, int, interface{})

func NewConvLayer(prev NeuralUnit, filter_height int, filter_width int, num_channels int, stride int, regularization_rate ValueType) *ConvolutionalLayer {
	this := ConvolutionalLayer{}
	this.filter_height = filter_height
	this.filter_width = filter_width
	this.num_channels = num_channels
	this.stride = stride
	this.num_inputs = filter_height * filter_width * prev.GetNumOutputs()
	this.input_height = prev.GetOutputHeight()
	this.input_width = prev.GetOutputWidth()
	this.input_depth = prev.GetNumOutputs()
	this.output_height = ((this.input_height - filter_height) / stride) + 1
	this.output_width = ((this.input_width - filter_width) / stride) + 1
	this.filter_row_size = filter_width * num_channels
	this.net = new_linear_transform(this.num_inputs, num_channels)
	this.velocity = NewMatrix(num_channels, this.num_inputs)
	this.activation = RectifiedLinearActivation{}
	this.regularization_rate = regularization_rate
	this.dropout = nil
	this.previous = prev
	return &this
}

func (this *ConvolutionalLayer) GetOutputHeight() int {
	return this.output_height
}

func (this *ConvolutionalLayer) GetOutputWidth() int {
	return this.output_width
}

func (this *ConvolutionalLayer) GetNumOutputs() int {
	return this.net.weights.GetNumRows()
}

func (this *ConvolutionalLayer) GetStackSize() int {
	stack_size := (this.net.GetStackSize() * this.output_height * this.output_width) + this.activation.GetStackSize()
	if this.previous != nil {
		stack_size += this.previous.GetStackSize()
	}
	return stack_size
}

func (this *ConvolutionalLayer) Convolve(layer_input Vector, layer_output Vector, operation ConvolutionFunction, operation_parameter interface{}) {
	output_index := 0
	for row_index := 0; row_index < this.input_height-this.filter_height; row_index += this.stride {
		for col_index := 0; col_index < this.input_width-this.filter_width; col_index += this.stride {
			output_offset := (row_index * this.output_width + col_index) * this.num_channels
			conv_output := layer_output.GetSlice(output_offset, output_offset+this.num_channels)
			for filter_row_index := 0; filter_row_index < this.filter_height; filter_row_index++ {
				input_offset := (row_index+filter_row_index) * this.input_width * this.input_depth + (col_index*this.input_depth)
				conv_input := layer_input.GetSlice(input_offset,input_offset+this.filter_row_size)
				operation(this, conv_input, conv_output, filter_row_index, operation_parameter)
			}
			output_index++
		}
	}
}

func forward_operation(this *ConvolutionalLayer, conv_input Vector, conv_output Vector, filter_row_index int, parameter interface{}) {
	if filter_row_index == 0 {
		conv_output.Copy(this.net.bias)
	}
	for channel_index := 0; channel_index < this.num_channels; channel_index++ {
		conv_output.SetValue(channel_index, this.net.weights.GetRow(channel_index).Dot(conv_input))
	}
}

func backward_operation(this *ConvolutionalLayer, conv_input Vector, conv_output Vector, filter_row_index int, parameter interface{}) {
	var step_size ValueType = parameter.(ValueType)
	if filter_row_index == 0 {
		conv_input.SetZero()
	}
	for channel_index := 0; channel_index < this.num_channels; channel_index++ {
		if step_size > 0 {
			this.net.weights.GetRow(channel_index).AddScaledVector(step_size * conv_output.GetValue(channel_index), conv_input);
			this.velocity.GetRow(channel_index).AddScaledVector(step_size * conv_output.GetValue(channel_index), conv_input);
		}
		conv_input.AddScaledVector(conv_output.GetValue(channel_index), this.net.weights.GetRow(channel_index))
	}
}

func accumulate_delta_operation(this *ConvolutionalLayer, conv_input Vector, conv_output Vector, filter_row_index int, parameter interface{}) {
	var delta_vector Vector = parameter.(Vector)
	if filter_row_index == 0 && conv_output != delta_vector {
		delta_vector.AddVector(conv_output)
	}
}

func (this *ConvolutionalLayer) CalculateOutput(data Vector, output_stack Vector) Vector {
	num_outputs := this.net.weights.GetNumRows()
	output_vector := output_stack.GetSlice(0, num_outputs * this.output_height * this.output_width)
	output_stack_remainder := output_stack.GetSlice(output_vector.GetSize(), -1)
	input_vector := this.previous.CalculateOutput(data, output_stack_remainder)
	this.Convolve(input_vector, output_vector, forward_operation, 0)
	this.activation.CalculateOutput(output_vector, output_vector)
	// Optional apply dropout
	if this.dropout != nil {
		this.dropout.CalculateOutput(output_vector, output_vector)
	}
	return output_vector
}

func (this *ConvolutionalLayer) EvaluateDelta(data Vector, delta_stack Vector, output_stack Vector, step_size ValueType) {
	num_outputs := this.net.weights.GetNumRows()
	output_vector := output_stack.GetSlice(0, num_outputs * this.output_height * this.output_width)
	output_stack_remainder := output_stack.GetSlice(output_vector.GetSize(), -1)
	delta_output_vector := delta_stack.GetSlice(0, num_outputs * this.output_height * this.output_width)
	delta_stack_remainder := delta_stack.GetSlice(delta_output_vector.GetSize(), -1)

	this.Convolve(delta_stack_remainder, delta_output_vector, accumulate_delta_operation, delta_output_vector)
	this.activation.EvaluateDelta(output_vector, delta_output_vector, delta_output_vector, step_size)
	this.Convolve(delta_stack_remainder, delta_output_vector, backward_operation, step_size)

	this.previous.EvaluateDelta(data, delta_stack_remainder, output_stack_remainder, step_size)
}

func (this *ConvolutionalLayer) ApplyRegularization(regularization_rate ValueType) ValueType {
	this.previous.ApplyRegularization(regularization_rate)
	return this.net.ApplyRegularization(regularization_rate*this.regularization_rate)
}

func (this *ConvolutionalLayer) ApplyMomentum(momentum_rate ValueType) {
	this.previous.ApplyMomentum(momentum_rate)
	this.velocity.Scale(momentum_rate)
	this.net.weights.AddMatrix(this.velocity)
}

func (this *ConvolutionalLayer) GetAsString() string {
	return this.net.GetAsString()
}

func (this *ConvolutionalLayer) WriteToFile(filename string) {
}
