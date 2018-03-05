package go_deep

type LongShortTermMemoryCell struct {
	input_net *LinearTransform
	recurrent_net *LinearTransform
	input_velocity Matrix
	recurrent_velocity Matrix
	regularization_rate ValueType
	cell_input_activation Activation
	gate_activation Activation
	state_activation Activation
	input_connection NeuralUnit
	recurrent_connection NeuralUnit
	num_inputs int
	num_hidden int
}

func NewLSTMCell(input_connection NeuralUnit, num_hidden int, regularization_rate ValueType) *LongShortTermMemoryCell {
	this := LongShortTermMemoryCell{}
	this.num_inputs = input_connection.GetNumOutputs()
	this.num_hidden = num_hidden
	this.input_net = new_linear_transform(this.num_inputs, this.num_hidden * 4)
	this.recurrent_net = new_linear_transform(this.num_hidden, this.num_hidden * 4)
	this.input_velocity = NewMatrix(this.num_hidden * 4, this.num_inputs)
	this.recurrent_velocity = NewMatrix(this.num_hidden * 4, this.num_hidden)
	this.cell_input_activation = HyperbolicTangentActivation{}
	this.gate_activation = LogisticActivation{}
	this.state_activation = RectifiedLinearActivation{}
	this.regularization_rate = regularization_rate
	this.input_connection = input_connection
	this.recurrent_connection = &this
	return &this
}

func (this *LongShortTermMemoryCell) GetOutputHeight() int {
	return 1
}

func (this *LongShortTermMemoryCell) GetOutputWidth() int {
	return 1
}

func (this *LongShortTermMemoryCell) GetNumOutputs() int {
	return this.input_net.weights.GetNumRows()
}

func (this *LongShortTermMemoryCell) GetStackSize() int {
	// (Cell input, 3 gates, cell state, and cell output) * num_hidden
	stack_size := (this.input_net.GetStackSize() + this.num_hidden * 2)
	if this.input_connection != nil {
		stack_size += this.input_connection.GetStackSize()
	}
	return stack_size
}

func (this *LongShortTermMemoryCell) CalculateOutput(data Vector, output_stack Vector) Vector {
	num_outputs := this.input_net.weights.GetNumRows()
	preactivation_vector := output_stack.GetSlice(0, num_outputs)
	cell_state := output_stack.GetSlice(num_outputs, num_outputs + this.num_hidden)
	cell_output := output_stack.GetSlice(num_outputs + this.num_hidden, num_outputs + this.num_hidden*2)
	output_stack_remainder := output_stack.GetSlice(num_outputs + this.num_hidden*2, -1)
	data_current := data.GetSlice(data.GetSize()-1, data.GetSize())
	data_remainder := data.GetSlice(0, data.GetSize()-1,)
	input_vector := this.input_connection.CalculateOutput(data_current, output_stack_remainder)
	// TODO: figure out how to handle branches & infinite recursion with stack or abandon it
	hidden_vector := this.recurrent_connection.CalculateOutput(data_remainder, output_stack_remainder)
	left_cell_state := output_stack_remainder.GetSlice(num_outputs, num_outputs + this.num_hidden)

	//g = Wx{t} + Vh{t-1} + b (4.2, 4.4, 4.6. 4.8)
	preactivation_vector = this.input_net.CalculateOutput(input_vector, preactivation_vector)
	preactivation_vector = this.recurrent_net.CalculateOutput(hidden_vector, preactivation_vector)

	cell_input := preactivation_vector.GetSlice(0, this.num_hidden)
	input_gate := preactivation_vector.GetSlice(this.num_hidden, this.num_hidden*2)
	forget_gate := preactivation_vector.GetSlice(this.num_hidden*2, this.num_hidden*3)
	output_gate := preactivation_vector.GetSlice(this.num_hidden*3, this.num_hidden*4)

	this.cell_input_activation.CalculateOutput(cell_input, cell_input)
	this.gate_activation.CalculateOutput(input_gate, input_gate)
	this.gate_activation.CalculateOutput(forget_gate, forget_gate)
	this.gate_activation.CalculateOutput(output_gate, output_gate)

	cell_state.HadamardProductOf(cell_input, input_gate, false)
	cell_state.HadamardProductOf(left_cell_state, forget_gate, true)
	this.state_activation.CalculateOutput(cell_state, cell_state)

	cell_output.HadamardProductOf(cell_state, output_gate, false)
	return cell_output
}

func (this *LongShortTermMemoryCell) EvaluateDelta(data Vector, delta_stack Vector, output_stack Vector, step_size ValueType) {
	num_outputs := this.input_net.weights.GetNumRows()
	cell_input := output_stack.GetSlice(0, this.num_hidden)
	input_gate := output_stack.GetSlice(this.num_hidden, this.num_hidden*2)
	forget_gate := output_stack.GetSlice(this.num_hidden*2, this.num_hidden*3)
	output_gate := output_stack.GetSlice(this.num_hidden*3, this.num_hidden*4)
	cell_state := output_stack.GetSlice(num_outputs, num_outputs + this.num_hidden)
	//cell_output := output_stack.GetSlice(num_outputs + this.num_hidden, num_outputs + this.num_hidden*2)
	output_stack_remainder := output_stack.GetSlice(num_outputs + this.num_hidden*2, -1)
	bottom_cell_output := output_stack_remainder.GetSlice(0, this.num_inputs)
	left_cell_state := output_stack_remainder.GetSlice(num_outputs, num_outputs + this.num_hidden)
	left_cell_output := output_stack_remainder.GetSlice(num_outputs + this.num_hidden, num_outputs + this.num_hidden*2)

	delta_cell_input := delta_stack.GetSlice(0, this.num_hidden)
	delta_input_gate := delta_stack.GetSlice(this.num_hidden, this.num_hidden*2)
	delta_forget_gate := delta_stack.GetSlice(this.num_hidden*2, this.num_hidden*3)
	delta_output_gate := delta_stack.GetSlice(this.num_hidden*3, this.num_hidden*4)
	delta_combined := delta_stack.GetSlice(0, this.num_hidden*4)
	delta_cell_state := delta_stack.GetSlice(this.num_hidden*4, this.num_hidden*5)
	delta_cell_output := delta_stack.GetSlice(this.num_hidden*5, this.num_hidden*6)
	delta_stack_remainder := delta_stack.GetSlice(this.num_hidden*6, -1)

	//d_h = U_T . d_y + V_T . d_g{t+1} (4.11)
	var right_combined_deltas Vector	// TODO: figure out how to get this properly from delta stack
	this.recurrent_net.EvaluateDelta(left_cell_output, delta_cell_output, right_combined_deltas, step_size)
 
	//d_o = d_h * s * o' (4.12)
	delta_output_gate.HadamardProductOf(delta_cell_output, cell_state, false)
	this.gate_activation.EvaluateDelta(output_gate, delta_output_gate, delta_output_gate, step_size)
 
	//d_s = d_h * o * s' + d_s{t+1} * f{t+1} (4.13, sans peephole weights)
	var right_cell_forget_gate Vector	// TODO: figure out how to get this properly from delta stack
	delta_cell_state.HadamardProductOf(delta_cell_output, output_gate, false)
	this.state_activation.EvaluateDelta(cell_state, delta_cell_state, delta_cell_state, step_size)
	delta_cell_state.HadamardProductOf(right_cell_forget_gate, forget_gate, true)
 
	//d_c = d_s * i * c'
	delta_cell_input.HadamardProductOf(delta_cell_state, input_gate, false)
	this.cell_input_activation.EvaluateDelta(cell_input, delta_cell_input, delta_cell_input, step_size)
 
	//d_f = d_s * s{t-1} * f'
	delta_forget_gate.HadamardProductOf(delta_cell_state, left_cell_state, false)
	this.gate_activation.EvaluateDelta(forget_gate, delta_forget_gate, delta_forget_gate, step_size)
 
	//d_i = d_s * c * i'
	delta_input_gate.HadamardProductOf(delta_cell_state, cell_input, false)
	this.gate_activation.EvaluateDelta(input_gate, delta_input_gate, delta_input_gate, step_size)

	this.input_net.EvaluateDelta(bottom_cell_output, delta_stack_remainder, delta_combined, step_size)

	this.input_connection.EvaluateDelta(data, delta_stack_remainder, output_stack_remainder, step_size)
	this.recurrent_connection.EvaluateDelta(data, delta_stack_remainder, output_stack_remainder, step_size)
}

func (this *LongShortTermMemoryCell) ApplyRegularization(regularization_rate ValueType) ValueType {
	this.input_connection.ApplyRegularization(regularization_rate)
	return this.input_net.ApplyRegularization(regularization_rate*this.regularization_rate) 
}

func (this *LongShortTermMemoryCell) ApplyMomentum(momentum_rate ValueType) {
	this.input_connection.ApplyMomentum(momentum_rate)
	this.input_velocity.Scale(momentum_rate)
	this.input_net.weights.AddMatrix(this.input_velocity)
	this.recurrent_velocity.Scale(momentum_rate)
	this.recurrent_net.weights.AddMatrix(this.recurrent_velocity)
}

func (this *LongShortTermMemoryCell) GetAsString() string {
	return this.input_net.GetAsString() + this.recurrent_net.GetAsString()
}

func (this *LongShortTermMemoryCell) WriteToFile(filename string) {
}
