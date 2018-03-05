package go_deep

import "fmt"

type Layer struct {
	net *LinearTransform
	velocity Matrix
	activation LogisticActivation
	regularization_rate ValueType
	previous NeuralUnit
}

func NewLayer(prev NeuralUnit, size int, regularization_rate ValueType) *Layer {
	this := Layer{}
	this.net = new_linear_transform(prev.GetNumOutputs(), size)
	this.velocity = NewMatrix(size, prev.GetNumOutputs())
	this.activation = LogisticActivation{}
	this.regularization_rate = regularization_rate
	this.previous = prev
	return &this
}

func (this *Layer) GetOutputHeight() int {
	return 1
}

func (this *Layer) GetOutputWidth() int {
	return 1
}

func (this *Layer) GetNumOutputs() int {
	return this.net.weights.GetNumRows()
}

func (this *Layer) GetStackSize() int {
	stack_size := this.net.GetStackSize() + this.activation.GetStackSize()
	if this.previous != nil {
		stack_size += this.previous.GetStackSize()
	}
	return stack_size
}

func (this *Layer) CalculateOutput(data Vector, output_stack Vector) Vector {
	_, num_outputs := this.net.weights.GetDimensions()
	output_vector := output_stack.GetSlice(0, num_outputs)
	output_stack_remainder := output_stack.GetSlice(num_outputs, -1)
	input_vector := this.previous.CalculateOutput(data, output_stack_remainder)
	this.net.CalculateOutput(input_vector, output_vector)
	this.activation.CalculateOutput(output_vector, output_vector)
	return output_vector
}

func (this *Layer) EvaluateDelta(data Vector, delta_stack Vector, output_stack Vector, step_size ValueType) {
	num_inputs, num_outputs := this.net.weights.GetDimensions()
	output_vector := output_stack.GetSlice(0, num_outputs)
	output_stack_remainder := output_stack.GetSlice(num_outputs, -1)
	var input_vector Vector
	if output_stack_remainder.GetSize() == 0 {
		input_vector = this.previous.CalculateOutput(data, output_stack_remainder)
	} else {
		input_vector = output_stack_remainder.GetSlice(0, num_inputs)
	}
	delta_output_vector := delta_stack.GetSlice(0, num_outputs)
	delta_stack_remainder := delta_stack.GetSlice(num_outputs,-1)
	this.activation.EvaluateDelta(output_vector, delta_output_vector, delta_output_vector, step_size)
	//this.CheckGradient(delta_output_vector, input_vector, data
	this.net.EvaluateDelta(input_vector, delta_stack_remainder, delta_output_vector, step_size)
	if step_size > 0.0 {
		this.velocity.AddScaledCrossProduct(step_size, delta_output_vector, input_vector)
	}
	this.previous.EvaluateDelta(data, delta_stack_remainder, input_vector, step_size)
}

func (this *Layer) CheckLoss(train_index int, data Dataset, model Model) ValueType {
	output_stack := NewVector(model.Network.GetStackSize())
	delta_stack := NewVector(model.Network.GetStackSize())

	input, target := data.GetTrainInstance(train_index)
	output := model.Network.CalculateOutput(input, output_stack)
	return model.CalcLoss(target, output, delta_stack)
}

func (this *Layer) CheckGradient(row_gradient Vector, column_gradient Vector, train_index int, data Dataset, model Model) {
	var epsilon ValueType = 0.000001
	num_columns, num_rows := this.net.weights.GetDimensions()
	for row_index := 0; row_index < num_rows; row_index++ {
		for col_index := 0; col_index < num_columns; col_index++ {
			weight_value := this.net.weights.GetValueAt(row_index, col_index)
			this.net.weights.SetValueAt(row_index, col_index, weight_value + epsilon)
			loss_up := this.CheckLoss(train_index, data, model)
			this.net.weights.SetValueAt(row_index, col_index, weight_value - epsilon)
			loss_down := this.CheckLoss(train_index, data, model)
			this.net.weights.SetValueAt(row_index, col_index, weight_value)
			numerical_gradient := ValueType(loss_down - loss_up) / ValueType(epsilon * 2.0);
			derived_gradient := row_gradient.GetValue(row_index) * column_gradient.GetValue(col_index)
			diff := numerical_gradient - derived_gradient
			ratio := ValueType(1.0)
			if derived_gradient != 0.0 && numerical_gradient != 0.0 {
				ratio = numerical_gradient / derived_gradient
			}
			if (diff > 0.01 || diff < -0.01) && (ratio < 0.9 || ratio > 1.1) {
				fmt.Printf("[%d, %d]: diff: %.4f, ratio: %.3f\n", row_index, col_index, diff, ratio);
			}
		}
	}
}

func (this *Layer) ApplyRegularization(regularization_rate ValueType) ValueType {
	this.previous.ApplyRegularization(regularization_rate)
	return this.net.ApplyRegularization(regularization_rate*this.regularization_rate)
}

func (this *Layer) ApplyMomentum(momentum_rate ValueType) {
	this.previous.ApplyMomentum(momentum_rate)
	this.velocity.Scale(momentum_rate)
	this.net.weights.AddMatrix(this.velocity)
}

func (this *Layer) GetAsString() string {
    return this.net.GetAsString()
}

func (this *Layer) WriteToFile(filename string) {
}
