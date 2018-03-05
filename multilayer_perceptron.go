package go_deep

import "fmt"

type MultilayerPerceptron struct {
	input_layer *InputLayer
	hidden_layers []*Layer
	output_layer *ClassificationOutputLayer
}

func NewMultilayerNetwork(input_size int, hidden_sizes []int, output_size int, regularization_rate ValueType) *MultilayerPerceptron {
	this := MultilayerPerceptron{}
	this.input_layer = NewInputLayer(input_size)
	var prev_layer NeuralUnit = this.input_layer
	for hidden_layer_index := 0; hidden_layer_index < len(hidden_sizes); hidden_layer_index++ {
		this.hidden_layers = append(this.hidden_layers, NewLayer( prev_layer, hidden_sizes[hidden_layer_index], regularization_rate ))
		prev_layer = this.hidden_layers[hidden_layer_index]
	}
	this.output_layer = NewClassificationOutputLayer( prev_layer, output_size, regularization_rate )
	return &this
}

func (this *MultilayerPerceptron) get_name() string {
	return "MultilayerPerceptron"
}

func (this *MultilayerPerceptron) GetOutputHeight() int {
	return 1
}

func (this *MultilayerPerceptron) GetOutputWidth() int {
	return 1
}

func (this *MultilayerPerceptron) GetNumOutputs() int {
	return this.output_layer.GetNumOutputs()
}

func (this *MultilayerPerceptron) GetStackSize() int {
	return this.output_layer.GetStackSize()
}

func (this *MultilayerPerceptron) CalculateOutput(input Vector, output Vector) Vector {
	return this.output_layer.CalculateOutput(input, output)
}

func (this *MultilayerPerceptron) calculate_loss(label Vector, output Vector, delta_loss Vector) ValueType {
	//cross entropy loss: SUM - t ln y
	var loss ValueType = 0.0
	for index := 0; index < output.GetSize(); index++ {
		delta_loss.SetValue(index, label.GetValue(index) - output.GetValue(index))
		if label.GetValue(index) > 0 && output.GetValue(index) > 0 {
			loss -= label.GetValue(index) * log(output.GetValue(index))
		}
	}
	return loss
}

func (this *MultilayerPerceptron) EvaluateDelta(input Vector, input_delta Vector, output_delta Vector, step_size ValueType) {
	this.output_layer.EvaluateDelta(input, input_delta, output_delta, step_size)
}

func (this *MultilayerPerceptron) ApplyRegularization(step_size ValueType) ValueType {
	if step_size > 0.0 {
		return this.output_layer.ApplyRegularization(step_size)
	} else {
		return this.output_layer.net.weights.GetNorm()
	}
}

func (this *MultilayerPerceptron) ApplyMomentum(momentum_rate ValueType) {
	this.output_layer.ApplyMomentum(momentum_rate)
}

func (this *MultilayerPerceptron) GetAsString() string {
	return this.output_layer.GetAsString()
}

func (this *MultilayerPerceptron) WriteToFile(filename string) {
}

func (this *MultilayerPerceptron) run_example() int {
	inputs := [] Vector { InitVector([]ValueType{1,0}), InitVector([]ValueType{1,1}), InitVector([]ValueType{0,1}), InitVector([]ValueType{0,0}) }
	labels := [] int { 1, 0, 1, 0 }
	max_num_iterations := 100000
	regularization_rate := ValueType(0.000001)
	this.input_layer = NewInputLayer(inputs[0].GetSize())
	this.hidden_layers = append(this.hidden_layers, NewLayer( this.input_layer, 20, regularization_rate ))
	this.hidden_layers = append(this.hidden_layers, NewLayer( this.hidden_layers[0], 20, regularization_rate ))
	this.output_layer = NewClassificationOutputLayer( this.hidden_layers[0], 2, regularization_rate )
	num_errors := 0
	var initial_step_size ValueType = 0.1
	output_stack := NewVector(this.output_layer.GetStackSize())
	delta_stack := NewVector(this.output_layer.GetStackSize())
	for iteration_num := 0; iteration_num < max_num_iterations; iteration_num++ {
		num_errors = 0
		step_size := initial_step_size/(1 + (ValueType(iteration_num)/ValueType(max_num_iterations)))
		var sum_loss ValueType = 0.0
		for example_index := 0; example_index < len(inputs); example_index++ {
			output := this.output_layer.CalculateOutput(inputs[example_index], output_stack)
			label := NewVector(2)
			label.SetValue(labels[example_index], 1.0)
			loss := this.calculate_loss(label, output, delta_stack)
			if loss > -1.0 * log(0.5) {
				num_errors += 1
//				fmt.Printf("%v, %d\n", output, labels[example_index])
			}
			sum_loss += loss
			this.output_layer.EvaluateDelta(inputs[example_index], delta_stack, output_stack, step_size)
		}
		norm := this.output_layer.ApplyRegularization(step_size)
		if (iteration_num + 1) % 10000 == 0 {
			fmt.Printf("%f loss, %f norm after %d iterations\n", sum_loss, norm, iteration_num + 1)
		}
	}
/*
	dataset := InitImageClassificationDataset(len(inputs))
	for example_index := 0; example_index < len(inputs); example_index++ {
		dataset.AddInput(inputs[example_index])
		label := NewVector(2)
		label.SetValue(labels[example_index], 1.0)
		dataset.AddLabel(label)
	}
	dataset.SetTestSplit(0)
	
	optimizer := NewStochasticGradientOptimizer(1.0, 0.0, 10)
	model := Model{Network:NewMultilayerNetwork(dataset.GetNumFeatures(), [] int { 10 }, dataset.GetNumClasses(), this.regularization_rate), CalcLoss: CrossEntropyLoss}
	optimizer.Train(max_num_iterations, dataset, model, 100)
*/
	return num_errors
}
