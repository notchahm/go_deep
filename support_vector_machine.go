package go_deep

import "fmt"

type SupportVectorMachine struct {
    weights Vector
    bias ValueType
	regularization_rate ValueType
}

func (this *SupportVectorMachine) get_name() string {
	return "SupportVectorMachine"
}

func (this *SupportVectorMachine) run_example() int {
	inputs := [] Vector { InitVector([]ValueType{1,2}), InitVector([]ValueType{2,1}), InitVector([]ValueType{-1,-2}), InitVector([]ValueType{0,-5}) }
	labels := [] int { 1, -1, 1, -1 }
	max_num_iterations := 100
	this.weights = NewVector( inputs[0].GetSize() )
	this.bias = 0.0
	this.regularization_rate = 0.1
	num_errors := 0
	for iteration_num := 0; iteration_num < max_num_iterations; iteration_num++ {
		var sum_loss ValueType = 0.0
		num_errors = 0
		for example_index := 0; example_index < len(inputs); example_index++ {
			loss := 1.0 - ValueType(labels[example_index])*(this.weights.Dot(inputs[example_index]) + this.bias)
			if (loss > 0.0) {
				sum_loss += loss
				update_vector := CopyVector(inputs[example_index])
				update_vector.Scale(ValueType(labels[example_index]))
				this.weights.AddVector(update_vector)
				this.bias += ValueType(labels[example_index])
				num_errors++
			}
		}
		this.weights.Scale(1.0-this.regularization_rate)
		norm := this.weights.GetNorm()
		if (iteration_num + 1) % 10 == 0 {
			fmt.Printf("%f loss, %f norm after %d iterations\n", sum_loss, norm, iteration_num + 1)
		}
	}
	return num_errors
}
