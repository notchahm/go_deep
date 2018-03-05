package go_deep

import "fmt"

type Perceptron struct {
    weights Vector
    bias ValueType
}

func (this *Perceptron) get_name() string {
    return "Perceptron"
}

func (this *Perceptron) run_example() int {
	inputs := [] Vector { InitVector([]ValueType{1,2}), InitVector([]ValueType{2,1}), InitVector([]ValueType{-1,-2}), InitVector([]ValueType{0,-5}) }
	labels := [] int { 1, 0, 1, 0 }
	max_num_iterations := 100
	this.weights = NewVector( inputs[0].GetSize() )
	this.bias = 0.0
	num_errors := 0
	for iteration_num := 0; iteration_num < max_num_iterations; iteration_num++ {
		num_errors = 0
		for example_index := 0; example_index < len(inputs); example_index++ {
			error := ValueType(labels[example_index])
			if (this.weights.Dot(inputs[example_index]) + this.bias > 0) {
				error -= 1.0
			}
			if (error != 0.0) {
				num_errors++
				update_vector := CopyVector(inputs[example_index])
				update_vector.Scale(error)
				this.weights.AddVector(update_vector)
				this.bias += error
			}
		}
		fmt.Printf("%d errors after %d iterations\n", num_errors, iteration_num + 1)
		if (num_errors == 0) {
			break
		}
	}
	return num_errors
}
