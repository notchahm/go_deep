package go_deep

import "fmt"

type LogisticRegression struct {
	weights Vector
	bias ValueType
	regularization_rate ValueType
}

func (this *LogisticRegression) get_name() string {
    return "LogisticRegression"
}

func (this *LogisticRegression) run_example() int {
	inputs := [] Vector { InitVector([]ValueType{1,2}), InitVector([]ValueType{2,1}), InitVector([]ValueType{-1,-2}), InitVector([]ValueType{0,-5}) }
	labels := [] ValueType { 1.0, 0.0, 1.0, 0.0 }
	max_num_iterations := 100
	this.weights = NewVector( inputs[0].GetSize() )
	this.bias = 0.0
	this.regularization_rate = 0.1
	num_errors := 0
	for iteration_num := 0; iteration_num < max_num_iterations; iteration_num++ {
		var sum_loss ValueType = 0.0
		num_errors = 0
		for example_index := 0; example_index < len(inputs); example_index++ {
			linear := this.weights.Dot(inputs[example_index]) + this.bias
			logistic := 1.0/(1.0+exp(-1.0*linear))
			logistic_prime := logistic * (1.0-logistic)
			diff := labels[example_index] - logistic
			if diff > 0.5 || diff < -0.5 {
				num_errors++
			}
			sum_loss += 0.5 * diff*diff
			gradient := diff * -1.0 * logistic_prime
			update_vector := CopyVector(inputs[example_index])
			update_vector.Scale(-1.0 * gradient)
			this.weights.AddVector(update_vector)
			this.bias -= ValueType(gradient)
		}
		this.weights.Scale(1.0-this.regularization_rate)
		norm := this.weights.GetNorm()
		if (iteration_num + 1) % 10 == 0 {
			fmt.Printf("%f loss, %f norm after %d iterations\n", sum_loss, norm, iteration_num + 1)
		}
	}
	return num_errors
}
