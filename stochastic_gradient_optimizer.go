package go_deep

import "math/rand"
import "runtime"
import "sync"
import "fmt"

type StochasticGradientOptimizer struct {
	initial_learning_rate ValueType
	momentum_rate ValueType
	batch_size int
}

func NewStochasticGradientOptimizer(initial_learning_rate ValueType, momentum_rate ValueType, batch_size int) *StochasticGradientOptimizer{
	return &StochasticGradientOptimizer{initial_learning_rate:initial_learning_rate, momentum_rate:momentum_rate, batch_size:batch_size}
}

func (this *StochasticGradientOptimizer) evaluate(data Dataset, model Model) {
	// Pre-allocate the memory where the output and delta values will be stored for backprop calculations
	output_stack := NewVector(model.Network.GetStackSize())
	delta_stack := NewVector(model.Network.GetStackSize())
	var sum_train_loss ValueType = 0.0
	var sum_test_loss ValueType = 0.0
	var train_accuracy ValueType = 0.0
	var test_accuracy ValueType = 0.0
	for train_index := 0; train_index < data.GetNumTrainInstances(); train_index++ {
		input, target := data.GetTrainInstance(train_index)
		output := model.Network.CalculateOutput(input, output_stack)
		sum_train_loss += model.CalcLoss(target, output, delta_stack)
		best_index, _ := output.GetMax()
		if target.GetValue(best_index) == 1.0 {
			train_accuracy += ValueType(1.0)/ValueType(data.GetNumTrainInstances())
		}
	}
	for test_index := 0; test_index < data.GetNumTestInstances(); test_index++ {
		test_input, test_target := data.GetTestInstance(test_index)
		test_output := model.Network.CalculateOutput(test_input, output_stack)
		sum_test_loss += model.CalcLoss(test_target, test_output, delta_stack)
		best_index, _ := test_output.GetMax()
		if test_target.GetValue(best_index) == 1.0 {
			test_accuracy += ValueType(1.0)/ValueType(data.GetNumTestInstances())
		}
	}
	sum_train_loss /= ValueType(data.GetNumTrainInstances())
	sum_test_loss /= ValueType(data.GetNumTestInstances())
	regularization_term := model.Network.ApplyRegularization(0.0)
	fmt.Printf("Train accuracy %f, Train loss: %f, Test accuracy %f, Test loss: %f, regularization term: %f\n", train_accuracy, sum_train_loss, test_accuracy, sum_test_loss, regularization_term)
}

func (this *StochasticGradientOptimizer) run_epoch(learning_rate ValueType, data Dataset, model Model, num_threads int) []ValueType {
	// Allocate the two values used to store the components of the objective function: loss and regularization
	objective := make([]ValueType, 2)

	// The batch size should be relatively small -- at most the size of the entire dataset
	if data.GetNumTrainInstances() < this.batch_size {
		this.batch_size = data.GetNumTrainInstances()
	}
	step_size := learning_rate / ValueType(this.batch_size)

	// Pre-allocate the memory where the output and delta values will be stored for backprop calculations
	output_stack := NewVector(model.Network.GetStackSize())
	delta_stack := NewVector(model.Network.GetStackSize())

	// The number of iterations to run in this function should be calculated so that each epoch is one pass
	max_num_iterations := int(float32(data.GetNumTrainInstances())/float32(this.batch_size*num_threads))+1

	// Iterate according the the number of batches to be processed per thread
	for iteration_num := 0; iteration_num < max_num_iterations; iteration_num++ {
		// Initialize sum of losses for batch as 0
		var sum_loss ValueType = 0.0

		// This is the "stochastic" part: get a random sampling fom the dataset for eeach batch
		permutation := rand.Perm(data.GetNumTrainInstances())

		// Run through the batch
		for batch_index := 0; batch_index < this.batch_size; batch_index++ {
			// Nesterov's accelerated gradient -- apply momentum first, then correct
			model.Network.ApplyMomentum(this.momentum_rate)

			// Get the instance data from the sampled batch(input and target)
			input, target := data.GetTrainInstance(permutation[batch_index])

			// Calculate network outputs through forward pass recursively through layers
			output := model.Network.CalculateOutput(input, output_stack)

			// Calculate loss using model's defined loss function
			loss := model.CalcLoss(target, output, delta_stack)
            if loss > -1.0 * log(0.5) {
                //fmt.Printf("%v, %v, %v\n", output, target, delta_stack)
            }	
			sum_loss += loss

			// Backward pass, recursively calculate and apply gradients to each layer
			// Hogwild! style updates -- apply gradient immediately, skipping any locking for thread-safety!
			model.Network.EvaluateDelta(input, delta_stack, output_stack, step_size)
		}
		// The primary component of the objective function is the sum of loss calculated in the batch
		objective[0] = sum_loss / ValueType(this.batch_size)
		// The secondary component of the objection function is the regularization term, independent of batch
		objective[1] = model.Network.ApplyRegularization(learning_rate)
	}
	return objective
}

func (this *StochasticGradientOptimizer) Train(max_num_epochs int, data Dataset, model Model, evaluation_interval int) {
	// Stochastic gradient descent allows work to be distributed in parallel across multiple threads
	num_threads := runtime.NumCPU()
	runtime.GOMAXPROCS(num_threads)
	fmt.Printf("Running stochastic gradient descent in parallel on %d threads\n", num_threads)

	// Optimization steps are run iteratively, with the outer iteration block defined as epochs
	// Each epoch is, on average, one pass through the entire data set
	for epoch_num := 0; epoch_num < max_num_epochs; epoch_num++ {
		// Periodically do full evaluation across entire train and test set to monitor progress
		if evaluation_interval != 0 && epoch_num % evaluation_interval == 0 {
			fmt.Printf("Epoch %d ", epoch_num)
			this.evaluate(data, model)
		} else {
			fmt.Printf(".")
		}

		// Each gradient calculated is applied with a small step size that decays over time
		learning_rate := this.initial_learning_rate/(1 + (100.0*ValueType(epoch_num)/ValueType(max_num_epochs)))

		// Each optimization thread runs in parallel, then are synchronized at the end of the epoch
		var thread_list sync.WaitGroup
		thread_list.Add(num_threads)

		// Use go routines to distribute work across multiple threads
		for thread_index := 0; thread_index < num_threads; thread_index++ {
			go func(thread_index int) {
				defer thread_list.Done()
				this.run_epoch(learning_rate, data, model, num_threads)
				//fmt.Printf("Epoch %d, thread %d: objective %f %f\n", epoch_num, thread_index, objective[0], objective[1])
			}(thread_index)
		}

		// Wait until all threads have finished processing before looping to the next epoch
		thread_list.Wait()
	}
	// Save final output to filesystem
	model.Network.WriteToFile("mlp.mnist.json")
}
