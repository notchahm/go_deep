package go_deep

import (
	"fmt"
	"os"
	"testing"
)

type example_test interface {
	run_example() int
	get_name() string
}

func TestExamples(tester *testing.T) {
	fmt.Printf("%s\n", os.Args[0])
	return

	test_algorithms := []example_test{ &Perceptron{}, &SupportVectorMachine{}, &LogisticRegression{}, &MultilayerPerceptron{} }
	for _, algorithm := range test_algorithms {
		algorithm_name := algorithm.get_name()
		fmt.Printf("%s:\n", algorithm_name)
		num_errors := algorithm.run_example()
		fmt.Printf("%d errors\n\n", num_errors)
		if num_errors > 0 {
			tester.Errorf("Error running %s, expected 0 errors\n", algorithm_name)
		}
	}
}

