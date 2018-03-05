package go_deep

import (
	"testing"
)

const float_tolerance ValueType = 0.0001

func assertEqual(tester *testing.T, actual interface{}, expected interface{}, test_description string) {
	if expected == actual {
		// Exact match: Pass assertion without error
    	return
	} else {
		// Mismatch greater than tolerance: Fail assertion with error message
		tester.Errorf("Error in " + test_description + ": expected %v, got %v\n", expected, actual)
	}
}

func assertEqualValue(tester *testing.T, actual ValueType, expected ValueType, test_description string) {
	if expected == actual {
		// Exact match: Pass assertion without error
    	return
	} else if abs(expected - actual) < float_tolerance  {
		// Near match, within numerical error: Pass assertion without error
    	return
	} else {
		// Mismatch greater than tolerance: Fail assertion with error message
		tester.Errorf("Error in " + test_description + ": expected %v, got %v\n", expected, actual)
	}
}

func assertEqualVector(tester *testing.T, actual Vector, expected Vector, test_description string) {
	for index := 0; index < expected.GetSize(); index++ {
		if abs(expected.GetValue(index) - actual.GetValue(index)) > float_tolerance  {
			// Mismatch greater than tolerance: Fail assertion with error message
			tester.Errorf("Error in " + test_description + ": expected %v, got %v\n", expected.GetValue(index), actual.GetValue(index))
		}
	}
}

func TestVectorMethods(tester *testing.T) {
	vector_1 := InitVector([]ValueType{1.0,2.0,3.0})
	assertEqual(tester, vector_1.GetSize(), 3, "vector_1.GetSize()")

	assertEqualValue(tester, vector_1.GetValue(0), ValueType(1.0), "vector_1.GetValue(0)")
	assertEqualValue(tester, vector_1.GetValue(1), ValueType(2.0), "vector_1.GetValue(1)")
	assertEqualValue(tester, vector_1.GetValue(2), ValueType(3.0), "vector_1.GetValue(2)")

	vector_1.SetValue(2, -4.0)
	assertEqualValue(tester, vector_1.GetValue(2), ValueType(-4.0), "vector_1.SetValue(2)")

	assertEqualVector(tester, vector_1.GetSlice(0,2), InitVector([]ValueType{1.0,2.0}), "vector_1.Slice(0,2)")
	assertEqualVector(tester, vector_1.GetSlice(2,-1), InitVector([]ValueType{-4.0}), "vector_1.Slice(2,-1)")

	vector_1.Scale(0.1)
	assertEqualVector(tester, vector_1, InitVector([]ValueType{0.1,0.2,-0.4}), "vector_1.Scale(0)")

	max_index, max_value := vector_1.GetMax()
	assertEqual(tester, max_index, 1, "vector_1.GetMax() index")
	assertEqualValue(tester, max_value, 0.2, "vector_1.GetMax() value")

	vector_2 := CopyVector(vector_1)
	assertEqualVector(tester, vector_2, vector_1, "CopyVector(vector_1)")

	vector_2.AddVector(InitVector([]ValueType{0.3,0.3,0.3}))
	assertEqualVector(tester, vector_2, InitVector([]ValueType{0.4,0.5,-0.1}), "vector_2.AddVector([0.3,0.3,0.3])")

	vector_3 := NewVector(3)
	vector_3.Copy(vector_2)
	assertEqualVector(tester, vector_3, vector_2, "Copy(vector_2)")

	vector_3.AddScaledVector(2.0, InitVector([]ValueType{0.1,0.0,-0.1}))
	assertEqualVector(tester, vector_3, InitVector([]ValueType{0.6,0.5,-0.3}), "vector_3.AddScaledVector(1.0,[0.1,0.0,0.1])")

	assertEqualValue(tester, vector_1.Dot(vector_2), ValueType(0.18), "vector_1.Dot(vector_2)")

	matrix := InitMatrix(2, 3, []ValueType{0.4,0.5,-0.1,-0.4,-0.5,0.1})
	product := NewVector(2)
	product.ProductOf(matrix, vector_1)
	assertEqual(tester, product.GetSize(), 2, "ProductOf(matrix, vector_1).GetSize()")
	assertEqualValue(tester, product.GetValue(0), ValueType(0.18), "ProductOf(matrix, vector_1)[0]")
	assertEqualValue(tester, product.GetValue(1), ValueType(-0.18), "ProductOf(matrix, vector_1)[1]")

	product_2 := NewVector(3)
	product_2.ProductOfTransposed(matrix, InitVector([]ValueType{-0.7,0.8}))
	assertEqualValue(tester, product_2.GetValue(0), ValueType(-0.6), "ProductOfTransposed(matrix, product)[0]")
	assertEqualValue(tester, product_2.GetValue(1), ValueType(-0.75), "ProductOfTransposed(matrix, product)[1]")
	assertEqualValue(tester, product_2.GetValue(2), ValueType(0.15), "ProductOfTransposed(matrix, product)[2]")

	matrix2 := InitMatrix(10, 10, []ValueType{ -0.073945, 0.26749, 0.28353, 0.19393, 0.068799, 0.25415, 0.23667, 0.22063, -0.034928, 0.1365, 0.41672, 0.03999, 0.23332, 0.10997, 0.019558, 0.30978, 0.20518, 0.22432, 0.1241, 0.23552, 0.29989, 0.089282, 0.082106, 0.139, 0.17547, 0.31573, -0.005835, 0.17583, 0.35354, 0.042792, 0.088048, 0.16035, 0.29209, 0.24592, 0.0040886, 0.066441, 0.12568, 0.26227, 0.028461, 0.43652, 0.35874, 0.31965, 0.24518, 0.044059, 0.099539, 0.13598, 0.046061, 0.11847, 0.42444, -0.10386, -0.090587, 0.24651, 0.27956, 0.27954, 0.23431, -0.16734, -0.15794, 0.22437, 0.075841, 0.28638, 0.10215, -0.048956, 0.017469, 0.13935, 0.30517, 0.24216, -0.12344, 0.34162, 0.31976, 0.2322, 0.22513, 0.049389, 0.42054, 0.18733, 0.28791, -0.028654, 0.37239, 0.12216, -0.058582, 0.30728, -0.14979, 0.44727, 0.075323, 0.23266, 0.2346, 0.41473, -0.016568, 0.19187, 0.022057, 0.16431, 0.14076, 0.39478, 0.17225, 0.29973, -0.089057, 0.22965, -0.098465, 0.27196, 0.29604, -0.081912 })
	vector_4 := InitVector([]ValueType{0.4479, 0.4552, 0.4840, 0.5791, 0.5118, 0.4464, 0.4452, 0.4976, 0.5263, 0.4504})
	product_3 := NewVector(10)
	product_3.ProductOf(matrix2, vector_4)
	assertEqualVector(tester, product_3, InitVector([]ValueType{0.74509, 0.90412, 0.81618, 0.82601, 0.81807, 0.62428, 0.76481, 0.90405, 0.78661, 0.76703}), "product")
}
