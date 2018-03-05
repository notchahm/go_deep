package go_deep

import "math"
import "math/rand"
import "strconv"

//type ValueType float32
type ValueType float64

//var source Rand = math.rand.NewSource(314)	// seed with deterministic value
//var generator Rand = math.rand.New(source)

const SmallestNonzeroValueType = math.SmallestNonzeroFloat32

func random(min ValueType, max ValueType) ValueType {
	var value_range float64 = float64(max) - float64(min)
	return min + ValueType(rand.Float64() * value_range)
}

func exp(value ValueType) ValueType {
	return ValueType(math.Exp(float64(value)))
}

func log(value ValueType) ValueType {
	return ValueType(math.Log(float64(value)))
}

func sqrt(value ValueType) ValueType {
	return ValueType(math.Sqrt(float64(value)))
}

func tanh(value ValueType) ValueType {
	return ValueType(math.Tanh(float64(value)))
}

func abs(value ValueType) ValueType {
	return ValueType(math.Abs(float64(value)))
}

func to_string(value ValueType) string {
	return strconv.FormatFloat(float64(value), 'g', 5, 64)
}
