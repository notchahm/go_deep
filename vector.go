package go_deep

type Vector interface {
	GetSize() int
	GetValue(index int) ValueType
	SetValue(index int, value ValueType)
	GetSlice(start_index int, end_index int) Vector
	Scale(scalar ValueType)
	GetMax() (int, ValueType)
	GetNorm() ValueType
	GetAsString() string
	SetZero()
	Dot(other Vector) ValueType
	Copy(other Vector)
	AddVector(other Vector)
	AddScaledVector(scalar ValueType, other Vector)
	ProductOf(left Matrix, right Vector)
	ProductOfTransposed(left Matrix, right Vector)
	HadamardProductOf(left Vector, right Vector, add_flag bool)
}

func NewVector(size int) Vector {
	return new_native_vector(size)
}

func InitVector(initial_values []ValueType) Vector {
    return init_native_vector(initial_values)
}

func CopyVector(other Vector) Vector {
    return copy_native_vector(other)
}
