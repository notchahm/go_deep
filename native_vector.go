package go_deep

type NativeVector struct {
	values []ValueType
	size int
}

func new_native_vector(size int) *NativeVector {
    return &NativeVector{size: size, values: make([]ValueType, size)}
}

func init_native_vector(initial_values []ValueType) *NativeVector {
    return &NativeVector{size: len(initial_values), values: initial_values}
}

func copy_native_vector(other Vector) *NativeVector {
	copied := new_native_vector(other.GetSize())
	for index := 0; index < copied.size; index++ {
		copied.values[index] = other.GetValue(index)
    }
	return copied
}

func (this *NativeVector) GetSize() int {
	return this.size
}

func (this *NativeVector) GetValue(index int) ValueType {
	return this.values[index]
}

func (this *NativeVector) SetValue(index int, value ValueType) {
	this.values[index] = value
}

func (this *NativeVector) GetSlice(start_index int, end_index int) Vector {
	if start_index < 0 {
		start_index = 0
	}
	if end_index > cap(this.values) || end_index < 0 {
//		fmt.Printf("end_index %d > size %d\n", end_index, this.size)
		end_index = cap(this.values)
	}
	return &NativeVector{size: end_index-start_index, values: this.values[start_index:end_index] }
}

func (this *NativeVector) Dot(other Vector) ValueType {
	var sum ValueType = 0
	for index := 0; index < this.size; index++ {
		sum += this.values[index] * other.GetValue(index)
    }
	return sum
}

func (this *NativeVector) Copy(other Vector) {
	for index := 0; index < this.size; index++ {
		this.values[index] = other.GetValue(index)
    }
}

func (this *NativeVector) AddVector(other Vector) {
	for index := 0; index < this.size; index++ {
		this.values[index] += other.GetValue(index)
    }
}

func (this *NativeVector) AddScaledVector(scalar ValueType, other Vector) {
	for index := 0; index < this.size; index++ {
		this.values[index] += scalar * other.GetValue(index)
    }
}

func (this *NativeVector) Scale(scalar ValueType) {
	for index := 0; index < this.size; index++ {
		this.values[index] *= scalar
    }
}

func (this *NativeVector) GetMax() (int, ValueType) {
	max_index := 0
	max_value := this.values[0]
    for index := 1; index < this.size; index++ {
		if this.values[index] > max_value {
			max_value = this.values[index]
			max_index = index
		}
	}
	return max_index, max_value
}

func (this *NativeVector) GetSum() ValueType {
	var sum ValueType = 0
	for index := 0; index < this.size; index++ {
		sum += this.values[index]
    }
	return sum
}

func (this *NativeVector) GetNorm() ValueType {
	var sum ValueType = this.Dot(this)
	return sqrt(sum)
}

func (this *NativeVector) GetAsString() string {
    string_representation := ""
	for index := 0; index < this.size; index++ {
        string_representation += to_string(this.GetValue(index)) + " "
    }
    return string_representation
}

func (this *NativeVector) SetZero() {
	for index := 0; index < this.size; index++ {
		this.values[index] = 0.0
    }
}

func (this *NativeVector) ProductOf(left Matrix, right Vector) {
	for row_index := 0; row_index < left.GetNumRows(); row_index++ {
		this.values[row_index] = left.GetRow(row_index).Dot(right)
	}
}

func (this *NativeVector) ProductOfTransposed(left Matrix, right Vector) {
	for col_index := 0; col_index < left.GetNumColumns(); col_index++ {
		this.values[col_index] = 0
		for row_index := 0; row_index < left.GetNumRows(); row_index++ {
			this.values[col_index] += left.GetValueAt(row_index, col_index) * right.GetValue(row_index)
		}
	}
}

func (this *NativeVector) HadamardProductOf(left Vector, right Vector, add_flag bool) {
	for index := 0; index < this.size; index++ {
		if add_flag {
			this.values[index] += left.GetValue(index) * right.GetValue(index)
		} else {
			this.values[index] = left.GetValue(index) * right.GetValue(index)
		}
    }
}
