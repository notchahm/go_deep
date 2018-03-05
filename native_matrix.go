package go_deep

//import "fmt"

type NativeMatrix struct {
	values []ValueType
	num_rows int
	num_columns int
}

func new_native_matrix(num_rows int, num_columns int) *NativeMatrix {
	return &NativeMatrix{num_rows: num_rows, num_columns: num_columns, values: make([]ValueType, num_rows*num_columns)}
}

func init_native_matrix(num_rows int, num_columns int, initial_values []ValueType) *NativeMatrix {
	return &NativeMatrix{num_rows: num_rows, num_columns: num_columns, values: initial_values}
}

func (this *NativeMatrix) GetNumRows() int {
	return this.num_rows
}

func (this *NativeMatrix) GetNumColumns() int {
	return this.num_columns
}

func (this *NativeMatrix) GetDimensions() (int, int) {
	return this.num_columns, this.num_rows
}

func (this *NativeMatrix) GetValueAt(row_index int, column_index int) ValueType {
	return this.values[row_index * this.num_columns + column_index]
}

func (this *NativeMatrix) SetValueAt(row_index int, column_index int, value ValueType) {
	this.values[row_index * this.num_columns + column_index] = value
}

func (this *NativeMatrix) GetRow(row_index int) Vector {
	//Skip error checking for speed
	//if (row_index < 0 || row_index >= this.num_rows) {
	//	return Vector{}, errors.New("NativeMatrix.get_row() invalid row index")
	//}
	offset := row_index * this.num_columns
	return &NativeVector{size: this.num_columns, values: this.values[offset:offset+this.num_columns]}
}

func (this *NativeMatrix) Randomize(value_range ValueType) {
	for index := 0; index < len(this.values); index++ {
		this.values[index] = random(-1.0 * value_range, value_range)
	}
}

func (this *NativeMatrix) Scale(scalar ValueType) {
	for index := 0; index < len(this.values); index++ {
		//orig := this.values[index]
		this.values[index] *= scalar
		//fmt.Printf("%f * %f-> %f\n", orig, scalar, this.values[index])
	}
}

func (this *NativeMatrix) GetNorm() ValueType {
    var sum ValueType = 0.0
	for index := 0; index < len(this.values); index++ {
		sum += this.values[index] * this.values[index]
	}
    return sqrt(sum)
}

// Adds the outer product of given parameters left and right, multiplied by given scalar (i.e. a rank-1 update A := scalar*left*right' + A)
func (this *NativeMatrix) AddScaledCrossProduct(scalar ValueType, left Vector, right Vector) {
	matrix_offset := 0
	for row_index := 0; row_index < this.num_rows; row_index++ {
		for col_index := 0; col_index < this.num_columns; col_index++ {
			this.values[matrix_offset] += scalar * left.GetValue(row_index) * right.GetValue(col_index)
			matrix_offset++
		}
	}
}

func (this *NativeMatrix) AddMatrix(other Matrix) {
	matrix_offset := 0
	for row_index := 0; row_index < this.num_rows; row_index++ {
		for col_index := 0; col_index < this.num_columns; col_index++ {
			this.values[matrix_offset] += other.GetValueAt(row_index, col_index)
			matrix_offset++
		}
	}
}

func (this *NativeMatrix) GetAsString() string {
	string_representation := "["
	for row_index := 0; row_index < this.num_rows; row_index++ {
		for col_index := 0; col_index < this.num_columns; col_index++ {
			string_representation += to_string(this.GetValueAt(row_index, col_index))
			if row_index < this.num_rows || col_index < this.num_columns {
				string_representation += ", "
			}
		}
		string_representation += "\n"
	}
	string_representation += "]"
	return string_representation
}
