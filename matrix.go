package go_deep

type Matrix interface {
	GetNumRows() int
	GetNumColumns() int
	GetDimensions() (int, int)
	GetValueAt(row_index int, column_index int) ValueType
	SetValueAt(row_index int, column_index int, value ValueType)
	GetRow(row_index int) Vector
	Randomize(value_range ValueType)
	Scale(scalar ValueType)
	GetNorm() ValueType
	AddScaledCrossProduct(scalar ValueType, left Vector, right Vector)
	AddMatrix(other Matrix)
	GetAsString() string
}

func NewMatrix(num_rows int, num_columns int) Matrix {
	return new_native_matrix(num_rows, num_columns)
}

func InitMatrix(num_rows int, num_columns int, initial_values []ValueType) Matrix {
	return init_native_matrix(num_rows, num_columns, initial_values)
}

