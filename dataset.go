package go_deep

type Dataset interface {
	GetNumTrainInstances() int
	GetNumTestInstances() int
	GetNumFeatures() int
	GetNumClasses() int
	SetInputDimensions([] int)
	GetInputDimensions() [] int
	GetInput(index int) Vector
	GetLabel(index int) Vector
	AddInput(new_input Vector)
	AddLabel(new_label Vector)
	GetTrainInstance(index int) (Vector, Vector)
	GetTestInstance(index int) (Vector, Vector)
	SetTestSplit(test_fraction float64)()
}

