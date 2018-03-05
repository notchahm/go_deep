package go_deep

type ImageClassificationDataset struct {
	inputs []Vector
	labels []Vector
	input_dimensions []int
	test_offset int
}

func InitImageClassificationDataset(init_size int) *ImageClassificationDataset {
	this := ImageClassificationDataset{}
	this.inputs = make([]Vector, 0, init_size)
	this.labels = make([]Vector, 0, init_size)
	return &this
}

func (this *ImageClassificationDataset) GetNumTrainInstances() int {
    return this.test_offset
}

func (this *ImageClassificationDataset) GetNumTestInstances() int {
    return len(this.inputs) - this.test_offset
}

func (this *ImageClassificationDataset) GetNumFeatures() int {
	if len(this.inputs) > 0 {
		return this.inputs[0].GetSize()
	} else {
		return 0
	}
}

func (this *ImageClassificationDataset) GetNumClasses() int {
	if len(this.labels) > 0 {
		return this.labels[0].GetSize()
	} else {
		return 0
	}
}

func (this *ImageClassificationDataset) GetInputDimensions() []int {
	return this.input_dimensions
}

func (this *ImageClassificationDataset) SetInputDimensions(input_dimensions []int) {
	this.input_dimensions = input_dimensions
}

func (this *ImageClassificationDataset) GetTrainInstance(index int) (Vector, Vector) {
	return this.inputs[index], this.labels[index]
}

func (this *ImageClassificationDataset) GetTestInstance(index int) (Vector, Vector) {
	return this.inputs[this.test_offset+index], this.labels[this.test_offset+index]
}

func (this *ImageClassificationDataset) GetInput(index int) Vector {
	return this.inputs[index]
}

func (this *ImageClassificationDataset) GetLabel(index int) Vector {
	return this.labels[index]
}

func (this *ImageClassificationDataset) AddInput(new_input Vector) {
	num_inputs := len(this.inputs)
	if num_inputs + 1 > cap(this.inputs) {
		resized := make([]Vector, (num_inputs * 2))
		copy(resized, this.inputs)
		this.inputs = resized
	}
	this.inputs = this.inputs[0:num_inputs+1]
	this.inputs[num_inputs] = new_input
}

func (this *ImageClassificationDataset) AddLabel(new_label Vector) {
	this.labels = append(this.labels, new_label)
}

func (this *ImageClassificationDataset) SetTestSplit(test_fraction float64)() {
    this.test_offset = int(float64(len(this.inputs)) * (1.0-test_fraction))
}
