package main

import "net/http"
import "compress/gzip"
import "encoding/binary"
import "fmt"
import "github.com/notchahm/go_deep"

type MnistImageHeader struct {
	MagicNumber uint32
	NumImages uint32
	NumRows uint32
	NumColumns uint32
}
type MnistLabelHeader struct {
	MagicNumber uint32
	NumLabels uint32
}

func load_images_from_url(url string, dataset go_deep.Dataset) {
	response, error := http.Get(url)
	if error != nil {
		fmt.Printf("Error fetching %s\n", url)
		return
	}
	defer response.Body.Close()
	gunzipper, error := gzip.NewReader(response.Body)
	if error != nil {
		fmt.Printf("Error unzipping %s\n", url)
		return
	}
	defer gunzipper.Close()
	header := MnistImageHeader{}
	binary.Read(gunzipper, binary.BigEndian, &header)
	header.NumImages = 10000
	if header.MagicNumber == 2051 && header.NumRows == 28 && header.NumColumns == 28 {
		for image_index := 0; image_index < int(header.NumImages); image_index++ {
			new_image := make([]go_deep.ValueType, header.NumRows * header.NumColumns)
			pixel_bytes := make([]byte, header.NumRows * header.NumColumns)
			gunzipper.Read(pixel_bytes)
			for pixel_index := 0; pixel_index < len(pixel_bytes); pixel_index++ {
				new_image[pixel_index] = go_deep.ValueType(pixel_bytes[pixel_index])/go_deep.ValueType(255)
			}
			dataset.AddInput(go_deep.InitVector(new_image))
		}
	} else {
		fmt.Printf("Unexpected values in header %v\n", header)
	}
}

func load_labels_from_url(url string, dataset go_deep.Dataset) {
	response, error := http.Get(url)
	if error != nil {
		fmt.Printf("Error fetching %s\n", url)
		return
	}
	defer response.Body.Close()
	gunzipper, error := gzip.NewReader(response.Body)
	if error != nil {
		fmt.Printf("Error unzipping %s\n", url)
		return
	}
	defer gunzipper.Close()
	header := MnistLabelHeader{}
	binary.Read(gunzipper, binary.BigEndian, &header)
	header.NumLabels = 10000
	if header.MagicNumber == 2049 {
		label_bytes := make([]byte, header.NumLabels)
		gunzipper.Read(label_bytes)
		for label_index := 0; label_index < int(header.NumLabels); label_index++ {
			new_label := make([]go_deep.ValueType, 10)
			new_label[label_bytes[label_index]] = go_deep.ValueType(1.0)
			dataset.AddLabel(go_deep.InitVector(new_label))
		}
	} else {
		fmt.Printf("Unexpected values in header %v\n", header)
	}
}

func calc_validation_accuracy(test_dataset go_deep.Dataset, model go_deep.Model) float64 {
	num_correct := 0
	output_stack := go_deep.NewVector(model.Network.GetStackSize())
	for test_index := 0; test_index < test_dataset.GetNumTestInstances(); test_index++ {
		input, label := test_dataset.GetTestInstance(test_index)
		output := model.Network.CalculateOutput(input, output_stack)
		best_index := 0
		for class_index := 1; class_index < output.GetSize(); class_index++ {
			if output.GetValue(class_index) > output.GetValue(best_index) {
				best_index = class_index
			}
		}
		if label.GetValue(best_index) == 1.0 {
//			fmt.Printf("%v %v %d\n", output, label, best_index)
			num_correct++
		}
	}
	return float64(num_correct)/float64(test_dataset.GetNumTestInstances())
}

func main() {
	fmt.Printf("Test MNIST\n")
	regularization_rate := go_deep.ValueType(0.001)
	initial_learning_rate := go_deep.ValueType(0.1)
	momentum := go_deep.ValueType(0.99)
	batch_size := 100
	evaluation_interval := 10
	train_dataset := go_deep.InitImageClassificationDataset(60000)
	validation_dataset := go_deep.InitImageClassificationDataset(10000)
	fmt.Printf("fetching dataset\n")
	load_images_from_url("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", train_dataset)
	load_labels_from_url("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", train_dataset)
	load_images_from_url("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", validation_dataset)
	load_labels_from_url("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", validation_dataset)
	train_dataset.SetTestSplit(0.1)
	validation_dataset.SetTestSplit(1.0)
	optimizer := go_deep.NewStochasticGradientOptimizer(initial_learning_rate, momentum, batch_size)

	input_layer := go_deep.NewInputLayer(train_dataset.GetNumFeatures())					// 28 x 28 x 1
	conv_layer_1 := go_deep.NewConvLayer(input_layer, 8, 8, 64, 1, regularization_rate) 	// 21 x 21 x 64
	conv_layer_2 := go_deep.NewConvLayer(conv_layer_1, 5, 5, 128, 2, regularization_rate)	// 9 x 9 x 128
	conv_layer_3 := go_deep.NewConvLayer(conv_layer_2, 3, 3, 256, 2, regularization_rate)	// 4 x 4 x 256
	fully_connected_layer := go_deep.NewLayer(conv_layer_3, 100, regularization_rate)
    output_layer := go_deep.NewClassificationOutputLayer( fully_connected_layer, train_dataset.GetNumClasses(), regularization_rate )
	//model := go_deep.Model{Network:go_deep.NewMultilayerNetwork(train_dataset.GetNumFeatures(), [] int { 300, 100 }, train_dataset.GetNumClasses(), 0.0001), CalcLoss: go_deep.CrossEntropyLoss }
	model := go_deep.Model{Network:output_layer, CalcLoss: go_deep.CrossEntropyLoss }
	max_num_iterations := 1000
	optimizer.Train(max_num_iterations, train_dataset, model, evaluation_interval)
	validation_accuracy := calc_validation_accuracy(validation_dataset, model)
	fmt.Printf("Validation accuracy: %f\n", validation_accuracy)
}

