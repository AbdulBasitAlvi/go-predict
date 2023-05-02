package main

import (
	"go-predict/pkg/rnn"
	"gonum.org/v1/gonum/mat"
	"math"
)

func main() {
	// Define the range of input values
	start := 0.0
	end := 101 * math.Pi
	step := math.Pi / 2
	totalRange := int((end - start) / step)

	// Create the input and output matrices
	inputs := mat.NewDense(totalRange, 1, nil)
	actualOutputs := mat.NewDense(totalRange, 1, nil)
	for i := 0; i < totalRange; i++ {
		inputs.Set(i, 0, start+float64(i)*step)
		actualOutputs.Set(i, 0, math.Sin(inputs.At(i, 0)))
	}

	// Define the architecture of the RNN
	numInputs := 1
	numOutputs := 1
	numHiddenLayers := 3

	// Create a new RNN builder and set its properties
	builder := rnn.Builder{}
	builder.
		SetNumInputs(numInputs).
		SetNumOutputs(numOutputs).
		SetNumHiddenLayers(numHiddenLayers).
		SetWxh(numInputs, numHiddenLayers).
		SetWhh(numHiddenLayers).
		SetWhy(numHiddenLayers, numOutputs)

	// Build the RNN
	net := builder.Build()

	// Train the RNN
	numEpochs := 1000
	learningRate := 0.1

	net.TrainRNN(numEpochs, inputs, actualOutputs, learningRate)
}
