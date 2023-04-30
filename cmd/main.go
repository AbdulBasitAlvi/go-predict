package main

import (
	"math"
)

// Main function to test RNN
func main() {
	// Define the range of input values
	start := 0.0
	end := 101 * math.Pi
	step := math.Pi / 2
	totalRange := int((end - start) / step)

	// fmt.Printf("Total range: %d\n", totalRange)
	// fmt.Printf("Range of training data: %d\n", rangeOfTrainingData)

	// Create the input and output vectors
	inputs := make([]float64, totalRange)
	actualOutputs := make([]float64, totalRange)
	for i := range inputs {
		inputs[i] = start + float64(i)*step
		actualOutputs[i] = math.Sin(inputs[i])
	}

	// fmt.Print(inputs)
	// fmt.Printf("\n")
	// fmt.Print(actualOutputs)
	// fmt.Printf("\n")
	// fmt.Print(RNN.Why)

	start = end
	end = end + 2*math.Pi
	totalRange = int((end - start) / step)

	// Create the input and output vectors
	xNew := make([]float64, totalRange)
	yActual := make([]float64, totalRange)
	for i := range xNew {
		xNew[i] = start + float64(i)*step
		yActual[i] = math.Sin(inputs[i])
	}

}
