package rnn

import (
	"../matrices"
	"fmt"
	"math/rand"
	"time"
)

// RNN struct
type RNN struct {
	numInputs       int
	numOutputs      int
	numHiddenLayers int
	Wxh             weightMatrix
	Whh             weightMatrix
	Why             weightMatrix
	hiddenLayer     [][]float64
}

// Struct for the weight matrix in an RNN
type weightMatrix struct {
	weight [][]float64
}

// RNN Builder builds the RNN
type Builder struct {
	neuralNet RNN
}

// Sets the number of inputs in the RNN
func (builder *Builder) SetNumInputs(numInputs int) *Builder {
	builder.neuralNet.numInputs = numInputs
	return builder
}

// Sets the number of outputs in the RNN
func (builder *Builder) SetNumOutputs(numOutputs int) *Builder {
	builder.neuralNet.numOutputs = numOutputs
	return builder
}

// Sets the number of hidden layers in the RNN
func (builder *Builder) SetNumHiddenLayers(numHiddenLayers int) *Builder {
	builder.neuralNet.numHiddenLayers = numHiddenLayers
	return builder
}

/* Sets the weight matrix between the inputs and the hidden layers,
creates a two-dimensional matrix of size numHiddenLayers by numInputs
and initializes all elements to a random value
*/
func (builder *Builder) SetWxh(numInputs, numHiddenLayers int) *Builder {
	rand.Seed(time.Now().UnixNano())

	matrix := make([][]float64, numInputs)
	for i := range matrix {
		matrix[i] = make([]float64, numHiddenLayers)
	}

	// Initialize the matrix elements
	for i := 0; i < numInputs; i++ {
		for j := 0; j < numHiddenLayers; j++ {
			matrix[i][j] = rand.Float64()
		}
	}

	builder.neuralNet.Wxh = weightMatrix{weight: matrix}
	return builder
}

/* Sets the weight matrix between the hidden layers,
creates a two-dimensional matrix of size numHiddenLayers by numHiddenLayers
and initializes all elements to a random value
*/
func (builder *Builder) SetWhh(numHiddenLayers int) *Builder {
	rand.Seed(time.Now().UnixNano())

	matrix := make([][]float64, numHiddenLayers)
	for i := range matrix {
		matrix[i] = make([]float64, numHiddenLayers)
	}

	// Initialize the matrix elements, start with index 1 since no connection exists between h0 and h0
	for i := 1; i < numHiddenLayers; i++ {
		for j := 1; j < numHiddenLayers; j++ {
			matrix[i][j] = rand.Float64()
		}
	}

	builder.neuralNet.Whh = weightMatrix{weight: matrix}
	return builder
}

/* Sets the weight matrix between the hidden layers and output neurons,
creates a two-dimensional matrix of size numHiddenLayers by numOutputs
and initializes all elements to a random value
*/
func (builder *Builder) SetWhy(numHiddenLayers, numOutputs int) *Builder {
	rand.Seed(time.Now().UnixNano())

	matrix := make([][]float64, numHiddenLayers)
	for i := range matrix {
		matrix[i] = make([]float64, numOutputs)
	}

	// Initialize the matrix elements
	for i := 0; i < numHiddenLayers; i++ {
		for j := 0; j < numOutputs; j++ {
			matrix[i][j] = rand.Float64()
		}
	}

	builder.neuralNet.Why = weightMatrix{weight: matrix}
	return builder
}

// Builder function to build the RNN
func (builder *Builder) Build() RNN {
	return builder.neuralNet
}

func NewRNNBuilder(numInputs int, numOutputs int, numHiddenLayers int) *Builder {
	builder := Builder{
		neuralNet: RNN{
			numInputs:       numInputs,
			numOutputs:      numOutputs,
			numHiddenLayers: numHiddenLayers,
		},
	}
	return &builder
}

/*
	forwardPass performs the forward pass of the RNN
	h(t) = f( h(t-1), x(t) )
	h(t) = tanh ( Whh.h(t-1) + Wxh.x(t) )
	y(t) = Why.h(t)

	The inputs are the input vector x(t)
	The outputs are therefore are h(t) and y(t)
*/
func (r *RNN) forwardRNN(x [][]float64) ([][]float64, error) {
	//TODO Optimise using Go Routines

	//Initialise hidden layer and output matrices
	var output [][]float64
	for i := 0; i < r.numHiddenLayers; i++ {
		r.hiddenLayer[0][i] = 0
	}

	//Calculate h(t) = tanh ( Whh.h(t-1) + Wxh.x(t) ) & y(t) = Why.h(t)
	for i := 1; i < len(x); i++ {
		r.hiddenLayer[i] = matrices.TanhPrimeMatrix1D(
			matrices.AddMatrices(matrices.MultiplyMatrices(r.Whh.weight, matrices.ShiftMatrix(r.hiddenLayer, i)),
				matrices.MultiplyMatrices(r.Wxh.weight, x[i])))

		output[i] = matrices.MultiplyMatrices(r.Why * r.hiddenLayer[i])
	}
	return output, nil
}

/* 	This function calculates the derivative of mean squared error (MSE) with respect to predicted values
using the formula 2/N Σ (y(t) - ŷ(t))
*/
func calculateMSEGrad(predictions [][]float64, targets [][]float64) float64 {
	var errorSum float64

	// Assuming predictions and targets have the same dimensions
	rowCount := len(predictions)
	colCount := len(predictions[0])

	// Calculate the error sum
	for i := 0; i < rowCount; i++ {
		for j := 0; j < colCount; j++ {
			errorSum += predictions[i][j] - targets[i][j]
		}
	}

	// Divide the error sum by the total number of elements
	N := float64(rowCount * colCount)
	mseGrad := (2 / N) * errorSum

	return mseGrad
}

/* 	This function calculates the derivative of error with respect to weight matrix Why
using the formula: 2/N Σ (y(t) - ŷ(t)) * h(t)
The resulting gradient is a two-dimensional matrix of size  len(h) x len(predictions)
where each element represents the partial derivative of the error with respect to a
specific weight in the Why matrix.
*/
func (r *RNN) calculateWhyGrad(predictions [][]float64, targets [][]float64) [][]float64 {
	mseGrad := calculateMSEGrad(predictions, targets)

	whyGrad := matrices.MultiplyScalar(r.hiddenLayer, mseGrad)
	return whyGrad
}

/* 	The updateWhy function updates the weight matrix Why by subtracting the product of the gradient and
the learning rate from each weight. The updated weights are returned as a new two-dimensional matrix.
*/
func (r *RNN) updateWhy(whyGrad [][]float64, learningRate float64) {
	r.Why.weight = matrices.SubtractMatrices(r.Why.weight, matrices.MultiplyScalar(whyGrad, learningRate))
}

// This function calculates the value of Dh(t-x)/DWhh at time shift x
func (r *RNN) calculateDhtDWhh(x int) {

}

// This function calculates the value of Dh(t-x)/Dh(t-x-1) at time shift x
func (r *RNN) calculateDhtDht(x int) {

}

// This function calculates the value of Dh(t-x)/DWhx at time shift x
func (r *RNN) calculateDhtDWhx(x int) {

}

/* 	This function calculates the derivative of error with respect to weight matrix Whh
using the formula: dE/dy(T) * dy(T)/dh(T) * Σ dh(t)/dh(k) * dh(k)/dWhh
*/
func (r *RNN) calculateWhhGrad(predictions, targets, x [][]float64) [][]float64 {

	// Calculate the derivative of the error with respect to y dE/dy
	dEdy := calculateMSEGrad(predictions, targets)

	// Calculate the derivative of y with respect to h
	dydh := r.Why.weight

	/* Make sum and product matrices and initialise them
	prod should have same dimensions as Whh
	sum should have same dimensions as hidden layer
	*/
	sum := matrices.InitialiseMatrix(len(r.hiddenLayer[0]), len(r.hiddenLayer), 0)

	/* Calculate the term Σ dh(t)/dh(k) * dh(k)/dWhh where summation goes from k=1 to t
	and dh(t)/dh(k) = Π dh(i) / dh(i-1) where this goes from i=k+1 to t
	dh(t)/dWhh = sech(Whh * h(t-1) + Wxh x(t)) * h(t-1)
	dh(t)/dh(t-1) = sech(Whh * h(t-1) + Wxh x(t)) * Whh
	*/
	for t := 0; t < len(x); t++ {
		// Initialise prod here to 1
		prod := matrices.InitialiseMatrix(len(r.Whh.weight[0]), len(r.Whh.weight), 1)
		if t > 0 {
			for i := t; i > 0; i-- {
				fmt.Print("x ")
				prod *= r.calculateDhtDht(i - 1)
			}
		}
		sum += r.calculateDhtDWhh(t) * prod
		fmt.Print("+ ")
	}

	result := matrices.MultiplyScalar(matrices.MultiplyMatrices(dydh, sum), dEdy)
	return result
}

/* 	The updateWhh function updates the weight matrix Whh by subtracting the product of the gradient and
the learning rate from each weight.
*/
func (r *RNN) updateWhh(whhGrad [][]float64, learningRate float64) {
	r.Whh.weight = matrices.SubtractMatrices(r.Whh.weight, matrices.MultiplyScalar(whhGrad, learningRate))
}

/* 	This function calculates the derivative of error with respect to weight matrix Whh
using the formula: dE/dy(T) * dy(T)/dh(T) * Σ dh(t)/dh(k) * dh(k)/dWhh
*/
func (r *RNN) calculateWxhGrad(predictions, targets, x [][]float64) [][]float64 {

	// Calculate the derivative of the error with respect to y dE/dy
	dEdy := calculateMSEGrad(predictions, targets)

	// Calculate the derivative of y with respect to h
	dydh := r.Why.weight

	/* Make sum and product matrices and initialise them
	prod should have same dimensions as Whh
	sum should have same dimensions as hidden layer
	*/
	sum := matrices.InitialiseMatrix(len(r.hiddenLayer[0]), len(r.hiddenLayer), 0)
	prod := matrices.InitialiseMatrix(len(r.Whh.weight[0]), len(r.Whh.weight), 1)

	/* Calculate the term Σ dh(t)/dh(k) * dh(k)/dWhh where summation goes from k=1 to t
	and dh(t)/dh(k) = Π dh(i) / dh(i-1) where this goes from i=k+1 to t
	dh(t)/dWhx = sech(Whh * h(t-1) + Wxh x(t)) * x(t)
	dh(t)/dh(t-1) = sech(Whh * h(t-1) + Wxh x(t)) * Whh
	*/
	for k := 1; k < len(x); k++ {
		for i := k + 1; i < len(x); i++ {
			prod = matrices.TanhPrimeMatrix(
				matrices.MultiplyMatrices(
					matrices.AddMatrices(
						matrices.MultiplyMatrices(r.Whh.weight, matrices.ShiftMatrix(r.hiddenLayer, i-2)),
						matrices.MultiplyMatrices(r.Wxh.weight, matrices.ShiftMatrix(x, i-1))),
					matrices.ShiftMatrix(x, i-1)))
			// prod *= tanhPrime(r.Whh[i-1][i]*h[i-1]+r.Wxh[i][i]*x[i]) * r.Whh[i-1][i]
			// fmt.Printf("dh%d/dh%d ", i, i-1)
		}
		sum = matrices.TanhPrimeMatrix(
			matrices.MultiplyMatrices(
				matrices.MultiplyMatrices(
					matrices.AddMatrices(
						matrices.MultiplyMatrices(r.Whh.weight, matrices.ShiftMatrix(r.hiddenLayer, k-2)),
						matrices.MultiplyMatrices(r.Wxh.weight, matrices.ShiftMatrix(x, k-1))),
					r.Whh.weight),
				prod))
		// sum += tanhPrime(r.Whh[k-1][k]*h[k-1]+r.Wxh[k][k]*x[k]) * h[k-1] * prod
		// fmt.Printf("dh%d/dWh%dh%d ", k, k-1, k)
		if k != len(x)-1 {
			// fmt.Print("+ ")
		}
	}

	result := matrices.MultiplyScalar(matrices.MultiplyMatrices(dydh, sum), dEdy)
	return result
}

/* 	The updateWxh function updates the weight matrix Wxh by subtracting the product of the gradient and
the learning rate from each weight.
*/
func (r *RNN) updateWxh(wxhGrad [][]float64, learningRate float64) {
	r.Wxh.weight = matrices.SubtractMatrices(r.Wxh.weight, matrices.MultiplyScalar(wxhGrad, learningRate))
}
