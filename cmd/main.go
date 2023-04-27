package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

// RNN struct
type RNN struct {
	trainSize   int // size of the training window for the RNN
	predictSize int // size of the predicted values in the future using the RNN
	Wxh         float64
	Whh         float64
	Why         float64
	h           []float64
}

// sigmoid implements the sigmoid function
func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// tanhPrime implements the derivative of tanh function
func tanhPrime(x float64) float64 {
	return 1 - math.Pow(math.Tanh(x), 2)
}

// initRNN initializes the RNN
func (r *RNN) initRNN(trainSize int, predictSize int) *RNN {
	// Initialize the random number generator with a seed based on the current time
	rand.Seed(time.Now().UnixNano())

	r.trainSize, r.predictSize = trainSize, predictSize

	// Initialize the weight matrices between input layer and hidden layer
	r.Wxh = rand.Float64()

	// Initialize the weight matrices between hidden layer and hidden layer
	// Adding 1 since Wh0h0 is 0 and the useful values start from Wh0h1
	r.Whh = rand.Float64()

	// Initialize the weight matrices between hidden layer and output layer
	r.Why = rand.Float64()

	// Initialize the bias vectors for the hidden layer
	r.h = make([]float64, trainSize)

	return r
}

/*
	forwardPass performs the forward pass of the RNN
	h(t) = f( h(t-1), x(t) )
	h(t) = tanh ( Whh.h(t-1) + Wxh.x(t) )
	y(t) = Why.h(t)

	The inputs are the input vector x(t)
	The outputs are therefore are h(t) and y(t)
*/

func (r *RNN) forwardRNN(x []float64) ([]float64, error) {
	//TODO Optimise using Go Routines
	//Create the output vector and calculate y(t) = Why.h(t)
	y := make([]float64, len(x))

	//Calculate h(t) = tanh ( Whh.h(t-1) + Wxh.x(t) ) & y(t) = Why.h(t)
	for i := 1; i < len(x); i++ {
		r.h[i] = math.Tanh(r.h[i-1]*r.Whh + x[i]*r.Wxh)
		y[i] = r.Why * r.h[i]
	}
	return y, nil
}

/* 	This is a generic function to obtain the cross product of multiplying a two dimensional matrix with
a vector
*/
func CrossProduct(matrix [][]float64, vector []float64) []float64 {
	result := make([]float64, len(matrix))
	for i := 0; i < len(matrix); i++ {
		var sum float64
		for j := 0; j < len(vector); j++ {
			sum += matrix[i][j] * vector[j]
		}
		result[i] = sum
	}
	return result
}

/*  This is a generic function to add two vectors
 */
func AddVector(vectorA []float64, vectorB []float64) ([]float64, error) {
	if len(vectorA) != len(vectorB) {
		return nil, fmt.Errorf("vectors are not of the same size")
	}
	result := make([]float64, len(vectorA))
	for i := 0; i < len(vectorA); i++ {
		result[i] = vectorA[i] + vectorB[i]
	}
	return result, nil
}

/* 	This function calculates the mean squared error (MSE) between predicted outputs and ground truth outputs
using the formula 1/N Σ (y(T) - ŷ(T))^2 at time step T
*/
func (r *RNN) calculateMSE(predictions []float64, targets []float64) float64 {
	var mse float64
	for i := 0; i < len(predictions); i++ {
		mse += math.Pow(predictions[i]-targets[i], 2)
	}
	mse /= float64(len(predictions))
	return mse
}

/* 	This function calculates the derivative of mean squared error (MSE) with respect to predicted values
using the formula 2/N Σ (y(t) - ŷ(t))
*/
func (r *RNN) calculateMSEGrad(predictions []float64, targets []float64) float64 {
	var grad float64
	for i := 0; i < len(predictions); i++ {
		grad += predictions[i] - targets[i]
	}
	grad /= float64(len(predictions))
	return grad
}

/* 	This function calculates the derivative of error with respect to weight matrix Why
using the formula: 2/N Σ (y(t) - ŷ(t)) * h(t)
*/
func (r *RNN) calculateWhyGrad(predictions, targets, h []float64) float64 {
	mseGrad := r.calculateMSEGrad(predictions, targets)
	whyGrad := mseGrad * h[len(predictions)-1]
	return whyGrad
}

/* 	The updateWhy function updates the weight matrix Why by subtracting the product of the gradient and
the learning rate from each weight.
*/
func (r *RNN) updateWhy(whyGrad float64, learningRate float64) {
	r.Why = r.Why - (learningRate * whyGrad)
}

/* 	This function calculates the derivative of error with respect to weight matrix Whh
using the formula: dE/dy(T) * dy(T)/dh(T) * Σ dh(t)/dh(k) * dh(k)/dWhh
*/
func (r *RNN) calculateWhhGrad(predictions, targets, x []float64, h []float64) float64 {

	// Calculate the derivative of the error with respect to y dE/dy
	dEdy := r.calculateMSEGrad(predictions, targets)

	// Calculate the derivative of y with respect to h
	dydh := r.Why

	/* Calculate the term Σ dh(t)/dh(k) * dh(k)/dWhh where summation goes from k=1 to t
	and dh(t)/dh(k) = Π dh(i) / dh(i-1) where this goes from i=k+1 to t
	dh(t)/dWhh = sech(Whh * h(t-1) + Wxh x(t)) * h(t-1)
	dh(t)/dh(t-1) = sech(Whh * h(t-1) + Wxh x(t)) * Whh
	*/
	sum := 0.0
	for k := 1; k < len(x); k++ {
		prod := 1.0
		for i := k + 1; i < len(x); i++ {
			prod *= tanhPrime(r.Whh*h[i-1]+r.Wxh*x[i]) * r.Whh
			// fmt.Printf("dh%d/dh%d ", i, i-1)
		}
		sum += tanhPrime(r.Whh*h[k-1]+r.Wxh*x[k]) * h[k-1] * prod
		// fmt.Printf("dh%d/dWh%dh%d ", k, k-1, k)
		if k != len(x)-1 {
			// fmt.Print("+ ")
		}
	}

	result := dydh * sum * dEdy
	return result
}

/* 	The updateWhh function updates the weight matrix Why by subtracting the product of the gradient and
the learning rate from each weight.
*/
func (r *RNN) updateWhh(whhGrad float64, learningRate float64) {
	r.Why = r.Why - (learningRate * whhGrad)
}

/* 	This function calculates the derivative of error with respect to weight matrix Whx
using the formula: dE/dy(T) * dy(T)/dh(T) * Σ dh(t)/dh(k) * dh(k)/dWhx
*/
func (r *RNN) calculateWxhGrad(predictions, targets, x []float64, h []float64) float64 {
	// Calculate the derivative of the error with respect to y dE/dy
	dEdy := r.calculateMSEGrad(predictions, targets)

	// Calculate the derivative of y with respect to h
	dydh := r.Why

	/* Calculate the term Σ dh(t)/dh(k) * dh(k)/dWhx where summation goes from k=1 to t
	and dh(t)/dh(k) = Π dh(i) / dh(i-1) where this goes from i=k+1 to t
	dh(t)/dWhx = sech(Whh * h(t-1) + Wxh x(t)) * x(t)
	dh(t)/dh(t-1) = sech(Whh * h(t-1) + Wxh x(t)) * Whh
	*/
	sum := 0.0
	for k := 1; k < len(x); k++ {
		prod := 1.0
		for i := k + 1; i < len(x); i++ {
			prod *= tanhPrime(r.Whh*h[i-1]+r.Wxh*x[i]) * r.Whh
		}
		sum += tanhPrime(r.Whh*h[k-1]+r.Wxh*x[k]) * x[k] * prod
	}

	result := dydh * sum * dEdy
	return result
}

/* 	The updateWhh function updates the weight matrix Why by subtracting the product of the gradient and
the learning rate from each weight.
*/
func (r *RNN) updateWxh(wxhGrad float64, learningRate float64) {
	r.Wxh = r.Wxh - (learningRate * wxhGrad)
}

/*	This trainRNN function takes training data computes the error and then does the backward pass to update the
	respective weight matrices. This process is repeated for a number of iterations. The number of backward and forward
	pass iterations are referred to as the number of epochs
*/
func (r *RNN) trainRNN(numEpochs int, trainingInputs []float64, actualOutputs []float64, learningRate float64) {

	for iter := 0; iter < numEpochs; iter++ {
		// Forward Pass using input data x
		predictedOutputs, _ := r.forwardRNN(trainingInputs)
		fmt.Printf("Mean Square Error: %f\n", r.calculateMSE(predictedOutputs, actualOutputs))
		// Backward Pass and adjust weights
		whyGrad := r.calculateWhyGrad(predictedOutputs, actualOutputs, r.h)
		r.updateWhy(whyGrad, learningRate)
		whhGrad := r.calculateWhhGrad(predictedOutputs, actualOutputs, trainingInputs, r.h)
		r.updateWhh(whhGrad, learningRate)
		whxGrad := r.calculateWxhGrad(predictedOutputs, actualOutputs, trainingInputs, r.h)
		r.updateWxh(whxGrad, learningRate)
	}
}

/*
	predictRNN performs the forward pass of the RNN using the training data upto time T
	h(t) = f( h(t-1), x(t) )
	h(t) = tanh ( Whh.h(t-1) + Wxh.x(t) )
	y(t) = Why.h(t)

	The inputs are the input vector x(t)
	The outputs are therefore are h(t) and y(t)
*/

func (r *RNN) predictRNN(x []float64) ([]float64, error) {
	//Create the output vector and calculate y(t) = Why.h(t)
	y := make([]float64, len(x))

	//Calculate h(t) = tanh ( Whh.h(t-1) + Wxh.x(t) ) & y(t) = Why.h(t)
	for i, v := range x {
		hT := r.h[len(r.h)-1]
		r.h = append(r.h, math.Tanh(hT*r.Whh+v*r.Wxh))
		fmt.Printf("h[%d]: %f\n", len(r.h)-1, math.Tanh(hT*r.Whh+v*r.Wxh))
		y[i] = r.Why * hT
	}
	return y, nil
}

// helper function to calculate dot product of two vectors
func dot(a, b []float64) float64 {
	res := 0.0
	for i := range a {
		res += a[i] * b[i]
	}
	return res
}

func scalarMultiply(matrix [][]float64, scalar float64) [][]float64 {
	rows := len(matrix)
	cols := len(matrix[0])

	result := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		result[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			// fmt.Printf("result[%d][%d] = %f\n",i ,j, matrix[i][j] * scalar)
			result[i][j] = matrix[i][j] * scalar
		}
	}

	return result
}

func addMatrices(mat1 [][]float64, mat2 [][]float64) [][]float64 {
	rows := len(mat1)
	cols := len(mat1[0])
	result := make([][]float64, rows)
	for i := range result {
		result[i] = make([]float64, cols)
		for j := range result[i] {
			result[i][j] = mat1[i][j] + mat2[i][j]
		}
	}
	return result
}

func multiplyMatrices(a, b [][]float64) [][]float64 {
	// Check if matrices can be multiplied
	n := len(a)
	m := len(a[0])
	q := len(b[0])

	// if m != p {
	//	return nil, errors.New("matrices cannot be multiplied")
	// }

	// Initialize result matrix
	result := make([][]float64, n)
	for i := range result {
		result[i] = make([]float64, q)
	}

	// Multiply matrices
	for i := 0; i < n; i++ {
		for j := 0; j < q; j++ {
			sum := 0.0
			for k := 0; k < m; k++ {
				sum += a[i][k] * b[k][j]
			}
			result[i][j] = sum
		}
	}

	return result
}

func tanhVec(vec [][]float64) float64 {
	sum := 0.0
	for i := 0; i < len(vec); i++ {
		for j := 0; j < len(vec[0]); j++ {
			sum += vec[i][j]
		}
	}
	return math.Tanh(sum)
}

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

	// Create a neural network and train it
	RNN := RNN{}
	RNN.initRNN(totalRange, 1)
	// fmt.Print(RNN.Whh)
	// fmt.Printf("\n")
	RNN.trainRNN(100, inputs, actualOutputs, 2)

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

	// xNew := []float64{10 * math.Pi + math.Pi/2} // new input vector for which we want to generate predictions

	yPred, err := RNN.predictRNN(xNew) // generate predictions using the trained RNN model
	if err != nil {
		// handle error
	}

	// yActual := []float64{math.Sin(10 * math.Pi + math.Pi/2)}
	fmt.Print(yActual)
	fmt.Printf("\n")
	fmt.Print(yPred)
}
