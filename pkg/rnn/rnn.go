package rnn

import (
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"math"
	"time"
)

// RNN struct
type RNN struct {
	numInputs       int
	numOutputs      int
	numHiddenLayers int
	Wxh             *mat.Dense
	Whh             *mat.Dense
	Why             *mat.Dense
	hiddenLayer     *mat.Dense
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
creates a Dense matrix of size numInputs by numHiddenLayers
and initializes all elements to a random value
*/
func (builder *Builder) SetWxh(numInputs, numHiddenLayers int) *Builder {
	rand.Seed(uint64(time.Now().UnixNano()))

	data := make([]float64, numInputs*numHiddenLayers)
	for i := range data {
		data[i] = rand.Float64()
	}

	builder.neuralNet.Wxh = mat.NewDense(numInputs, numHiddenLayers, data)
	return builder
}

/* Sets the weight matrix between the hidden layers,
creates a Dense matrix of size numHiddenLayers by numHiddenLayers
and initializes all elements to a random value
*/
func (builder *Builder) SetWhh(numHiddenLayers int) *Builder {
	rand.Seed(uint64(time.Now().UnixNano()))

	data := make([]float64, numHiddenLayers*numHiddenLayers)
	for i := range data {
		data[i] = rand.Float64()
	}

	builder.neuralNet.Whh = mat.NewDense(numHiddenLayers, numHiddenLayers, data)
	return builder
}

/* Sets the weight matrix between the hidden layers and output neurons,
creates a Dense matrix of size numHiddenLayers by numOutputs
and initializes all elements to a random value
*/
func (builder *Builder) SetWhy(numHiddenLayers, numOutputs int) *Builder {
	rand.Seed(uint64(time.Now().UnixNano()))

	data := make([]float64, numHiddenLayers*numOutputs)
	for i := range data {
		data[i] = rand.Float64()
	}

	builder.neuralNet.Why = mat.NewDense(numHiddenLayers, numOutputs, data)
	return builder
}

// Builder function to build the RNN
func (builder *Builder) Build() RNN {
	return builder.neuralNet
}

/*
	forwardPass performs the forward pass of the RNN
	h(t) = f( h(t-1), x(t) )
	h(t) = tanh ( Whh.h(t-1) + Wxh.x(t) )
	y(t) = Why.h(t)
	The inputs are the input vector x(t)
	The outputs are therefore are h(t) and y(t)
*/
func (r *RNN) forwardRNN(x *mat.Dense) (*mat.Dense, error) {
	// Initialize hidden layer and output matrices
	r.hiddenLayer = mat.NewDense(x.RawMatrix().Rows, r.numHiddenLayers, nil)
	output := mat.NewDense(x.RawMatrix().Rows, r.numOutputs, nil)

	// Define a wrapper function for math.Tanh
	tanhWrapper := func(_, _ int, v float64) float64 {
		return math.Tanh(v)
	}

	// Calculate h(t) = tanh(Whh.h(t-1) + Wxh.x(t)) & y(t) = Why.h(t)
	for i := 1; i < x.RawMatrix().Rows; i++ {
		hiddenPrev := r.hiddenLayer.RawRowView(i - 1)
		xCurr := mat.NewDense(1, x.RawMatrix().Cols, nil)
		xCurr.SetRow(0, x.RawRowView(i))
		hiddenCurr := mat.NewDense(1, r.numHiddenLayers, nil)
		hiddenCurr.Mul(xCurr, r.Wxh)
		hiddenCurr.Add(hiddenCurr, mat.NewDense(1, r.numHiddenLayers, hiddenPrev))
		hiddenCurr.Apply(tanhWrapper, hiddenCurr)
		r.hiddenLayer.SetRow(i, hiddenCurr.RawRowView(0))
		outputCurr := mat.NewDense(1, r.numOutputs, nil)
		outputCurr.Mul(hiddenCurr, r.Why)
		output.SetRow(i, outputCurr.RawRowView(0))
	}
	return output, nil
}

/* 	This function calculates the derivative of mean squared error (MSE) with respect to predicted values
using the formula 2/N Σ (y(t) - ŷ(t))
*/
func calculateMSEGrad(predictions, targets *mat.Dense) float64 {
	// Calculate the error sum
	// Assume predictions and targets have same dimensions
	diff := mat.NewDense(predictions.RawMatrix().Rows, predictions.RawMatrix().Cols, nil)
	diff.Sub(predictions, targets)
	errorSum := mat.Sum(diff)

	// Divide the error sum by the total number of elements
	_, cols := predictions.Dims()
	N := float64(cols)
	mseGrad := (2 / N) * errorSum

	return mseGrad
}

/* 	This function calculates the derivative of error with respect to weight matrix Why
using the formula: 2/N Σ (y(t) - ŷ(t)) * h(t)
The resulting gradient is a two-dimensional matrix of size  len(h) x len(predictions)
where each element represents the partial derivative of the error with respect to a
specific weight in the Why matrix.
*/
func (r *RNN) calculateWhyGrad(predictions, targets *mat.Dense) *mat.Dense {
	mseGrad := calculateMSEGrad(predictions, targets)
	// Get the last row of the hidden layer
	whyGrad := mat.DenseCopyOf(r.hiddenLayer.RowView(r.hiddenLayer.RawMatrix().Rows - 1))
	// Multiply it with delta of MSE
	whyGrad.Scale(mseGrad, whyGrad)
	return whyGrad
}

/* 	The updateWhy function updates the weight matrix Why by subtracting the product of the gradient and
the learning rate from each weight. The updated weights are returned as a new two-dimensional matrix.
*/
func (r *RNN) updateWhy(whyGrad *mat.Dense, learningRate float64) {
	// Calculate the update for r.Why
	rows, cols := r.Why.Dims()
	update := mat.NewDense(rows, cols, nil)
	update.Scale(learningRate, whyGrad)
	r.Why.Sub(r.Why, update)
}

// This function calculates the value of Dh(t-x)/DWhh at time shift x
// dh(t)/dWhh = sech(Whh * h(t-1) + Wxh x(t)) * h(t-1)
func (r *RNN) calculateDhtDWhh(inputs *mat.Dense, x int) *mat.Dense {
	// Get the hidden layer at time t-x-1
	hiddenPrev := mat.DenseCopyOf(r.hiddenLayer.RowView(x - 1))

	// Calculate Whh * h(t-x-1)
	whhHiddenPrev := mat.NewDense(r.numHiddenLayers, 1, nil)
	whhHiddenPrev.Mul(r.Whh, hiddenPrev)

	// Calculate Wxh * x(t)
	wxhX := mat.NewDense(r.numHiddenLayers, 1, nil)
	wxhX.Mul(r.Wxh, inputs.RowView(x))

	// Calculate Whh * h(t-x-1) + Wxh * x(t)
	linearSum := mat.NewDense(r.numHiddenLayers, 1, nil)
	linearSum.Add(whhHiddenPrev, wxhX)

	// Calculate sech(Whh * h(t-x-1) + Wxh * x(t))
	tanhVal := mat.NewDense(r.numHiddenLayers, 1, nil)
	tanhVal.Apply(func(_, _ int, v float64) float64 {
		return 1 / math.Cosh(v)
	}, linearSum)

	// Calculate Dh(t-x)/DWhh = sech(Whh * h(t-x-1) + Wxh * x(t)) * h(t-x-1)
	dhtDWhh := mat.NewDense(r.numHiddenLayers, r.numHiddenLayers, nil)
	dhtDWhh.Mul(tanhVal, hiddenPrev.T())

	return dhtDWhh
}

// This function calculates the value of Dh(t-x)/Dh(t-x-1) at time shift x
func (r *RNN) calculateDhtDht(inputs *mat.Dense, x int) *mat.Dense {
	// Get the hidden layer at time t-x-1
	hiddenPrev := mat.DenseCopyOf(r.hiddenLayer.RowView(x - 1))

	// Calculate Whh * h(t-x-1)
	whhHiddenPrev := mat.NewDense(r.numHiddenLayers, 1, nil)
	whhHiddenPrev.Mul(r.Whh, hiddenPrev)

	// Calculate Wxh * x(t)
	wxhX := mat.NewDense(r.numHiddenLayers, 1, nil)
	wxhX.Mul(r.Wxh, inputs.RowView(x))

	// Calculate Whh * h(t-x-1) + Wxh * x(t)
	linearSum := mat.NewDense(r.numHiddenLayers, 1, nil)
	linearSum.Add(whhHiddenPrev, wxhX)

	// Calculate sech(Whh * h(t-x-1) + Wxh * x(t))
	tanhVal := mat.NewDense(r.numHiddenLayers, 1, nil)
	tanhVal.Apply(func(_, _ int, v float64) float64 {
		return 1 / math.Cosh(v)
	}, linearSum)

	// Calculate Dh(t)/Dh(t-1) = sech(Whh * h(t-x-1) + Wxh * x(t)) * Whh
	dhtDhtMinusOne := mat.NewDense(r.numHiddenLayers, r.numHiddenLayers, nil)
	dhtDhtMinusOne.Mul(tanhVal, r.Whh)

	return dhtDhtMinusOne
}

// This function calculates the value of Dh(t)/DWhx at time shift x
func (r *RNN) calculateDhtDWhx(inputs *mat.Dense, x int) *mat.Dense {
	// Get the hidden layer at time t-x-1
	hiddenPrev := mat.DenseCopyOf(r.hiddenLayer.RowView(x - 1))

	// Calculate Whh * h(t-x-1)
	whhHiddenPrev := mat.NewDense(r.numHiddenLayers, 1, nil)
	whhHiddenPrev.Mul(r.Whh, hiddenPrev)

	// Calculate Wxh * x(t)
	wxhX := mat.NewDense(r.numHiddenLayers, 1, nil)
	wxhX.Mul(r.Wxh, inputs.RowView(x))

	// Calculate Whh * h(t-x-1) + Wxh * x(t)
	linearSum := mat.NewDense(r.numHiddenLayers, 1, nil)
	linearSum.Add(whhHiddenPrev, wxhX)

	// Calculate sech(Whh * h(t-x-1) + Wxh * x(t))
	tanhVal := mat.NewDense(r.numHiddenLayers, 1, nil)
	tanhVal.Apply(func(_, _ int, v float64) float64 {
		return 1 / math.Cosh(v)
	}, linearSum)

	// Calculate Dh(t)/DWhx = sech(Whh * h(t-x-1) + Wxh * x(t)) * x(t)
	dhtDWhx := mat.NewDense(r.numHiddenLayers, r.numInputs, nil)
	dhtDWhx.Mul(tanhVal, inputs.RowView(x).T())

	return dhtDWhx
}

/* 	This function calculates the derivative of error with respect to weight matrix Whh
using the formula: dE/dy(T) * dy(T)/dh(T) * Σ dh(t)/dh(k) * dh(k)/dWhh
*/
func (r *RNN) calculateWhhGrad(predictions, targets, x *mat.Dense) *mat.Dense {
	// Calculate the derivative of the error with respect to y, dE/dy
	dEdy := calculateMSEGrad(predictions, targets)

	// Calculate the derivative of y with respect to h, dy/dh
	dydh := r.Why

	// Create matrices for the sum and product
	sum := mat.NewDense(r.numHiddenLayers, x.RawMatrix().Rows, nil)
	prod := mat.NewDense(r.numHiddenLayers, r.numHiddenLayers, nil)

	// Calculate the term Σ dh(t)/dh(k) * dh(k)/dWhh where the summation goes from k=1 to t
	// dh(t)/dh(k) = Π dh(i)/dh(i-1) where i goes from k+1 to t
	// dh(t)/dWhh = sech(Whh * h(t-1) + Wxh * x(t)) * h(t-1)
	// dh(t)/dh(t-1) = sech(Whh * h(t-1) + Wxh * x(t)) * Whh
	for t := 0; t < x.RawMatrix().Rows; t++ {
		// Initialize prod to the identity matrix
		if t > 0 {
			prod.Reset()
			for i := t; i > 0; i-- {
				dhtDht := r.calculateDhtDht(x, i-1)
				prod.Mul(dhtDht, prod)
			}
		}
		dhtDWhh := r.calculateDhtDWhh(x, t)
		prod.Mul(dhtDWhh, prod)
		sum.Add(sum, prod)
	}

	result := mat.NewDense(0, 0, nil)
	result.Mul(dydh, sum)
	result.Scale(dEdy, result)

	return result
}

/* 	The updateWhh function updates the weight matrix Whh by subtracting the product of the gradient and
the learning rate from each weight.
*/
func (r *RNN) updateWhh(whhGrad *mat.Dense, learningRate float64) {
	// Calculate the update for r.Whh
	update := mat.NewDense(r.Whh.RawMatrix().Rows, r.Whh.RawMatrix().Cols, nil)
	update.Scale(learningRate, whhGrad)
	r.Whh.Sub(r.Whh, update)
}

/* 	This function calculates the derivative of error with respect to weight matrix Whh
using the formula: dE/dy(T) * dy(T)/dh(T) * Σ dh(t)/dh(k) * dh(k)/dWhh
*/
func (r *RNN) calculateWhxGrad(predictions, targets, x *mat.Dense) *mat.Dense {
	// Calculate the derivative of the error with respect to y, dE/dy
	dEdy := calculateMSEGrad(predictions, targets)

	// Calculate the derivative of y with respect to h, dy/dh
	dydh := r.Why

	// Create matrices for the sum and product
	sum := mat.NewDense(r.numHiddenLayers, x.RawMatrix().Rows, nil)
	prod := mat.NewDense(r.numHiddenLayers, r.numHiddenLayers, nil)

	/* Calculate the term Σ dh(t)/dh(k) * dh(k)/dWhh where summation goes from k=1 to t
	and dh(t)/dh(k) = Π dh(i) / dh(i-1) where this goes from i=k+1 to t
	dh(t)/dWhx = sech(Whh * h(t-1) + Wxh x(t)) * x(t)
	dh(t)/dh(t-1) = sech(Whh * h(t-1) + Wxh x(t)) * Whh
	*/
	for t := 0; t < x.RawMatrix().Rows; t++ {
		// Initialize prod to the identity matrix
		if t > 0 {
			prod.Reset()
			for i := t; i > 0; i-- {
				dhtDht := r.calculateDhtDht(x, i-1)
				prod.Mul(dhtDht, prod)
			}
		}
		dhtDWhx := r.calculateDhtDWhx(x, t)
		prod.Mul(dhtDWhx, prod)
		sum.Add(sum, prod)
	}

	result := mat.NewDense(0, 0, nil)
	result.Mul(dydh, sum)
	result.Scale(dEdy, result)

	return result
}

/* 	The updateWxh function updates the weight matrix Wxh by subtracting the product of the gradient and
the learning rate from each weight.
*/
func (r *RNN) updateWxh(wxhGrad *mat.Dense, learningRate float64) {
	// Calculate the update for r.Wxh
	update := mat.NewDense(r.Wxh.RawMatrix().Rows, r.Wxh.RawMatrix().Cols, nil)
	update.Scale(learningRate, wxhGrad)
	r.Wxh.Sub(r.Wxh, update)
}

/*	This trainRNN function takes training data computes the error and then does the backward pass to update the
	respective weight matrices. This process is repeated for a number of iterations. The number of backward and forward
	pass iterations are referred to as the number of epochs
*/
func (r *RNN) TrainRNN(numEpochs int, trainingInputs, actualOutputs *mat.Dense, learningRate float64) {
	for iter := 0; iter < numEpochs; iter++ {
		// Forward Pass using input data x
		predictedOutputs, _ := r.forwardRNN(trainingInputs)
		// Backward Pass and adjust weights
		whyGrad := r.calculateWhyGrad(predictedOutputs, actualOutputs)
		r.updateWhy(whyGrad, learningRate)
		whhGrad := r.calculateWhhGrad(predictedOutputs, actualOutputs, trainingInputs)
		r.updateWhh(whhGrad, learningRate)
		whxGrad := r.calculateWhxGrad(predictedOutputs, actualOutputs, trainingInputs)
		r.updateWxh(whxGrad, learningRate)
	}
}
