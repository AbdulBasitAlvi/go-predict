package matrices

import (
	"fmt"
	"math"
)

func MultiplyMatrices(matrix1, matrix2 [][]float64) [][]float64 {
	rows1, cols1 := len(matrix1), len(matrix1[0])
	rows2, cols2 := len(matrix2), len(matrix2[0])

	// Check if the matrices can be multiplied
	if cols1 != rows2 {
		fmt.Println("Cannot multiply the matrices. Invalid dimensions.")
		return nil
	}

	// Create the resulting matrix
	result := make([][]float64, rows1)
	for i := 0; i < rows1; i++ {
		result[i] = make([]float64, cols2)
	}

	// Perform matrix multiplication
	for i := 0; i < rows1; i++ {
		for j := 0; j < cols2; j++ {
			sum := 0.0
			for k := 0; k < cols1; k++ {
				sum += matrix1[i][k] * matrix2[k][j]
			}
			result[i][j] = sum
		}
	}

	return result
}

func AddMatrices(matrix1, matrix2 [][]float64) [][]float64 {
	rows1, cols1 := len(matrix1), len(matrix1[0])
	rows2, cols2 := len(matrix2), len(matrix2[0])

	// Check if the matrices have the same dimensions
	if rows1 != rows2 || cols1 != cols2 {
		fmt.Println("Cannot add the matrices. Invalid dimensions.")
		return nil
	}

	// Create the resulting matrix
	result := make([][]float64, rows1)
	for i := 0; i < rows1; i++ {
		result[i] = make([]float64, cols1)
	}

	// Perform matrix addition
	for i := 0; i < rows1; i++ {
		for j := 0; j < cols1; j++ {
			result[i][j] = matrix1[i][j] + matrix2[i][j]
		}
	}

	return result
}

func SubtractMatrices(matrix1, matrix2 [][]float64) [][]float64 {
	rows1, cols1 := len(matrix1), len(matrix1[0])
	rows2, cols2 := len(matrix2), len(matrix2[0])

	// Check if the matrices have the same dimensions
	if rows1 != rows2 || cols1 != cols2 {
		fmt.Println("Cannot subtract the matrices. Invalid dimensions.")
		return nil
	}

	// Create the resulting matrix
	result := make([][]float64, rows1)
	for i := 0; i < rows1; i++ {
		result[i] = make([]float64, cols1)
	}

	// Perform matrix subtraction
	for i := 0; i < rows1; i++ {
		for j := 0; j < cols1; j++ {
			result[i][j] = matrix1[i][j] - matrix2[i][j]
		}
	}

	return result
}

func MultiplyScalar(matrix [][]float64, scalar float64) [][]float64 {
	rows, cols := len(matrix), len(matrix[0])

	// Create the resulting matrix
	result := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		result[i] = make([]float64, cols)
	}

	// Perform scalar multiplication
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result[i][j] = matrix[i][j] * scalar
		}
	}

	return result
}

func TanhPrimeMatrix(matrix [][]float64) [][]float64 {
	rows, cols := len(matrix), len(matrix[0])

	// Create the resulting matrix
	result := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		result[i] = make([]float64, cols)
	}

	// Perform element-wise tanh prime calculation
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result[i][j] = 1 - math.Pow(math.Tanh(matrix[i][j]), 2)
		}
	}

	return result
}

func TanhPrimeMatrix1D(matrix []float64) []float64 {
	size := len(matrix)

	// Create the resulting matrix
	result := make([]float64, size)

	// Perform element-wise tanh prime calculation
	for i := 0; i < size; i++ {
		result[i] = 1 - math.Pow(math.Tanh(matrix[i]), 2)
	}

	return result
}

func ShiftMatrix(mat [][]float64, shift int) [][]float64 {
	rows := len(mat)
	cols := len(mat[0])

	// Create a shifted matrix with the same dimensions as the input matrix
	shiftedMat := make([][]float64, rows)
	for i := range shiftedMat {
		shiftedMat[i] = make([]float64, cols)
	}

	// Shift the values of the matrix
	copy(shiftedMat, mat[shift:])

	return shiftedMat
}

func InitialiseMatrix(numColumns int, numRows int, value float64) [][]float64 {
	matrix := make([][]float64, numColumns)
	for i := 0; i < numColumns; i++ {
		matrix[i] = make([]float64, numRows)
		for j := 0; j < numRows; j++ {
			matrix[i][j] = value
		}
	}
	return matrix
}
