package com.moulidutta;

import java.math.BigDecimal;
import java.math.MathContext;
import java.math.RoundingMode;
import java.util.Arrays;
import java.util.Scanner;
import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 *  {@code Matrix} class can perform the following matrix operations
 *  1.  Generate Random Matrix ✔
 *  2.  Take user input ✔
 *  3.  Print matrix to console ✔
 *  4.  Perform Matrix Addition ✔
 *  5.  Perform Matrix Subtraction ✔
 *  6.  Perform Matrix Multiplication ✔
 *  7.  Perform Matrix Transpose ✔
 *  8.  Find Determinant ✔
 *  9.  Find Inverse ✔
 *  10. Multiply matrix with a scalar ✔
 *  11. Find Trace ✔
 *  12. Find Normal ✔
 *  13. Find Rank ✔
 *  14. Check if orthogonal ✔
 *  15. Check if Symmetric ✔
 *  16. Check if skew Symmetric ✔
 *  17. Check if Sparse ✔
 */

public class Matrix {
    private final int rows;
    private final int columns;
    private BigDecimal[][] matrix;

    public Matrix(int rows, int columns) {
        this.rows = rows;
        this.columns = columns;
        matrix = new BigDecimal[rows][columns];

        // fill matrix with zeros
        IntStream.range(0, this.rows).forEach(i -> Arrays.fill(this.matrix[i], BigDecimal.ZERO));

    }

    /**
     * Method for generating matrix with random values
     * @param lower lower bound for random numbers
     * @param upper upper bound for random numbers
     */
    public void generateRandomMatrix(long lower, long upper) {

        if (this.rows <= 0 || this.columns <= 0)
            throw new IllegalStateException("Rows and columns of the matrix must be greater than zero.");

        this.matrix =
                IntStream.range(0, this.rows)
                        .mapToObj(i ->
                                IntStream.range(0, this.columns)
                                        .mapToObj(j -> BigDecimal.valueOf(ThreadLocalRandom.current().nextLong(lower, upper)))
                                        .toArray(BigDecimal[]::new)
                        )
                        .toArray(BigDecimal[][]::new);
    }

    /**
     *  method to initialise the matrix
     * @param sc Scanner object to initialise matrix with user entered values.
     */
    public void initMatrix(Scanner sc) {
        if (this.rows <= 0 || this.columns <= 0)
            throw new IllegalStateException("Rows and columns of the matrix must be greater than zero.");

        this.matrix = IntStream.range(0, this.rows)
                .mapToObj(i ->
                        IntStream.range(0, this.columns)
                                .mapToObj(j -> new BigDecimal(sc.next()))
                                .toArray(BigDecimal[]::new)
                )
                .toArray(BigDecimal[][]::new);
    }

    /**
     * method to add two matrices
     * @param mat second matrix
     * @return addition matrix
     */
    public Matrix add(Matrix mat) {

        Matrix result = new Matrix(this.rows, this.columns);
        if (mat.rows != this.rows || mat.columns != this.columns)
            throw new IllegalStateException("Column of the matrix must be equal.");

        result.matrix =
                IntStream.range(0, this.rows)
                        .mapToObj(i ->
                                IntStream.range(0, this.columns)
                                        .mapToObj(j -> this.matrix[i][j].add(mat.matrix[i][j]))
                                        .toArray(BigDecimal[]::new)
                        )
                        .toArray(BigDecimal[][]::new);

        return result;
    }

    /**
     * method to subtract two matrices
     * @param mat second matrix
     * @return subtraction matrix
     */
    public Matrix subtract(Matrix mat) {

        Matrix result = new Matrix(this.rows, this.columns);
        if (mat.rows != this.rows || mat.columns != this.columns)
            throw new IllegalStateException("Column of the matrix must be equal.");

        result.matrix =
                IntStream.range(0, this.rows)
                        .mapToObj(i ->
                                IntStream.range(0, this.columns)
                                        .mapToObj(j -> this.matrix[i][j].subtract(mat.matrix[i][j]))
                                        .toArray(BigDecimal[]::new)
                        )
                        .toArray(BigDecimal[][]::new);

        return result;
    }

    /**
     * method to multiply two matrices
     * @param mat second matrix
     * @return multiplication matrix
     */
    public Matrix multiply(Matrix mat) {

        Matrix result = new Matrix(this.rows, this.columns);
        if (mat.rows != this.columns)
            throw new IllegalStateException("Column of the first matrix should be equal to the rows of the second matrix.");

        // multiply the two matrices
        result.matrix =
                IntStream.range(0, this.rows)
                        .mapToObj(r ->
                                IntStream.range(0, this.columns)
                                        .mapToObj(c ->
                                                IntStream.range(0, this.columns)
                                                        .mapToObj(k -> (this.matrix[r][k].multiply( mat.matrix[k][c])))
                                                        .reduce(BigDecimal.ZERO, BigDecimal::add)
                                        )
                                        .toArray(BigDecimal[]::new)
                        )
                        .toArray(BigDecimal[][]::new);

        return result;
    }

    /**
     * transpose of a matrix
     * @return transpose matrix
     */
    public Matrix getTranspose() {
        Matrix transpose = new Matrix(this.columns, this.rows);
        transpose.matrix = IntStream.range(0, this.columns)
                .mapToObj(i ->
                        IntStream.range(0, this.rows)
                                .mapToObj(j -> this.matrix[j][i])
                                .toArray(BigDecimal[]::new)
                ).toArray(BigDecimal[][]::new);

        return transpose;
    }

    /**
     * mathod to find the determinant of the given matrix
     * @return determinant
     */
    public BigDecimal getDeterminant() {
        if (this.rows != this.columns) throw new IllegalStateException("Only square matrix have determinant");

        if (this.rows == 1) return this.matrix[0][0];

        return IntStream.range(0, this.rows)
                .mapToObj(i ->
                        BigDecimal.ONE.negate().pow(i)
                                .multiply(this.matrix[0][i]
                                        .multiply(
                                                minor(this.matrix, 0, i)
                                                        .getDeterminant()
                                        )
                                )
                )
                .reduce(BigDecimal.ZERO, BigDecimal::add);
    }

    /**
     * method to generate minor matrices
     * @param matrix 2d array of doubles
     * @param rows matrix row
     * @param column matrix column
     * @return minor matrix
     */
    private static Matrix minor(BigDecimal[][] matrix, int rows, int column) {
        Matrix minor = new Matrix(matrix.length-1, matrix[0].length-1 );

        minor.matrix =
                IntStream.range(0, minor.matrix.length)
                        .mapToObj(i ->
                                IntStream.range(0, minor.matrix[0].length)
                                        .mapToObj(j -> matrix[i < rows ? i : i+1] [j < column ? j : j+1])
                                        .toArray(BigDecimal[]::new)
                        )
                        .toArray(BigDecimal[][]::new);

        return minor;
    }

    /**
     * get Inverse of given matrix
     * @return inverse matrix
     */
    public Matrix getInverse() {

        if (this.rows != this.columns) throw new IllegalStateException("Only square matrix have inverse");

        double determinant = this.getDeterminant().doubleValue();
        if (determinant == 0) throw new IllegalStateException("Matrix inverse is not possible since determinant is 0.");

        Matrix cofactor = getCofactorMatrix();
        Matrix adjugate = cofactor.getTranspose();
        double invScalar = 1 / determinant;

        return adjugate.multiplyMatrixByScalar(Math.round(invScalar * 100.0) / 100.0); // as inverse of the matrix => adj(A)/det(A)
    }

    /**
     * @return cofactor matrix
     */
    private Matrix getCofactorMatrix() {
        Matrix cofactor = new Matrix(this.rows, this.rows);

        cofactor.matrix =
                IntStream.range(0, this.rows)
                        .mapToObj(i ->
                                IntStream.range(0, this.rows)
                                        .mapToObj(j -> getCofactors(this.matrix, i, j))
                                        .toArray(BigDecimal[]::new)
                        )
                        .toArray(BigDecimal[][]::new);

        return cofactor;
    }


    /**
     * @param matrix the 2d array whose cofactors needs to be found
     * @param i current row
     * @param j current column
     * @return cofactor
     */
    private static BigDecimal getCofactors(BigDecimal[][] matrix, int i, int j) {
        // C = ((-1)^(i+j)  *  Mij) 1 <= i, j <= n
        return minor(matrix, i, j).getDeterminant().multiply(BigDecimal.valueOf(Math.pow(-1, i + j)));
    }


    /**
     * multiply a matrix with a scalar quantity
     * @param scalar scalar component
     * @return scaled matrix
     */
    public Matrix multiplyMatrixByScalar(final double scalar) {
        Matrix scalarMultiply = new Matrix(this.rows, this.columns);

        scalarMultiply.matrix =
                IntStream.range(0, this.rows)
                        .mapToObj(i ->
                                IntStream.range(0, this.columns)
                                        .mapToObj(j -> this.matrix[i][j].multiply(BigDecimal.valueOf(scalar)))
                                        .toArray(BigDecimal[]::new)
                        )
                        .toArray(BigDecimal[][]::new);

        return scalarMultiply;
    }

    /**
     * method to get exponential matrix
     * @param n exponent
     * @return matrix raised to the power n
     */
    public Matrix exponential(int n) {
        if (this.rows != this.columns)
            throw new IllegalArgumentException("Matrix must be a square matrix");
        Matrix exponential = getIdentityMatrix(this.rows);

        for (int i = 0; i < n; i++) {
            exponential = exponential.multiply(this);
        }
        return exponential;
    }

    // Identity matrix
    private static Matrix getIdentityMatrix(int rows) {
        Matrix I = new Matrix(rows, rows);
        IntStream.range(0, rows).forEach(i -> I.matrix[i][i] = BigDecimal.ONE);
        return I;
    }

    /**
     * @return trace of matrix (sum of all elements in leading diagonal)
     */
    public BigDecimal getTrace() {
        if (this.rows != this.columns) throw new RuntimeException("Matrix must be a square matrix");

        return IntStream.range(0, this.rows)
                .mapToObj(i -> matrix[i][i])
                .reduce(BigDecimal.ZERO, BigDecimal::add);
    }

    /**
     * normal of a matrix
     * @return normal (sqrt of sum of squares of all matrix components)
     */
    public BigDecimal getNormal() {
        if (this.rows != this.columns) throw new RuntimeException("Matrix must be a square matrix");

        BigDecimal sum =
                IntStream.range(0, this.rows)
                        .mapToObj(i ->
                                IntStream.range(0, this.rows)
                                        .mapToObj(j -> this.matrix[i][j].multiply( this.matrix[i][j]))
                                        .toArray(BigDecimal[]::new))
                        .flatMap(Arrays::stream)
                        .reduce(BigDecimal.ZERO, BigDecimal::add);

        return sum.sqrt(new MathContext(50)).setScale(10, RoundingMode.HALF_UP);
    }

    /**
     * Rank of matrix (Number of linearly independent columns)
     * @return rank
     */
    public long getRank() {
        BigDecimal[][] rref = rref(this.matrix);
        // count how many non zero rows are present in the RREF matrix
        return Arrays.stream(rref).filter(row -> Arrays.stream(row).anyMatch(col -> !col.equals(BigDecimal.ZERO))).count();
    }

    /**
     * Method to find the Row Reduced Echelon Form (rref)
     * @param matrix, 2d array whose rref form is needed
     * @return rref 2d array
     */
    private BigDecimal[][] rref(BigDecimal[][] matrix) {
        BigDecimal[][] rref = new BigDecimal[matrix.length][matrix[0].length];

        // copy matrix
        IntStream.range(0, rref.length).forEach(r1 -> System.arraycopy(matrix[r1], 0, rref[r1], 0, rref[r1].length));

        pivotingProcess(rref, matrix.length < matrix[0].length ? rref.length : rref[0].length);
        return rref;
    }

    private void pivotingProcess(BigDecimal[][] rref, int limit) {
        for (int p = 0; p < limit; p++) {
            BigDecimal pivot = rref[p][p];

            if (!pivot.equals(BigDecimal.ZERO)) {
                double invPivot = 1.0 / pivot.doubleValue();
                for (int i = 0; i < rref[p].length; i++)
                    rref[p][i] = rref[p][i].multiply(BigDecimal.valueOf(invPivot));
            }

            // make other rows zero
            for (int r = 0; r < rref.length; r++) {
                if (r != p) {
                    BigDecimal f = rref[r][p];
                    for (int i = 0; i < rref[r].length; i++)
                        rref[r][i] = rref[r][i].subtract(f.multiply( rref[p][i]));
                }
            }
        }
    }

    public boolean isSparseMatrix() {

        return IntStream.range(0, this.rows)
                       .mapToObj(i ->
                               IntStream.range(0, this.columns)
                                       .filter(j -> this.matrix[i][j].doubleValue() == 0)
                                       .count()
                       )
                       .mapToLong(Long::longValue)
                       .sum()
               > (((long) this.rows * this.columns) / 2);
    }


    /**
     * Check orthogonality of a matrix
     * @return whether the matrix is orthogonal or not
     */
    public boolean isOrthogonal() {
        if (this.rows != this.columns) throw new RuntimeException("Matrix must be a square matrix");

        return this.multiply(this.getTranspose()).isEqual(getIdentityMatrix(this.rows));
    }

    /**
     * method to check equality of two matrices
     * @param mat another matrix
     * @return whether two matrices are equal or not
     */
    private boolean isEqual(Matrix mat) {
        if (mat.rows != this.rows || mat.columns != this.columns)
            throw new RuntimeException("Dimension of the two matrix must be equal");

        return IntStream.range(0, this.rows)
                .mapToObj(i ->
                        IntStream.range(0, this.columns)
                                .allMatch(j -> this.matrix[i][j].equals(mat.matrix[i][j]))
                ).findAny().orElse(false);
    }

    /**
     * check if the matrix is symmetric
     * @return whether matrix is symmetric or not
     */
    public boolean isSymmetric() {
        if (this.rows != this.columns) throw new RuntimeException("Matrix must be a square matrix");

        return this.isEqual(this.getTranspose());
    }

    /**
     * check if the matrix is skew-symmetric
     * @return whether matrix is symmetric or not
     */
    public boolean isSkewSymmetric() {
        if (this.rows != this.columns) throw new RuntimeException("Matrix must be a square matrix");

        return this.getTrace().equals(BigDecimal.ZERO) && this.isEqual(this.getTranspose().multiplyMatrixByScalar(-1));
    }


    /**
     * print matrix (2d array) to console
     * @return matrix string
     */
    public String toString() {
        return Arrays.stream(matrix).map(rows -> Arrays.toString(rows).replaceAll(",", "") + "\n").collect(Collectors.joining());
    }
}