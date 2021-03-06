package com.moulidutta;

/*
    TestMatrix.java
    Used to demonstrate the methods defined in Matrix class
 */
import java.math.BigDecimal;
import java.util.Scanner;

public class MatrixTest {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.println("Enter row of first matrix");
        int row = sc.nextInt();
        System.out.println("Enter column of first matrix");
        int column = sc.nextInt();

        Matrix matrixA = new Matrix(row, column);

        System.out.println("Enter elements of the first matrix row wise.");
        matrixA.initMatrix(sc);

        System.out.println("Enter row of second matrix");
        int row2 = sc.nextInt();
        System.out.println("Enter column of second matrix");
        int column2 = sc.nextInt();

        Matrix matrixB = new Matrix(row2, column2);
        System.out.println("Enter lower bound");
        int lower = sc.nextInt();
        System.out.println("Enter upper bound");
        int upper = sc.nextInt();
        matrixB.generateRandomMatrix(lower, upper);

        Matrix add = matrixA.add(matrixB);
        Matrix subtract = matrixA.subtract(matrixB);
        Matrix multiply = matrixA.multiply(matrixB);
        Matrix inverseA = matrixA.getInverse();
        Matrix transposeA = matrixA.getTranspose();

        System.out.println("Enter scalar multiplier.");
        int scalar = sc.nextInt();
        Matrix matrixByScalar = matrixA.multiplyMatrixByScalar(scalar);

        System.out.println("Enter exponent.");
        int exponent = sc.nextInt();
        Matrix exponential = matrixA.exponential(exponent);

        BigDecimal determinantA = matrixA.getDeterminant();
        BigDecimal trace = matrixA.getTrace();
        BigDecimal normal = matrixA.getNormal();
        long rank = matrixA.getRank();

        boolean sparseMatrix = matrixA.isSparseMatrix();
        boolean orthogonal = matrixA.isOrthogonal();
        boolean symmetric = matrixA.isSymmetric();
        boolean skewSymmetric = matrixA.isSkewSymmetric();

        System.out.println(
                "Matrix A\n" + matrixA +
                "\nmatrix B\n" + matrixB +
                "\naddition\n" + add +
                "\nsubtraction\n" + subtract +
                "\nmultiplication\n" + multiply +
                "\ntranspose\n" + transposeA +
                "\ninverse\n" + inverseA +
                "\nexponential\n" + exponential +
                "\nscalar multiply\n" + matrixByScalar +
                "\ndeterminant: " + determinantA +
                "\nrank: " + rank +
                "\ntrace: " + trace +
                "\nnormal: " + normal +
                "\nIs sparse matrix: " + sparseMatrix +
                "\nIs orthogonal: " + orthogonal +
                "\nIs symmetric: " + symmetric +
                "\nIs skew-symmetric: " + skewSymmetric);

    }
}
