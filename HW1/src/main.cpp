#include "../include/Convolution.h"
#include <fstream>
#include <iostream>


using namespace std;

void read_matrix(double* matrix, int size, istream &input) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            input >> matrix[i * size + j];
        }
    }
}

int main() {
    ifstream input;
    input.open("input.txt");

    int size_a, size_b;
    input >> size_a >> size_b;
    auto *a = new double[size_a * size_a];
    read_matrix(a, size_a, input);
    auto *b = new double[size_b * size_b];
    read_matrix(b, size_b, input);
    input.close();

    Matrix A(a, size_a);
    Matrix B(b, size_b);
    Convolution convolution;
    Matrix &C = *convolution.conv(A, B);

    ofstream output;
    output.open("output.txt");
    for (int i = 0; i < C.getWidth(); ++i) {
        for (int j = 0; j < C.getWidth(); ++j) {
            output << C.getData()[i * C.getWidth() + j] << ' ';
        }
        output << "\n";
    }
    output.close();
    delete &C;
}

