#ifndef CONVOLUTION_MATRIX_H
#define CONVOLUTION_MATRIX_H


class Matrix {
    private:
        double *data;
        std::size_t width;

    public:
        ~Matrix() {
            delete[] data;
        }

        Matrix(double data[], std::size_t width) : data(data), width(width) {}

        double *getData() {
            return data;
        }

        std::size_t getWidth() const {
            return width;
        }

        std::size_t size() const {
            return getWidth() * getWidth();
        }
};

#endif //CONVOLUTION_MATRIX_H
