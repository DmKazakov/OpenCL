#ifndef CONVOLUTION_CONVOLUTION_H
#define CONVOLUTION_CONVOLUTION_H
#define __CL_ENABLE_EXCEPTIONS

#include "cl.hpp"
#include "Matrix.h"
#include <fstream>


class Convolution {
    private:
        size_t const BLOCK_SIZE = 16;
        cl::Context context;
        cl::Program program;
        cl::CommandQueue queue;

    public:
        Convolution();

        Matrix *conv(Matrix &A, Matrix &B);
};

#endif //CONVOLUTION_CONVOLUTION_H
