#ifndef CONVOLUTION_CONVOLUTION_H
#define CONVOLUTION_CONVOLUTION_H
#define __CL_ENABLE_EXCEPTIONS

#include "cl.hpp"
#include <fstream>


class Scan {
    private:
        size_t const BLOCK_SIZE = 256;
        cl::Context context;
        cl::Program program;
        cl::CommandQueue queue;

    public:
        Scan();

        void hillis_steele(std::vector<double> &input, std::vector<double> &output);

    private:
        void copy_groups_sum(std::vector<double> &input, std::vector<double> &output);

        size_t round_to_block_size_dividend(size_t size);

        void add_partial_sums(std::vector<double> &input, std::vector<double> &sums);
};

#endif //CONVOLUTION_CONVOLUTION_H
