#include "../include/Convolution.h"


Convolution::Convolution() {
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;

    cl::Platform::get(&platforms);
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

    context = cl::Context(devices);
    queue = cl::CommandQueue(context, devices[0]);

    std::ifstream cl_file("conv.cl");
    std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(cl_string.c_str(), cl_string.length() + 1));
    program = cl::Program(context, source);
    program.build(devices, ("-D BLOCK_SIZE=" + std::to_string(BLOCK_SIZE)).c_str());
}

Matrix *Convolution::conv(Matrix &A, Matrix &B) {
    auto *C = new double[A.size()];
    cl::Buffer dev_a(context, CL_MEM_READ_ONLY, sizeof(double) * A.size());
    cl::Buffer dev_b(context, CL_MEM_READ_ONLY, sizeof(double) * B.size());
    cl::Buffer dev_c(context, CL_MEM_WRITE_ONLY, sizeof(double) * A.size());

    queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(double) * A.size(), A.getData());
    queue.enqueueWriteBuffer(dev_b, CL_TRUE, 0, sizeof(double) * B.size(), B.getData());

    size_t group_size;
    if (A.getWidth() % BLOCK_SIZE == 0) {
        group_size = A.getWidth();
    } else {
        group_size = (A.getWidth() / BLOCK_SIZE + 1) * BLOCK_SIZE;
    }

    cl::Kernel kernel(program, "convolution");
    cl::KernelFunctor convolution(kernel, queue, cl::NullRange, cl::NDRange(group_size, group_size),
                                  cl::NDRange(BLOCK_SIZE, BLOCK_SIZE));
    convolution(dev_a, (int) A.getWidth(), dev_b, (int) B.getWidth(), dev_c);
    queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, sizeof(double) * A.size(), C);

    return new Matrix(C, A.getWidth());
}
