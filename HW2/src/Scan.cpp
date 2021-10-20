#include "../include/Scan.h"


Scan::Scan() {
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;

    cl::Platform::get(&platforms);
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

    context = cl::Context(devices);
    queue = cl::CommandQueue(context, devices[0]);

    std::ifstream cl_file("sum.cl");
    std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(cl_string.c_str(), cl_string.length() + 1));
    program = cl::Program(context, source);
    program.build(devices);
}

void Scan::hillis_steele(std::vector<double> &input, std::vector<double> &output) {
    cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(double) * input.size());
    cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, sizeof(double) * output.size());
    queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(double) * input.size(), &input[0]);
    size_t size = round_to_block_size_dividend(input.size());

    cl::Kernel kernel(program, "scan_hillis_steele");
    cl::KernelFunctor scan_hillis_steele(kernel, queue, cl::NullRange, cl::NDRange(size), cl::NDRange(BLOCK_SIZE));
    scan_hillis_steele(dev_input, dev_output, input.size(), cl::__local(sizeof(double) * BLOCK_SIZE), cl::__local(sizeof(double) * BLOCK_SIZE));
    queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(double) * output.size(), &output[0]);

    if (input.size() > BLOCK_SIZE) {
        std::vector<double> groups_sum(output.size() / BLOCK_SIZE);
        copy_groups_sum(output, groups_sum);
        std::vector<double> groups_partial_sum(output.size() / BLOCK_SIZE);
        hillis_steele(groups_sum, groups_partial_sum);
        add_partial_sums(output, groups_partial_sum);
    }
}

void Scan::copy_groups_sum(std::vector<double> &input, std::vector<double> &output) {
    cl::Buffer dev_input(context, CL_MEM_READ_ONLY, sizeof(double) * input.size());
    cl::Buffer dev_output(context, CL_MEM_WRITE_ONLY, sizeof(double) * output.size());
    queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(double) * input.size(), &input[0]);
    size_t size = round_to_block_size_dividend(output.size());

    cl::Kernel kernel(program, "copy_groups_sum");
    cl::KernelFunctor copy_groups_sum(kernel, queue, cl::NullRange, cl::NDRange(size), cl::NDRange(BLOCK_SIZE));
    copy_groups_sum(dev_input, dev_output, output.size());
    queue.enqueueReadBuffer(dev_output, CL_TRUE, 0, sizeof(double) * output.size(), &output[0]);
}

void Scan::add_partial_sums(std::vector<double> &input, std::vector<double> &sums) {
    cl::Buffer dev_input(context, CL_MEM_READ_WRITE, sizeof(double) * input.size());
    cl::Buffer dev_sums(context, CL_MEM_READ_ONLY, sizeof(double) * sums.size());
    queue.enqueueWriteBuffer(dev_input, CL_TRUE, 0, sizeof(double) * input.size(), &input[0]);
    queue.enqueueWriteBuffer(dev_sums, CL_TRUE, 0, sizeof(double) * sums.size(), &sums[0]);
    size_t size = round_to_block_size_dividend(input.size());

    cl::Kernel kernel(program, "add_partial_sums");
    cl::KernelFunctor add_partial_sums(kernel, queue, cl::NullRange, cl::NDRange(size), cl::NDRange(BLOCK_SIZE));
    add_partial_sums(dev_input, dev_sums, input.size());
    queue.enqueueReadBuffer(dev_input, CL_TRUE, 0, sizeof(double) * input.size(), &input[0]);
}

size_t Scan::round_to_block_size_dividend(size_t size) {
    size_t rounded;
    if (size % BLOCK_SIZE == 0) {
        rounded = size;
    } else {
        rounded = (size / BLOCK_SIZE + 1) * BLOCK_SIZE;
    }
    return rounded;
}
