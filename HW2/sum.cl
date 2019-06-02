#define SWAP(a,b) {__local double *tmp=a; a=b; b=tmp;}

__kernel void scan_hillis_steele(__global double *input, __global double *output, int size,
                                 __local double *a, __local double *b) {
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);
    uint block_size = get_local_size(0);

    if (size > gid) {
        a[lid] = b[lid] = input[gid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint s = 1; s < block_size; s *= 2) {
        if(lid > (s-1)) {
            b[lid] = a[lid] + a[lid-s];
        }
        else {
            b[lid] = a[lid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        SWAP(a,b);
    }

    if (size > gid) {
        output[gid] = a[lid];
    }
}

__kernel void copy_groups_sum(__global double *input, __global double *output, int size) {
    uint gid = get_global_id(0);
    uint block_size = get_local_size(0);

    if (size > gid) {
        output[gid] = input[(gid + 1) * block_size - 1];
    }
}

__kernel void add_partial_sums(__global double *input, __global double *sums, int size) {
    uint gid = get_global_id(0);
    uint group = get_group_id(0);
    uint block_size = get_local_size(0);

    if (size > gid && gid >= block_size) {
        input[gid] += sums[group - 1];
    }
}