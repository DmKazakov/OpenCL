__kernel void convolution(__global double *A, int a_width,
						  __global double *B, int b_width,
						  __global double *C) {
	int i = get_global_id(0);
	int j = get_global_id(1);
	if (i >= a_width || j >= a_width) {
		return;
	}

	double result = 0;
	int HM = (b_width - 1) / 2;
	for(int k = max(-i, -HM); k <= min(a_width - 1 - i, HM); k++) {
		for (int l = max(-j, -HM); l <= min(a_width - 1 - j, HM); ++l) {
			result += A[(i + k) * a_width + j + l] * B[(k + HM) * b_width + l + HM];
		}
	}
	C[i * a_width + j] = result;
}

