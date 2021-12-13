#include <math.h>
/* */
#include <nvector/nvector_cuda.h>
#include <sunmatrix/sunmatrix_cusparse.h>
/* */
/*  */
#include "naunet_ode.h"
/*  */
#include "naunet_constants.h"
#include "naunet_macros.h"
#include "naunet_physics.h"

#define IJth(A, i, j)        SM_ELEMENT_D(A, i, j)
#define NVEC_CUDA_CONTENT(x) ((N_VectorContent_Cuda)(x->content))
#define NVEC_CUDA_STREAM(x)  (NVEC_CUDA_CONTENT(x)->stream_exec_policy->stream())

/* */
int InitJac(SUNMatrix jmatrix) {
    // Zero out the Jacobian
    SUNMatZero(jmatrix);

    // clang-format off
    // number of non-zero elements in each row
    int rowptrs[NEQUATIONS + 1] = { 
        0, 4, 8, 12, 16
    };

    // the column index of non-zero elements
    int colvals[NNZ] = {
        0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3
    };

    // clang-format on

    // copy rowptrs, colvals to the device
    SUNMatrix_cuSparse_CopyToDevice(jmatrix, NULL, rowptrs, colvals);
    cudaDeviceSynchronize();

    return NAUNET_SUCCESS;
}

__global__ void JacKernel(realtype *y, realtype *data, NaunetData *d_udata,
                          int nsystem) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gs   = blockDim.x * gridDim.x;

    for (int cur = tidx; cur < nsystem; cur += gs) {
        int yistart            = cur * NEQUATIONS;
        int jistart            = cur * NNZ;
        realtype *y_cur        = y + yistart;
        realtype k[NREACTIONS] = {0.0};
        NaunetData *udata      = &d_udata[cur];

        // clang-format off
                
        // clang-format on

        EvalRates(k, y_cur, udata);

        // clang-format off
        data[jistart + 0] = 0.0 - k[0]*y_cur[IDX_H2I];
        data[jistart + 1] = 0.0 + k[1]*y_cur[IDX_HDI];
        data[jistart + 2] = 0.0 - k[0]*y_cur[IDX_DI];
        data[jistart + 3] = 0.0 + k[1]*y_cur[IDX_HI];
        data[jistart + 4] = 0.0 + k[0]*y_cur[IDX_H2I];
        data[jistart + 5] = 0.0 - k[1]*y_cur[IDX_HDI];
        data[jistart + 6] = 0.0 + k[0]*y_cur[IDX_DI];
        data[jistart + 7] = 0.0 - k[1]*y_cur[IDX_HI];
        data[jistart + 8] = 0.0 - k[0]*y_cur[IDX_H2I];
        data[jistart + 9] = 0.0 + k[1]*y_cur[IDX_HDI];
        data[jistart + 10] = 0.0 - k[0]*y_cur[IDX_DI];
        data[jistart + 11] = 0.0 + k[1]*y_cur[IDX_HI];
        data[jistart + 12] = 0.0 + k[0]*y_cur[IDX_H2I];
        data[jistart + 13] = 0.0 - k[1]*y_cur[IDX_HDI];
        data[jistart + 14] = 0.0 + k[0]*y_cur[IDX_DI];
        data[jistart + 15] = 0.0 - k[1]*y_cur[IDX_HI];
                // clang-format on
    }
}
/* */

int Jac(realtype t, N_Vector u, N_Vector fu, SUNMatrix jmatrix, void *user_data,
        N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
    /* */

    cudaStream_t stream = *(NVEC_CUDA_STREAM(u));

    realtype *y         = N_VGetDeviceArrayPointer_Cuda(u);
    realtype *data      = SUNMatrix_cuSparse_Data(jmatrix);
    NaunetData *h_udata = (NaunetData *)user_data;
    NaunetData *d_udata;

    int nsystem = SUNMatrix_cuSparse_NumBlocks(jmatrix);

    cudaMalloc((void **)&d_udata, sizeof(NaunetData) * nsystem);
    cudaMemcpyAsync(d_udata, h_udata, sizeof(NaunetData) * nsystem,
               cudaMemcpyHostToDevice, stream);
    // cudaStreamSynchronize();
    // cudaDeviceSynchronize();

    unsigned block_size = min(BLOCKSIZE, nsystem);
    unsigned grid_size =
        max(1, min(MAX_NSYSTEMS_PER_STREAM / BLOCKSIZE, nsystem / BLOCKSIZE));
    JacKernel<<<grid_size, block_size, 0, stream>>>(y, data, d_udata, nsystem);

    // cudaStreamSynchronize();
    // cudaDeviceSynchronize();
    cudaError_t cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess) {
        fprintf(stderr, ">>> ERROR in jac: cudaGetLastError returned %s\n",
                cudaGetErrorName(cuerr));
        return -1;
    }
    cudaFree(d_udata);

    /* */

    return NAUNET_SUCCESS;
}