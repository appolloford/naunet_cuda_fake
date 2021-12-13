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
__global__ void FexKernel(realtype *y, realtype *ydot, NaunetData *d_udata,
                          int nsystem) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gs   = blockDim.x * gridDim.x;

    for (int cur = tidx; cur < nsystem; cur += gs) {
        int yistart            = cur * NEQUATIONS;
        realtype *y_cur        = y + yistart;
        realtype k[NREACTIONS] = {0.0};
        NaunetData *udata      = &d_udata[cur];

        // clang-format off
        realtype nH = udata->nH;
        realtype Tgas = udata->Tgas;
        
                
        // clang-format on

        EvalRates(k, y_cur, udata);

        // clang-format off
        ydot[yistart + IDX_DI] = 0.0 - k[0]*y_cur[IDX_H2I]*y_cur[IDX_DI] +
            k[1]*y_cur[IDX_HDI]*y_cur[IDX_HI];
        ydot[yistart + IDX_HI] = 0.0 + k[0]*y_cur[IDX_H2I]*y_cur[IDX_DI] -
            k[1]*y_cur[IDX_HDI]*y_cur[IDX_HI];
        ydot[yistart + IDX_H2I] = 0.0 - k[0]*y_cur[IDX_H2I]*y_cur[IDX_DI] +
            k[1]*y_cur[IDX_HDI]*y_cur[IDX_HI];
        ydot[yistart + IDX_HDI] = 0.0 + k[0]*y_cur[IDX_H2I]*y_cur[IDX_DI] -
            k[1]*y_cur[IDX_HDI]*y_cur[IDX_HI];
        
                // clang-format on
    }
}

/* */

int Fex(realtype t, N_Vector u, N_Vector udot, void *user_data) {
    /* */

    cudaStream_t stream = *(NVEC_CUDA_STREAM(u));

    realtype *y         = N_VGetDeviceArrayPointer_Cuda(u);
    realtype *ydot      = N_VGetDeviceArrayPointer_Cuda(udot);
    NaunetData *h_udata = (NaunetData *)user_data;
    NaunetData *d_udata;

    // check the size of system (number of cells/ a batch)
    sunindextype lrw, liw;
    N_VSpace_Cuda(u, &lrw, &liw);
    int nsystem = lrw / NEQUATIONS;

    // copy the user data for each system/cell
    cudaMalloc((void **)&d_udata, sizeof(NaunetData) * nsystem);
    cudaMemcpyAsync(d_udata, h_udata, sizeof(NaunetData) * nsystem,
               cudaMemcpyHostToDevice, stream);
    // cudaDeviceSynchronize();

    unsigned block_size = min(BLOCKSIZE, nsystem);
    unsigned grid_size =
        max(1, min(MAX_NSYSTEMS_PER_STREAM / BLOCKSIZE, nsystem / BLOCKSIZE));
    FexKernel<<<grid_size, block_size, 0, stream>>>(y, ydot, d_udata, nsystem);

    // cudaDeviceSynchronize();
    cudaError_t cuerr = cudaGetLastError();
    if (cuerr != cudaSuccess) {
        fprintf(stderr, ">>> ERROR in fex: cudaGetLastError returned %s\n",
                cudaGetErrorName(cuerr));
        return -1;
    }
    cudaFree(d_udata);

    /* */

    return NAUNET_SUCCESS;
}