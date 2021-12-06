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

#define IJth(A, i, j) SM_ELEMENT_D(A, i, j)

// clang-format off
__device__ int EvalRates(realtype *k, realtype *y, NaunetData *u_data) {

    realtype nH = u_data->nH;
    realtype Tgas = u_data->Tgas;
    
    
    // clang-format on

    // Some variable definitions from krome
    realtype Te      = Tgas * 8.617343e-5;            // Tgas in eV (eV)
    realtype lnTe    = log(Te);                       // ln of Te (#)
    realtype T32     = Tgas * 0.0033333333333333335;  // Tgas/(300 K) (#)
    realtype invT    = 1.0 / Tgas;                    // inverse of T (1/K)
    realtype invTe   = 1.0 / Te;                      // inverse of T (1/eV)
    realtype sqrTgas = sqrt(Tgas);  // Tgas rootsquare (K**0.5)

    // reaaction rate (k) of each reaction
    // clang-format off
    k[0] = 3e-10;
        
    k[1] = 5e-11;
        
    
        // clang-format on

    return NAUNET_SUCCESS;
}

/* */
int InitJac(SUNMatrix jmatrix) {
    int rowptrs[NEQUATIONS + 1], colvals[NNZ];

    // Zero out the Jacobian
    SUNMatZero(jmatrix);

    // clang-format off
    // number of non-zero elements in each row
    rowptrs[0] = 0;
    rowptrs[1] = 4;
    rowptrs[2] = 8;
    rowptrs[3] = 12;
    rowptrs[4] = 16;
    
    // the column index of non-zero elements
    colvals[0] = 0;
    colvals[1] = 1;
    colvals[2] = 2;
    colvals[3] = 3;
    colvals[4] = 0;
    colvals[5] = 1;
    colvals[6] = 2;
    colvals[7] = 3;
    colvals[8] = 0;
    colvals[9] = 1;
    colvals[10] = 2;
    colvals[11] = 3;
    colvals[12] = 0;
    colvals[13] = 1;
    colvals[14] = 2;
    colvals[15] = 3;
    
    // clang-format on

    // copy rowptrs, colvals to the device
    SUNMatrix_cuSparse_CopyToDevice(jmatrix, NULL, rowptrs, colvals);
    cudaDeviceSynchronize();

    return NAUNET_SUCCESS;
}

__global__ void FexKernel(realtype *y, realtype *ydot, NaunetData *d_udata,
                          int nsystem) {
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gs   = blockDim.x * gridDim.x;

    // clang-format off
    realtype nH = d_udata->nH;
    realtype Tgas = d_udata->Tgas;
    
        
    // clang-format on

    for (int cur = tidx; cur < nsystem; cur += gs) {
        int yistart            = cur * NEQUATIONS;
        realtype *y_cur        = y + yistart;
        realtype k[NREACTIONS] = {0.0};
        NaunetData *udata      = &d_udata[cur];

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

int Fex(realtype t, N_Vector u, N_Vector udot, void *user_data) {
    /* */

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
    cudaMemcpy(d_udata, h_udata, sizeof(NaunetData) * nsystem,
               cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    unsigned block_size = min(BLOCKSIZE, nsystem);
    unsigned grid_size =
        max(1, min(MAXNGROUPS / BLOCKSIZE, nsystem / BLOCKSIZE));
    FexKernel<<<grid_size, block_size>>>(y, ydot, d_udata, nsystem);

    cudaDeviceSynchronize();
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

int Jac(realtype t, N_Vector u, N_Vector fu, SUNMatrix jmatrix, void *user_data,
        N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
    /* */
    realtype *y         = N_VGetDeviceArrayPointer_Cuda(u);
    realtype *data      = SUNMatrix_cuSparse_Data(jmatrix);
    NaunetData *h_udata = (NaunetData *)user_data;
    NaunetData *d_udata;

    int nsystem = SUNMatrix_cuSparse_NumBlocks(jmatrix);

    cudaMalloc((void **)&d_udata, sizeof(NaunetData) * nsystem);
    cudaMemcpy(d_udata, h_udata, sizeof(NaunetData) * nsystem,
               cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    unsigned block_size = min(BLOCKSIZE, nsystem);
    unsigned grid_size =
        max(1, min(MAXNGROUPS / BLOCKSIZE, nsystem / BLOCKSIZE));
    JacKernel<<<grid_size, block_size>>>(y, data, d_udata, nsystem);

    cudaDeviceSynchronize();
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