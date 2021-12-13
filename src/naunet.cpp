#include <cvode/cvode.h>  // prototypes for CVODE fcts., consts.
/* */
#include <nvector/nvector_cuda.h>
#include <sunlinsol/sunlinsol_cusolversp_batchqr.h>
#include <sunmatrix/sunmatrix_cusparse.h>
/* */
/*  */
#include "naunet.h"
/*  */
#include "naunet_ode.h"

// check_flag function is from the cvDiurnals_ky.c example from the CVODE
// package. Check function return value...
//   opt == 0 means SUNDIALS function allocates memory so check if
//            returned NULL pointer
//   opt == 1 means SUNDIALS function returns a flag so check if
//            flag >= 0
//   opt == 2 means function allocates memory so check if returned
//            NULL pointer
static int check_flag(void *flagvalue, const char *funcname, int opt) {
    int *errflag;

    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if (opt == 0 && flagvalue == NULL) {
        fprintf(stderr,
                "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
                funcname);
        return 1;
    }

    /* Check if flag < 0 */
    else if (opt == 1) {
        errflag = (int *)flagvalue;
        if (*errflag < 0) {
            fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
                    funcname, *errflag);
            return 1;
        }
    }

    /* Check if function returned NULL pointer - no memory allocated */
    else if (opt == 2 && flagvalue == NULL) {
        fprintf(stderr,
                "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
                funcname);
        return 1;
    }

    return 0;
}

Naunet::Naunet(){};

Naunet::~Naunet(){};

int Naunet::Init(int nsystem, double atol, double rtol) {
    n_system_ = nsystem;
    atol_     = atol;
    rtol_     = rtol;

    /* */

    if (nsystem < NSTREAMS ||  nsystem % NSTREAMS != 0) {
        printf("Invalid size of system!");
        return NAUNET_FAIL;
    }

    cudaError_t cuerr;
    int flag;

    for (int i = 0; i < NSTREAMS; i++) {

        cuerr = cudaStreamCreate(&custream_[i]);
        // SUNCudaThreadDirectExecPolicy stream_exec_policy(nsystem / NSTREAMS, custream_[i]);
        // SUNCudaBlockReduceExecPolicy reduce_exec_policy(nsystem / NSTREAMS, 0, custream_[i]);
        stream_exec_policy_[i] = new SUNCudaThreadDirectExecPolicy(nsystem / NSTREAMS, custream_[i]);
        reduce_exec_policy_[i] = new SUNCudaBlockReduceExecPolicy(nsystem / NSTREAMS, 0, custream_[i]);

        cusparseCreate(&cusp_handle_[i]);
        cusolverSpCreate(&cusol_handle_[i]);
        cv_y_[i]  = N_VNew_Cuda(NEQUATIONS * nsystem / NSTREAMS);
        flag = N_VSetKernelExecPolicy_Cuda(cv_y_[i], stream_exec_policy_[i], reduce_exec_policy_[i]);
        if (check_flag(&flag, "N_VSetKernelExecPolicy_Cuda", 0)) return 1;
        cv_a_[i]  = SUNMatrix_cuSparse_NewBlockCSR(nsystem / NSTREAMS, NEQUATIONS, NEQUATIONS,
                                                NNZ, cusp_handle_[i]);
        cv_ls_[i] = SUNLinSol_cuSolverSp_batchQR(cv_y_[i], cv_a_[i], cusol_handle_[i]);
        // abstol = N_VNew_Cuda(neq);
        SUNMatrix_cuSparse_SetFixedPattern(cv_a_[i], 1);
        InitJac(cv_a_[i]);

        cv_mem_[i] = CVodeCreate(CV_BDF);

        flag = CVodeInit(cv_mem_[i], Fex, 0.0, cv_y_[i]);
        if (check_flag(&flag, "CVodeInit", 1)) return 1;
        flag = CVodeSStolerances(cv_mem_[i], rtol_, atol_);
        if (check_flag(&flag, "CVodeSStolerances", 1)) return 1;
        flag = CVodeSetLinearSolver(cv_mem_[i], cv_ls_[i], cv_a_[i]);
        if (check_flag(&flag, "CVodeSetLinearSolver", 1)) return 1;
        flag = CVodeSetJacFn(cv_mem_[i], Jac);
        if (check_flag(&flag, "CVodeSetJacFn", 1)) return 1;

    }

    /*  */

    // reset the n_vector to empty, maybe not necessary
    /* */

    // N_VDestroy(cv_y_);
    // cv_y_ = N_VNewEmpty_Cuda();

    /* */

    return NAUNET_SUCCESS;
};

int Naunet::Finalize() {

    /* */
    for (int i = 0; i < NSTREAMS; i++) {
        N_VFreeEmpty(cv_y_[i]);
        SUNMatDestroy(cv_a_[i]);
        CVodeFree(&cv_mem_[i]);
        SUNLinSolFree(cv_ls_[i]);

        cusparseDestroy(cusp_handle_[i]);
        cusolverSpDestroy(cusol_handle_[i]);
        cudaStreamDestroy(custream_[i]);
    }
    /*  */

    return NAUNET_SUCCESS;
};

/*  */
// To reset the size of cusparse solver
int Naunet::Reset(int nsystem, double atol, double rtol) {

    if (nsystem < NSTREAMS ||  nsystem % NSTREAMS != 0) {
        printf("Invalid size of system!");
        return NAUNET_FAIL;
    }

    n_system_ = nsystem;
    atol_     = atol;
    rtol_     = rtol;

    int flag;

    for (int i = 0; i < NSTREAMS; i++) {
        N_VDestroy(cv_y_[i]);
        SUNMatDestroy(cv_a_[i]);
        SUNLinSolFree(cv_ls_[i]);
        CVodeFree(&cv_mem_[i]);

        // SUNCudaThreadDirectExecPolicy stream_exec_policy(nsystem / NSTREAMS, custream_[i]);
        // SUNCudaBlockReduceExecPolicy reduce_exec_policy(nsystem / NSTREAMS, 0, custream_[i]);

        cv_y_[i] = N_VNew_Cuda(NEQUATIONS * nsystem / NSTREAMS);
        flag = N_VSetKernelExecPolicy_Cuda(cv_y_[i], stream_exec_policy_[i], reduce_exec_policy_[i]);
        if (check_flag(&flag, "N_VSetKernelExecPolicy_Cuda", 0)) return 1;
        cv_a_[i] = SUNMatrix_cuSparse_NewBlockCSR(nsystem / NSTREAMS, NEQUATIONS, NEQUATIONS, NNZ,
                                                  cusp_handle_[i]);
        cv_ls_[i] = SUNLinSol_cuSolverSp_batchQR(cv_y_[i], cv_a_[i], cusol_handle_[i]);
        SUNMatrix_cuSparse_SetFixedPattern(cv_a_[i], 1);
        InitJac(cv_a_[i]);

        cv_mem_[i] = CVodeCreate(CV_BDF);

        flag = CVodeInit(cv_mem_[i], Fex, 0.0, cv_y_[i]);
        if (check_flag(&flag, "CVodeInit", 1)) return 1;
        flag = CVodeSStolerances(cv_mem_[i], rtol_, atol_);
        if (check_flag(&flag, "CVodeSStolerances", 1)) return 1;
        flag = CVodeSetLinearSolver(cv_mem_[i], cv_ls_[i], cv_a_[i]);
        if (check_flag(&flag, "CVodeSetLinearSolver", 1)) return 1;
        flag = CVodeSetJacFn(cv_mem_[i], Jac);
        if (check_flag(&flag, "CVodeSetJacFn", 1)) return 1;

        // reset the n_vector to empty, maybe not necessary
        // N_VDestroy(cv_y_);
        // cv_y_ = N_VNewEmpty_Cuda();
    }

    return NAUNET_SUCCESS;
};
/*  */

int Naunet::Solve(realtype *ab, realtype dt, NaunetData *data) {

    int flag;

    /* */

    for (int i = 0; i < NSTREAMS; i++) {
        realtype t0 = 0.0;

        realtype *local_ab;
        NaunetData *local_data;

        cudaMallocHost((void **)&local_ab, sizeof(realtype) * n_system_ * NEQUATIONS / NSTREAMS);
        cudaMallocHost((void **)&local_data, sizeof(NaunetData) * n_system_ / NSTREAMS);
        for (int idx = 0; idx < n_system_ * NEQUATIONS / NSTREAMS; idx++)
        {
            local_ab[idx] = ab[idx + i * n_system_ * NEQUATIONS / NSTREAMS];
        }
        for (int idx = 0; idx < n_system_ / NSTREAMS; idx++)
        {
            local_data[idx] = data[i * n_system_ / NSTREAMS + idx];
        }

        // ! Bug: I don't know why n_vector does not save the stream_exec_policy and reduce_exec_policy
        N_VSetKernelExecPolicy_Cuda(cv_y_[i], stream_exec_policy_[i], reduce_exec_policy_[i]);

        // This way is too slow
        // realtype *ydata = N_VGetArrayPointer(cv_y_[i]);
        // for (int i = 0; i < NEQUATIONS; i++)
        // {
        //     ydata[i] = ab[i];
        // }
        N_VSetHostArrayPointer_Cuda(local_ab, cv_y_[i]);
        N_VCopyToDevice_Cuda(cv_y_[i]);

#ifdef NAUNET_DEBUG
        // sunindextype lrw, liw;
        // N_VSpace_Cuda(cv_y_[i], &lrw, &liw);
        // printf("NVector space: real-%d, int-%d\n", lrw, liw);
#endif

        flag = CVodeReInit(cv_mem_[i], 0.0, cv_y_[i]);
        if (check_flag(&flag, "CVodeReInit", 1)) return 1;
        flag = CVodeSetUserData(cv_mem_[i], local_data);
        if (check_flag(&flag, "CVodeSetUserData", 1)) return 1;

        flag = CVode(cv_mem_[i], dt, cv_y_[i], &t0, CV_NORMAL);

        N_VCopyFromDevice_Cuda(cv_y_[i]);
        local_ab = N_VGetHostArrayPointer_Cuda(cv_y_[i]);
        for (int idx = 0; idx < n_system_ * NEQUATIONS / NSTREAMS; idx++)
        {
            ab[idx + i * n_system_ * NEQUATIONS / NSTREAMS] = local_ab[idx];
        }

        cudaDeviceSynchronize();

        cudaFreeHost(local_ab);
        cudaFreeHost(local_data);
    }

    /* */

    return NAUNET_SUCCESS;
};

#ifdef PYMODULE
py::array_t<realtype> Naunet::PyWrapSolve(py::array_t<realtype> arr,
                                          realtype dt, NaunetData *data) {
    py::buffer_info info = arr.request();
    realtype *ab         = static_cast<realtype *>(info.ptr);

    Solve(ab, dt, data);

    return py::array_t<realtype>(info.shape, ab);
}
#endif