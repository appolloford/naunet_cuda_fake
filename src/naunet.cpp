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

    cusparseCreate(&cusp_handle_);
    cusolverSpCreate(&cusol_handle_);
    cv_y_  = N_VNew_Cuda(MAXNGROUPS * NEQUATIONS);
    cv_a_  = SUNMatrix_cuSparse_NewBlockCSR(MAXNGROUPS, NEQUATIONS, NEQUATIONS,
                                            NNZ, cusp_handle_);
    cv_ls_ = SUNLinSol_cuSolverSp_batchQR(cv_y_, cv_a_, cusol_handle_);
    // abstol = N_VNew_Cuda(neq);
    SUNMatrix_cuSparse_SetFixedPattern(cv_a_, 1);
    InitJac(cv_a_);

    /*  */

    cv_mem_ = CVodeCreate(CV_BDF);

    int flag;
    flag = CVodeInit(cv_mem_, Fex, 0.0, cv_y_);
    if (check_flag(&flag, "CVodeInit", 1)) return 1;
    flag = CVodeSStolerances(cv_mem_, rtol_, atol_);
    if (check_flag(&flag, "CVodeSStolerances", 1)) return 1;
    flag = CVodeSetLinearSolver(cv_mem_, cv_ls_, cv_a_);
    if (check_flag(&flag, "CVodeSetLinearSolver", 1)) return 1;
    flag = CVodeSetJacFn(cv_mem_, Jac);
    if (check_flag(&flag, "CVodeSetJacFn", 1)) return 1;

    // reset the n_vector to empty, maybe not necessary
    /* */

    // N_VDestroy(cv_y_);
    // cv_y_ = N_VNewEmpty_Cuda();

    /* */

    return NAUNET_SUCCESS;
};

int Naunet::Finalize() {
    // N_VDestroy(cv_y_);
    N_VFreeEmpty(cv_y_);
    SUNMatDestroy(cv_a_);
    CVodeFree(&cv_mem_);
    SUNLinSolFree(cv_ls_);
    // delete m_data;

    /* */
    cusparseDestroy(cusp_handle_);
    cusolverSpDestroy(cusol_handle_);
    /*  */

    return NAUNET_SUCCESS;
};

/*  */
// To reset the size of cusparse solver
int Naunet::Reset(int nsystem, double atol, double rtol) {
    n_system_ = nsystem;
    atol_     = atol;
    rtol_     = rtol;

    N_VDestroy(cv_y_);
    SUNMatDestroy(cv_a_);
    SUNLinSolFree(cv_ls_);
    CVodeFree(&cv_mem_);

    cv_y_ = N_VNew_Cuda(nsystem * NEQUATIONS);
    cv_a_ = SUNMatrix_cuSparse_NewBlockCSR(nsystem, NEQUATIONS, NEQUATIONS, NNZ,
                                           cusp_handle_);
    cv_ls_ = SUNLinSol_cuSolverSp_batchQR(cv_y_, cv_a_, cusol_handle_);
    SUNMatrix_cuSparse_SetFixedPattern(cv_a_, 1);
    InitJac(cv_a_);

    cv_mem_ = CVodeCreate(CV_BDF);

    int flag;
    flag = CVodeInit(cv_mem_, Fex, 0.0, cv_y_);
    if (check_flag(&flag, "CVodeInit", 1)) return 1;
    flag = CVodeSStolerances(cv_mem_, rtol_, atol_);
    if (check_flag(&flag, "CVodeSStolerances", 1)) return 1;
    flag = CVodeSetLinearSolver(cv_mem_, cv_ls_, cv_a_);
    if (check_flag(&flag, "CVodeSetLinearSolver", 1)) return 1;
    flag = CVodeSetJacFn(cv_mem_, Jac);
    if (check_flag(&flag, "CVodeSetJacFn", 1)) return 1;

    // reset the n_vector to empty, maybe not necessary
    // N_VDestroy(cv_y_);
    // cv_y_ = N_VNewEmpty_Cuda();

    return NAUNET_SUCCESS;
};
/*  */

int Naunet::Solve(realtype *ab, realtype dt, NaunetData *data) {
    realtype t0 = 0.0;
    int flag;

    /* */

    // This way is too slow
    // realtype *ydata = N_VGetArrayPointer(cv_y_);
    // for (int i=0; i<NEQUATIONS; i++)
    // {
    //     ydata[i] = ab[i];
    // }
    N_VSetHostArrayPointer_Cuda(ab, cv_y_);
    N_VCopyToDevice_Cuda(cv_y_);

    /*  */

#ifdef NAUNET_DEBUG
    /* */
    // sunindextype lrw, liw;
    // N_VSpace_Cuda(cv_y_, &lrw, &liw);
    // printf("NVector space: real-%d, int-%d\n", lrw, liw);
    /*  */
#endif

    flag = CVodeReInit(cv_mem_, 0.0, cv_y_);
    if (check_flag(&flag, "CVodeReInit", 1)) return 1;
    flag = CVodeSetUserData(cv_mem_, data);
    if (check_flag(&flag, "CVodeSetUserData", 1)) return 1;

    flag = CVode(cv_mem_, dt, cv_y_, &t0, CV_NORMAL);

    /* */

    N_VCopyFromDevice_Cuda(cv_y_);
    ab = N_VGetHostArrayPointer_Cuda(cv_y_);

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