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