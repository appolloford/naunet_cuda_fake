#ifndef __NAUNET_MACROS_H__
#define __NAUNET_MACROS_H__

// clang-format off
#define NAUNET_SUCCESS 0
#define NAUNET_FAIL 1

#define NSPECIES 4
#define NEQUATIONS 4
#define NREACTIONS 2
#define NNZ 16
#define USE_CUDA
#define MAX_NSYSTEMS 4096
#define NSTREAMS 16
#define MAX_NSYSTEMS_PER_STREAM (MAX_NSYSTEMS/NSTREAMS)
#define BLOCKSIZE 64

#define IDX_DI 0
#define IDX_HI 1
#define IDX_H2I 2
#define IDX_HDI 3
#endif