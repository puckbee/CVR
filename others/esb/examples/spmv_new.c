/*
********************************************************************************
*   Copyright(C) 2013-2014 Intel Corporation. All Rights Reserved.
*   
*   The source code, information  and  material ("Material") contained herein is
*   owned  by Intel Corporation or its suppliers or licensors, and title to such
*   Material remains  with Intel Corporation  or its suppliers or licensors. The
*   Material  contains proprietary information  of  Intel or  its  suppliers and
*   licensors. The  Material is protected by worldwide copyright laws and treaty
*   provisions. No  part  of  the  Material  may  be  used,  copied, reproduced,
*   modified, published, uploaded, posted, transmitted, distributed or disclosed
*   in any way  without Intel's  prior  express written  permission. No  license
*   under  any patent, copyright  or  other intellectual property rights  in the
*   Material  is  granted  to  or  conferred  upon  you,  either  expressly,  by
*   implication, inducement,  estoppel or  otherwise.  Any  license  under  such
*   intellectual  property  rights must  be express  and  approved  by  Intel in
*   writing.
*   
*   *Third Party trademarks are the property of their respective owners.
*   
*   Unless otherwise  agreed  by Intel  in writing, you may not remove  or alter
*   this  notice or  any other notice embedded  in Materials by Intel or Intel's
*   suppliers or licensors in any way.
*
********************************************************************************
*   Content : Intel(R) MKL SpMV Format Prototype Package C native example
*
********************************************************************************
*/
/*
!
! Consider the matrix A (see 'Sparse Storage Formats for Sparse BLAS Level 2
! and Level 3 in the Intel MKL Reference Manual')
!
!                 |   1       -1      0   -3     0   |
!                 |  -2        5      0    0     0   |
!   A    =        |   0        0      4    6     4   |,
!                 |  -4        0      2    7     0   |
!                 |   0        8      0    0    -5   |
!
!  The matrix A is represented in a zero-based compressed sparse row storage
!  scheme with three arrays (see 'Sparse Matrix Storage Schemes' in the 
!  Intel MKL Reference Manual) as follows:
!
!         values  = ( 1 -1 -3 -2 5 4 6 4 -4 2 7 8 -5 )
!         columns = ( 0 1 3 0 1 2 3 4 0 2 3 1 4 )
!         rowIndex = ( 0  3  5  8  11 13 )
!
!  The test performs the following operations :
!
!       The code computes A*S = F using sparseDesbmv and sparseDcsrmv 
!          where A is a general sparse matrix and S and F are vectors.
!
!*******************************************************************************
*/
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "spmv_interface.h"

int main() {
    //*******************************************************************************
    //     Declaration and initialization of parameters for sparse representation of
    //     the matrix A in the compressed sparse row format:
    //*******************************************************************************
#define M 5
#define N 5
#define NNZ 13
    int m = M, n = N, nnz = NNZ;
    //*******************************************************************************
    //    Sparse representation of the matrix A
    //*******************************************************************************
    double csrVal[NNZ]    = { 1.0,  -1.0,      -3.0, 
                              -2.0,  5.0, 
                                          4.0,  6.0,  4.0, 
                              -4.0,       2.0,  7.0, 
                                     8.0,            -5.0 };
    int    csrColInd[NNZ] = { 0, 1,    3, 
                              0, 1, 
                                    2, 3, 4, 
                              0,    2, 3, 
                                 1,       4 };
    int    csrRowPtr[M+1] = { 0, 3, 5, 8, 11, 13 };
    // Matrix descriptor
    sparseMatDescr_t    descrA;
    // CSR matrix representation 
    sparseCSRMatrix_t   csrA;
    // ESB matrix representation
    sparseESBMatrix_t   esbA;
    //*******************************************************************************
    //    Declaration of local variables:
    //*******************************************************************************
    double      x[M]  = { 1.0, 5.0, 1.0, 4.0, 1.0};
    double      y[M]  = { 0.0, 0.0, 0.0, 0.0, 0.0};
    double      alpha = 1.0, beta = 0.0;
    int         i;

    printf( "\n EXAMPLE PROGRAM FOR sparseDcsrmv and sparseDesbmv \n" );
    printf( "---------------------------------------------------\n" );
    printf( "\n" );
    printf( "   INPUT DATA FOR sparseDcsrmv    \n" );
    printf( "   WITH GENERAL SPARSE MATRIX     \n" );
    printf( "   ALPHA = %4.1f  BETA = %4.1f    \n", alpha, beta );
    printf( "   SPARSE_OPERATION_NON_TRANSPOSE \n" );
    printf( "   Input vector                   \n" );
    for ( i = 0; i < m; i++ )
    {
        printf( "%7.1f\n", x[i] );
    };

    // Create CSR matrix with static workload balancing algorithm
    sparseCreateCSRMatrix ( &csrA, SPARSE_SCHEDULE_STATIC );

    // Create matrix descriptor
    sparseCreateMatDescr ( &descrA );

    // Analyze input matrix and create its internal representation in the 
    // csrA structure optimized for static workload balancing 
    sparseDcsr2csr ( m, n, descrA, csrVal, csrRowPtr, csrColInd, csrA );

    // Compute y = alpha * A * x + beta * y
    sparseDcsrmv ( SPARSE_OPERATION_NON_TRANSPOSE, &alpha, csrA, x, &beta, y );

    // Release internal representation of CSR matrix
    sparseDestroyCSRMatrix ( csrA );

    printf( "                                \n" );
    printf( "   OUTPUT DATA FOR sparseDcsrmv \n" );
    for ( i = 0; i < m; i++ )
    {
        printf( "%7.1f\n", y[i] );
    };

    printf("-----------------------------------------------\n" );
    printf("   INPUT DATA FOR sparseDesbmv    \n" );
    printf("   WITH GENERAL SPARSE MATRIX     \n" );
    printf("   ALPHA = %4.1f  BETA = %4.1f    \n", alpha, beta );
    printf("   SPARSE_OPERATION_NON_TRANSPOSE \n" );
    printf("   Input vector                   \n" );

    for ( i = 0; i < m; i++ )
    {
        printf( "%7.1f\n", x[i] );
    };

    // Create ESB matrix with static workload balancing algorithm
    sparseCreateESBMatrix ( &esbA, SPARSE_SCHEDULE_STATIC );

    // Analyze input CSR matrix and create its internal ESB representation in the
    // esbA structure optimized for static workload balancing 
    sparseDcsr2esb ( m, n, descrA, csrVal, csrRowPtr, csrColInd, esbA );

    // Compute y = alpha * A * x + beta * y
    sparseDesbmv ( SPARSE_OPERATION_NON_TRANSPOSE, &alpha, esbA, x, &beta, y );

    // Release internal representation of ESB matrix
    sparseDestroyESBMatrix ( esbA );

    printf( "                               \n" );
    printf( "   OUTPUT DATA FOR sparseDesbmv\n" );
    for ( i = 0; i < m; i++ )
    {
        printf( "%7.1f\n", y[i] );
    };

    printf( "---------------------------------------------------\n" );
    return 0;
}