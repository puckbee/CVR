#*******************************************************************************
#   Copyright(C) 2013 Intel Corporation. All Rights Reserved.
#
#   The source code, information  and  material ("Material") contained herein is
#   owned  by Intel Corporation or its suppliers or licensors, and title to such
#   Material remains  with Intel Corporation  or its suppliers or licensors. The
#   Material  contains proprietary information  of  Intel or  its  suppliers and
#   licensors. The  Material is protected by worldwide copyright laws and treaty
#   provisions. No  part  of  the  Material  may  be  used,  copied, reproduced,
#   modified, published, uploaded, posted, transmitted, distributed or disclosed
#   in any way  without Intel's  prior  express written  permission. No  license
#   under  any patent, copyright  or  other intellectual property rights  in the
#   Material  is  granted  to  or  conferred  upon  you,  either  expressly,  by
#   implication, inducement,  estoppel or  otherwise.  Any  license  under  such
#   intellectual  property  rights must  be express  and  approved  by  Intel in
#   writing.
#
#   *Third Party trademarks are the property of their respective owners.
#
#   Unless otherwise  agreed  by Intel  in writing, you may not remove  or alter
#   this  notice or  any other notice embedded  in Materials by Intel or Intel's
#   suppliers or licensors in any way.
#
#*******************************************************************************
#
#   Content :  Double-precision performance benchmark for
#              Intel(R) MKL SpMV Format Prototype Package, 
#              version 0.2, run script
#
#   usage (on host):  ./run_host.sh 
#
#   This script sets proper environment parameters for ./run_lnx_minicom.sh
#
#*******************************************************************************

#!/bin/bash

# Path to the matrices in Matrix Market format
export MATRIX_PATH=./matrices
# List of matrices for benchmarking
#export MATRICES="G41.mtx"
#export MATRICES="google"
export MATRICES=$1
# Name of the Intel Xeon Phi coprocessor
export DEVICE=mic0
# Folder on the coprocessor for uploading matrices and executable
export BASEDIR=/tmp

# Number of threads on the coprocessor for benchmarking
if [ -z "$MIC_OMP_NUM_THREADS" ]; then export MIC_OMP_NUM_THREADS=240 ; fi

# Setting KMP_AFFINITY based on the number of threads
export MIC_KMP_AFFINITY=explicit,proclist=[1-$(expr $MIC_OMP_NUM_THREADS - 4):1,0,$(expr $MIC_OMP_NUM_THREADS - 3),$(expr $MIC_OMP_NUM_THREADS - 2),$(expr $MIC_OMP_NUM_THREADS - 1)],granularity=fine

# Upload all data and run benchmark on the coprocessor
./run_lnx_minicom.sh mm_bench.exe
