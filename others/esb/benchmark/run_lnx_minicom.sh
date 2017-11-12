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
#   usage (on host):  ./run_lnx_minicom.sh [executable_name] 
#
#*******************************************************************************

# Default path to the matrices

MATRIX_PATH_DEF=./matrices

# Default output folder for performance results

OUT_PATH_DEF=./results
OUT_PREFIX=benchSpMV
LIBIOMP5=$MKLROOT/../compiler/lib/mic/libiomp5.so

if [ \! -f "$LIBIOMP5" ]; then
  LIBIOMP5=../lib/mic/libiomp5.so
fi

echo "libiomp5 to be used: [$LIBIOMP5]"

if [ -z "$1" ]; then
        echo ERROR: Parameter is required: name of executable file for testing
        exit 1
fi
if [ \! -f "$1" ]; then
        echo ERROR: Executable $1 does not exist
        exit 2
fi

# Intel Xeon Phi coprocessor native parameters

knc_basedir=$BASEDIR
exefile=a.out

if [ -z "$OUT_PATH" ]; then
  OUT_PATH=$OUT_PATH_DEF
fi

echo "Output path: $OUT_PATH"

if [ \! -d "$OUT_PATH" ]; then
  mkdir $OUT_PATH
fi

if [ -z "$MATRIX_PATH" ]; then
  MATRIX_PATH=$MATRIX_PATH_DEF
fi

echo "Matrix path: $MATRIX_PATH"

if [ -z "$MATRICES" ]; then
  echo "MATRICES environment variable is not set"
  exit 3
else
  MATRIX_LIST=$MATRICES
  echo "MATRICES environment variable is used for testing"
fi

echo "Testing matrices: [$MATRIX_LIST]"

if [ -z "$ENVIRONMENT_MIC" ]; then
  ENVIRONMENT_MIC="export OMP_NUM_THREADS=$MIC_OMP_NUM_THREADS;export KMP_AFFINITY=$MIC_KMP_AFFINITY"
fi

echo "Environment: [$ENVIRONMENT_MIC]"

# Uploading executable to the coprocessor
scp "$1" "${DEVICE}:${knc_basedir}/${exefile}"
echo scp "$1" "${DEVICE}:${knc_basedir}/${exefile}"
ssh "${DEVICE}" "chmod +x ${knc_basedir}/${exefile}"

scp "$LIBIOMP5" "${DEVICE}:${knc_basedir}/libiomp5.so"
echo scp "$LIBIOMP5" "${DEVICE}:${knc_basedir}/libiomp5.so"

for matr in $MATRIX_LIST ; do

# Uploading matrix to the coprocessor
scp "${MATRIX_PATH}/${matr}" "${DEVICE}:${knc_basedir}/${matr}"
echo scp "${MATRIX_PATH}/${matr}" "${DEVICE}:${knc_basedir}/${matr}"

infile=$knc_basedir/${matr}
errfile=${matr}.err
resfile=${matr}.log

localprefix=$OUT_PATH/${OUT_PREFIX}_$1.$matr

echo ssh "$DEVICE" "cd ${knc_basedir};export LD_LIBRARY_PATH=${knc_basedir}:/lib64;${ENVIRONMENT_MIC};./${exefile} ${infile} >./${resfile} 2>&1;errcode=\$?;echo \$errcode >./${errfile};rm -f ${infile}"
ssh "$DEVICE" "cd ${knc_basedir};export LD_LIBRARY_PATH=${knc_basedir}:/lib64;${ENVIRONMENT_MIC};./${exefile} ${infile} >./${resfile} 2>&1;errcode=\$?;echo \$errcode >./${errfile};rm -f ${infile}" >/dev/null

scp "${DEVICE}:$knc_basedir/$resfile" "$localprefix.log"
scp "${DEVICE}:$knc_basedir/$errfile" "$localprefix.err"

cat $localprefix.err

if [ \! -f "$localprefix.log" ]; then
  cat $localprefix.err
  exit 1
else
  cat $localprefix.log
fi

done
