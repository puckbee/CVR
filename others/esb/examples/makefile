##******************************************************************************
##  Copyright(C) 2013-2014 Intel Corporation. All Rights Reserved.
##  
##  The source code, information  and  material ("Material") contained herein is
##  owned  by Intel Corporation or its suppliers or licensors, and title to such
##  Material remains  with Intel Corporation  or its suppliers or licensors. The
##  Material  contains proprietary information  of  Intel or  its  suppliers and
##  licensors. The  Material is protected by worldwide copyright laws and treaty
##  provisions. No  part  of  the  Material  may  be  used,  copied, reproduced,
##  modified, published, uploaded, posted, transmitted, distributed or disclosed
##  in any way  without Intel's  prior  express written  permission. No  license
##  under  any patent, copyright  or  other intellectual property rights  in the
##  Material  is  granted  to  or  conferred  upon  you,  either  expressly,  by
##  implication, inducement,  estoppel or  otherwise.  Any  license  under  such
##  intellectual  property  rights must  be express  and  approved  by  Intel in
##  writing.
##  
##  *Third Party trademarks are the property of their respective owners.
##  
##  Unless otherwise  agreed  by Intel  in writing, you may not remove  or alter
##  this  notice or  any other notice embedded  in Materials by Intel or Intel's
##  suppliers or licensors in any way.
##
##******************************************************************************
##  Content:
##      Intel(R) MKL SpMV Format Prototype Package
##      C examples creation and run
##******************************************************************************

MIC.DEVICE=mic0

isa=avx

LIB.OMP=../lib/mic/libiomp5.so
LIB.MIC=../lib/mic/libmkl_spmv_pkg.a
LIB.HOST=../lib/intel64/libmkl_spmv_pkg_$(isa).a

NATIVE.EXE=example.exe
NATIVE.SRC=spmv_new.c
NATIVE.OUT=$(NATIVE.EXE).out
NATIVE.ERR=$(NATIVE.EXE).err
NATIVE.OPTS=-mmic

OFFLOAD.EXE=example_ofload.exe
OFFLOAD.SRC=spmv_offload.c
OFFLOAD.OUT=$(OFFLOAD.EXE).out
OFFLOAD.ERR=$(OFFLOAD.EXE).err
OFFLOAD.OPTS=-offload-attribute-target=mic
OFFLOAD.LINK=-offload-option,mic,compiler,"$(LIB.MIC)"

COPTS=-openmp -I../include

CC=icc
LD=icc

all: clean build execute

clean:
	rm -rf *.o *.out *.err *.exe

build: build.native build.offload

build.native:
	$(CC)    $(COPTS) $(NATIVE.OPTS)  $(NATIVE.SRC)   $(LIB.MIC)  -o $(NATIVE.EXE)

build.offload:
	$(CC) -c $(COPTS) $(OFFLOAD.OPTS) $(OFFLOAD.SRC)
	$(LD) spmv_offload.o $(LIB.HOST)  $(OFFLOAD.LINK) -o $(OFFLOAD.EXE)

execute: execute.native execute.offload

execute.native:
	scp $(NATIVE.EXE) $(LIB.OMP) $(MIC.DEVICE):/tmp >/dev/null 2>&1
	ssh $(MIC.DEVICE) "cd /tmp;export LD_LIBRARY_PATH=.;./$(NATIVE.EXE) >./$(NATIVE.OUT) 2>&1;echo \$$?>./$(NATIVE.ERR)" >/dev/null 2>&1
	scp $(MIC.DEVICE):/tmp/$(NATIVE.OUT) . >/dev/null 2>&1
	scp $(MIC.DEVICE):/tmp/$(NATIVE.ERR) . >/dev/null 2>&1
	ssh $(MIC.DEVICE) "cd /tmp; rm -f $(NATIVE.EXE) $(NATIVE.OUT) $(NATIVE.ERR)" >/dev/null 2>&1

execute.offload:
	./$(OFFLOAD.EXE) >./$(OFFLOAD.OUT) 2>&1;echo $$?>./$(OFFLOAD.ERR)
