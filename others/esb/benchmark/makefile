##******************************************************************************
##  Copyright(C) 2013 Intel Corporation. All Rights Reserved.
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
##      Creation of a C performance benchmark for 
##      Intel(R) MKL SpMV Format Prototype Package
##******************************************************************************

#OPTS= -mmic -D__KNC__
OPTS= -xMIC-AVX512 -D__KNC__
SRCDIR=./src

ifndef MKLROOT
  $(error Path to Intel MKL is not defined in MKLROOT)
endif

MKL_LIB=$(MKLROOT)/lib/intel64_lin

spmvlib=../lib/intel64/libmkl_spmv_pkg_avx.a 
MKL_LINK= -Wl,--start-group $(MKL_LIB)/libmkl_intel_lp64.a $(MKL_LIB)/libmkl_intel_thread.a $(MKL_LIB)/libmkl_core.a -Wl,--end-group

#all: spmv.esb spmv.esb.mmap spmv.esb.100 spmv.esb.mmap.100
all: spmv.esb spmv.csr spmv.mkl

spmv.esb:
	icc -c $(OPTS)         $(SRCDIR)/mmio.c       -I$(SRCDIR)
	icc -c $(OPTS) -qopenmp $(SRCDIR)/spmv_bench.c -I$(MKLROOT)/include -I../include -mkl -D_SpMV_ESB_
	icc    $(OPTS) -qopenmp mmio.o spmv_bench.o -o spmv.esb $(spmvlib) $(MKL_LINK) -mkl

spmv.csr:
	icc -c $(OPTS)         $(SRCDIR)/mmio.c       -I$(SRCDIR)
	icc -c $(OPTS) -qopenmp $(SRCDIR)/spmv_bench.c -I$(MKLROOT)/include -I../include -mkl -D_SpMV_CSR_
	icc    $(OPTS) -qopenmp mmio.o spmv_bench.o -o spmv.csr $(spmvlib) $(MKL_LINK) -mkl

spmv.mkl:
	icc -c $(OPTS)         $(SRCDIR)/mmio.c       -I$(SRCDIR)
	icc -c $(OPTS) -qopenmp $(SRCDIR)/spmv_bench.c -I$(MKLROOT)/include -I../include -mkl -D_SpMV_MKL_
	icc    $(OPTS) -qopenmp mmio.o spmv_bench.o -o spmv.mkl $(spmvlib) $(MKL_LINK) -mkl

spmv.esb.mmap:
	icc -c $(OPTS)         $(SRCDIR)/mmio.c       -I$(SRCDIR)
	icc -c $(OPTS) -qopenmp $(SRCDIR)/spmv_bench.c -I$(MKLROOT)/include -I../include -mkl -DMMAP
	icc    $(OPTS) -qopenmp mmio.o spmv_bench.o -o spmv.esb.mmap $(spmvlib) $(MKL_LINK) -mkl

spmv.esb.100:
	icc -c $(OPTS)         $(SRCDIR)/mmio.c       -I$(SRCDIR)
	icc -c $(OPTS) -qopenmp $(SRCDIR)/spmv_bench.c -I$(MKLROOT)/include -I../include -mkl -DNITERS=100
	icc    $(OPTS) -qopenmp mmio.o spmv_bench.o -o spmv.esb.100 $(spmvlib) $(MKL_LINK) -mkl

spmv.esb.mmap.100:
	icc -c $(OPTS)         $(SRCDIR)/mmio.c       -I$(SRCDIR)
	icc -c $(OPTS) -qopenmp $(SRCDIR)/spmv_bench.c -I$(MKLROOT)/include -I../include -mkl -DMMAP -DNITERS=100
	icc    $(OPTS) -qopenmp mmio.o spmv_bench.o -o spmv.esb.mmap.100 $(spmvlib) $(MKL_LINK) -mkl
clean:
	rm -f *.o *.exe spmv.esb* spmv.csr* spmv.mkl*











