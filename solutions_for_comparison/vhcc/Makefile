
ICC = icpc
ICCFLAGS = -O3 -std=c++0x -restrict -D__MIC__

DEFINES = 
#INCLUDE_DIRS = -I/opt/intel/mkl/include  -I$(HOME)/local/include/poski -I$(HOME)/local/build_oski/include/oski/  -L$(HOME)/local/lib/
#MKLROOT = /opt/intel/compilers_and_libraries_2016.1.150/linux/mkl
#MKLROOT = /home/public/intel/mkl
INCLUDE_DIRS = -I$(MKLROOT)/include
#MKL_DIRS = -L$(MKLROOT)/lib/intel64
#MKL_LIBS_SEQ = -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm
#MKL_LIBS_THR = -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -lm
#MMIC = -mmic
MMIC = -xMIC-AVX512
OUTPATH = ./

#all : spmv.vhcc spmv.vhcc.mmap spmv.vhcc.100 spmv.vhcc.mmap.100
all : spmv.vhcc

#phi_spmv: $(wildcard *.h) $(wildcard *.cpp)
spmv.vhcc: $(*.h) $(*.cpp)
	$(ICC) $(ICCFLAGS) $(DEFINES) $(MMIC) -c -o $(OUTPATH)/mmio.o mmio.cpp  
	$(ICC) $(ICCFLAGS) $(DEFINES) $(MMIC) -c -o $(OUTPATH)/MatrixDataConverter.o MatrixDataConverter.cpp  
	$(ICC) $(ICCFLAGS) $(DEFINES) $(MMIC) -c -o $(OUTPATH)/SparseMatrixReader.o SparseMatrixReader.cpp  
	$(ICC) $(ICCFLAGS) $(DEFINES) $(INCLUDE_DIRS) \
		$(MMIC) -qopenmp -o $(OUTPATH)/spmv.vhcc $(OUTPATH)/mmio.o $(OUTPATH)/SparseMatrixReader.o $(OUTPATH)/MatrixDataConverter.o phi_spmv.cpp $(LIBS) \
		-Wl,--start -group  -Wl,--end -group

spmv.vhcc.mmap: $(*.h) $(*.cpp)
	$(ICC) $(ICCFLAGS) $(DEFINES) $(MMIC) -c -o $(OUTPATH)/mmio.o mmio.cpp  
	$(ICC) $(ICCFLAGS) $(DEFINES) $(MMIC) -c -o $(OUTPATH)/MatrixDataConverter.o MatrixDataConverter.cpp  
	$(ICC) $(ICCFLAGS) $(DEFINES) $(MMIC) -c -o $(OUTPATH)/SparseMatrixReader.o SparseMatrixReader.cpp  
	$(ICC) $(ICCFLAGS) $(DEFINES) $(INCLUDE_DIRS) -DMMAP \
		$(MMIC) -qopenmp -o $(OUTPATH)/spmv.vhcc.mmap $(OUTPATH)/mmio.o $(OUTPATH)/SparseMatrixReader.o $(OUTPATH)/MatrixDataConverter.o phi_spmv.cpp $(LIBS) \
		-Wl,--start -group  -Wl,--end -group

spmv.vhcc.100: $(*.h) $(*.cpp)
	$(ICC) $(ICCFLAGS) $(DEFINES) $(MMIC) -c -o $(OUTPATH)/mmio.o mmio.cpp  
	$(ICC) $(ICCFLAGS) $(DEFINES) $(MMIC) -c -o $(OUTPATH)/MatrixDataConverter.o MatrixDataConverter.cpp  
	$(ICC) $(ICCFLAGS) $(DEFINES) $(MMIC) -c -o $(OUTPATH)/SparseMatrixReader.o SparseMatrixReader.cpp  
	$(ICC) $(ICCFLAGS) $(DEFINES) $(INCLUDE_DIRS) -DNITERS=100 \
		$(MMIC) -qopenmp -o $(OUTPATH)/spmv.vhcc.100 $(OUTPATH)/mmio.o $(OUTPATH)/SparseMatrixReader.o $(OUTPATH)/MatrixDataConverter.o phi_spmv.cpp $(LIBS) \
		-Wl,--start -group  -Wl,--end -group

spmv.vhcc.mmap.100: $(*.h) $(*.cpp)
	$(ICC) $(ICCFLAGS) $(DEFINES) $(MMIC) -c -o $(OUTPATH)/mmio.o mmio.cpp  
	$(ICC) $(ICCFLAGS) $(DEFINES) $(MMIC) -c -o $(OUTPATH)/MatrixDataConverter.o MatrixDataConverter.cpp  
	$(ICC) $(ICCFLAGS) $(DEFINES) $(MMIC) -c -o $(OUTPATH)/SparseMatrixReader.o SparseMatrixReader.cpp  
	$(ICC) $(ICCFLAGS) $(DEFINES) $(INCLUDE_DIRS) -DMMAP -DNITERS=100 \
		$(MMIC) -qopenmp -o $(OUTPATH)/spmv.vhcc.mmap.100 $(OUTPATH)/mmio.o $(OUTPATH)/SparseMatrixReader.o $(OUTPATH)/MatrixDataConverter.o phi_spmv.cpp $(LIBS) \
		-Wl,--start -group  -Wl,--end -group
clean:
	rm -rf ./*.o spmv.vhcc*
