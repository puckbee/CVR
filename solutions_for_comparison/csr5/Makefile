#compilers
CC=icc

#GLOBAL_PARAMETERS
VALUE_TYPE = double
NUM_RUN = 1000

#ENVIRONMENT_PARAMETERS
#source /opt/intel/composer_xe_2015.1.133/bin/compilervars.sh intel64
#export OMP_NUM_THREADS=236;export KMP_AFFINITY=default

#all: spmv.csr5 spmv.csr5.mmap spmv.csr5.100 spmv.csr5.mmap.100
all: spmv.csr5

spmv.csr5:
	$(CC) -O3 -xMIC-AVX512 -qopenmp main.cpp -o spmv.csr5 -D VALUE_TYPE=$(VALUE_TYPE)
spmv.csr5.mmap:
	$(CC) -O3 -xMIC-AVX512 -qopenmp main.cpp -o spmv.csr5.mmap -D VALUE_TYPE=$(VALUE_TYPE) -D NUM_RUN=$(NUM_RUN) -DMMAP
spmv.csr5.100:
	$(CC) -O3 -xMIC-AVX512 -qopenmp main.cpp -o spmv.csr5.100 -D VALUE_TYPE=$(VALUE_TYPE) -D NUM_RUN=100
spmv.csr5.mmap.100:
	$(CC) -O3 -xMIC-AVX512 -qopenmp main.cpp -o spmv.csr5.mmap.100 -D VALUE_TYPE=$(VALUE_TYPE) -D NUM_RUN=100 -DMMAP

clean:
	rm spmv.csr5*
