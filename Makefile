
all: spmv.cpp
	icpc -O3 -ansi -ansi-alias -xMIC-AVX512 -qopenmp spmv.cpp -o spmv.cvr
#	icpc -O3 -ansi-alias -xMIC-AVX512 -qopenmp spmv.cpp -DMMAP -o spmv.ours.mmap
#	icpc -O3 -ansi-alias -xMIC-AVX512 -qopenmp spmv.cpp -DNITERS=100 -o spmv.ours.100
#	icpc -O3 -ansi-alias -xMIC-AVX512 -qopenmp spmv.cpp -DNITERS=100 -DMMAP -o spmv.ours.mmap.100
#debug:
#	icpc -O0 -ansi-alias -xMIC-AVX512 -qopenmp spmv.cpp -o spmv.g -g
clean:
	rm spmv.cvr
