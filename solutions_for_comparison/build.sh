#!/bin/bash
echo "=========  building csr5  ============"
cd ./csr5
make clean;  make
cd -
cp ./csr5/spmv.csr5 ./bin/

echo "=========  building vhcc  ============"
cd ./vhcc
make clean;  make
cd -
cp ./vhcc/spmv.vhcc ./bin/

#echo "========   building mkl,csr_i and esb =========="
#cd ./esb/benchmark
#make clean;  make
#cd -
#cp ./esb/benchmark/spmv.esb ./bin/
#cp ./esb/benchmark/spmv.csr ./bin/
#cp ./esb/benchmark/spmv.mkl ./bin/

echo "========   building cvr  ============="
cd ../
make clean;  make
cd -
cp ../spmv.cvr ./bin/
