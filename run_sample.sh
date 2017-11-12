#!/bin/bash
  if [ ! -d "dataset" ]; then  
    mkdir "dataset"  
  fi  
wget https://sparse.tamu.edu/MM/SNAP/web-Google.tar.gz
tar xvf web-Google.tar.gz
mv web-Google/web-Google.mtx ./dataset/
rm -rf web-Google*

numactl --membind=1 ./spmv.cvr ./dataset/web-Google.mtx 68 100
