#!/bin/bash

#amplxe-cl -collect-with runsa -knob event-config=MEM_UOPS_RETIRED.L1_MISS_LOADS,MEM_UOPS.RETIRED.ALL_LOADS,MEM_UOPS_RETIRED.L2_HIT_LOADS,MEM_UOPS_RETIRED.L2_MISS_LOADS,MEM_UOPS_RETIRED.ALL_LOADS,MEM_UOPS_RETIRED.ALL_STORES ./spmv.ours ../../dataset/amazon -r .


#make it clean first
rm -rf vtune*
rm -rf results.csv


filepath=$1

  if [ ! -d "../dataset" ]; then  
    mkdir "../dataset"  
  fi  
if [ ! -f $filepath ]; then
   wget https://sparse.tamu.edu/MM/SNAP/web-Google.tar.gz
   tar xvf web-Google.tar.gz
   mv web-Google/web-Google.mtx ../dataset/
   rm -rf web-Google*
   filepath=../dataset/web-Google.mtx
fi

file_name=${filepath##*/}
filename=${file_name%.*}
echo $filename

  if [ ! -d "vtune_result" ]; then  
    mkdir "vtune_result"  
  fi  

  if [ ! -d "./vtune_result/vtune_$filename" ]; then  
    mkdir "./vtune_result/vtune_$filename"  
  fi  
  if [ ! -d "vtune_log" ]; then  
    mkdir "vtune_log"  
  fi  

amplxe-cl -collect-with runsa -knob event-config=MEM_UOPS_RETIRED.L1_MISS_LOADS,MEM_UOPS.RETIRED.ALL_LOADS,MEM_UOPS_RETIRED.L2_HIT_LOADS,MEM_UOPS_RETIRED.L2_MISS_LOADS,MEM_UOPS_RETIRED.ALL_LOADS,MEM_UOPS_RETIRED.ALL_STORES,INST_RETIRED.ANY_P -r ./vtune_result/vtune_$filename/vtune_cvr/ numactl --membind=1 ./bin/spmv.cvr $filepath $2 1000
amplxe-cl -collect-with runsa -knob event-config=MEM_UOPS_RETIRED.L1_MISS_LOADS,MEM_UOPS.RETIRED.ALL_LOADS,MEM_UOPS_RETIRED.L2_HIT_LOADS,MEM_UOPS_RETIRED.L2_MISS_LOADS,MEM_UOPS_RETIRED.ALL_LOADS,MEM_UOPS_RETIRED.ALL_STORES,INST_RETIRED.ANY_P -r ./vtune_result/vtune_$filename/vtune_csr5/ numactl --membind=1 ./bin/spmv.csr5 $filepath $3 1000
amplxe-cl -collect-with runsa -knob event-config=MEM_UOPS_RETIRED.L1_MISS_LOADS,MEM_UOPS.RETIRED.ALL_LOADS,MEM_UOPS_RETIRED.L2_HIT_LOADS,MEM_UOPS_RETIRED.L2_MISS_LOADS,MEM_UOPS_RETIRED.ALL_LOADS,MEM_UOPS_RETIRED.ALL_STORES,INST_RETIRED.ANY_P -r ./vtune_result/vtune_$filename/vtune_vhcc/ numactl --membind=1 ./bin/spmv.vhcc $filepath $4 1000 $5
amplxe-cl -collect-with runsa -knob event-config=MEM_UOPS_RETIRED.L1_MISS_LOADS,MEM_UOPS.RETIRED.ALL_LOADS,MEM_UOPS_RETIRED.L2_HIT_LOADS,MEM_UOPS_RETIRED.L2_MISS_LOADS,MEM_UOPS_RETIRED.ALL_LOADS,MEM_UOPS_RETIRED.ALL_STORES,INST_RETIRED.ANY_P -r ./vtune_result/vtune_$filename/vtune_csr/ numactl --membind=1 ./bin/spmv.csr $filepath $6 1000
amplxe-cl -collect-with runsa -knob event-config=MEM_UOPS_RETIRED.L1_MISS_LOADS,MEM_UOPS.RETIRED.ALL_LOADS,MEM_UOPS_RETIRED.L2_HIT_LOADS,MEM_UOPS_RETIRED.L2_MISS_LOADS,MEM_UOPS_RETIRED.ALL_LOADS,MEM_UOPS_RETIRED.ALL_STORES,INST_RETIRED.ANY_P -r ./vtune_result/vtune_$filename/vtune_esb/ numactl --membind=1 ./bin/spmv.esb $filepath $7 1000 $8
amplxe-cl -collect-with runsa -knob event-config=MEM_UOPS_RETIRED.L1_MISS_LOADS,MEM_UOPS.RETIRED.ALL_LOADS,MEM_UOPS_RETIRED.L2_HIT_LOADS,MEM_UOPS_RETIRED.L2_MISS_LOADS,MEM_UOPS_RETIRED.ALL_LOADS,MEM_UOPS_RETIRED.ALL_STORES,INST_RETIRED.ANY_P -r ./vtune_result/vtune_$filename/vtune_mkl/ numactl --membind=1 ./bin/spmv.mkl $filepath $9 1000


#amplxe-cl -report hotspots -r ./vtune_result/vtune_$filename/vtune_mkl/ | grep 'avx512_mic_dcsr0ng__c__mvout_par'| awk '{print "MKL",$4,$10,$5,$8,$6,$7,$9}'>> vtune_log/log_$filename.txt
#amplxe-cl -report hotspots -r ./vtune_result/vtune_$filename/vtune_csr/ | grep '_csr_kernel_beta1' | awk '{print "CSRI",$2,$8,$3,$6,$4,$5,$7}' >> vtune_log/log_$filename.txt
#amplxe-cl -report hotspots -r ./vtune_result/vtune_$filename/vtune_esb/ | grep 'calcEsbBlock_dp' | awk '{print "ESB",$2,$8,$3,$6,$4,$5,$7}' >> vtune_log/log_$filename.txt
#amplxe-cl -report hotspots -r ./vtune_result/vtune_$filename/vtune_vhcc/ | grep 'run_spmv_vhcc' | awk '{print "VHCC",$2,$8,$3,$6,$4,$5,$7}' >> vtune_log/log_$filename.txt
#amplxe-cl -report hotspots -r ./vtune_result/vtune_$filename/vtune_csr5/ | grep 'runKernel' | awk '{print "CSR5",$2,$8,$3,$6,$4,$5,$7}' >> vtune_log/log_$filename.txt
#amplxe-cl -report hotspots -r ./vtune_result/vtune_$filename/vtune_ours/ | grep 'spmv_compute_kernel' | awk '{print "OURS",$2,$8,$3,$6,$4,$5,$7}' >> vtune_log/log_$filename.txt

echo "Format  L2_Hit  L2_Miss  Miss_Ratio " >> vtune_log/log_$filename.txt
amplxe-cl -report hotspots -r ./vtune_result/vtune_$filename/vtune_cvr/ | grep 'spmv_compute_kernel' | sed 's/,//g'| awk '{print "CVR",$4,$5,$5*100/($4+$5)"%"}' >> vtune_log/log_$filename.txt
amplxe-cl -report hotspots -r ./vtune_result/vtune_$filename/vtune_csr5/ | grep 'runKernel' | sed 's/,//g'| awk '{print "CSR5",$4,$5,$5*100/($4+$5)"%"}' >> vtune_log/log_$filename.txt
amplxe-cl -report hotspots -r ./vtune_result/vtune_$filename/vtune_vhcc/ | grep 'run_spmv_vhcc' | sed 's/,//g'| awk '{print "VHCC",$4,$5,$5*100/($4+$5)"%"}' >> vtune_log/log_$filename.txt
amplxe-cl -report hotspots -r ./vtune_result/vtune_$filename/vtune_csr/ | grep '_csr_kernel_beta1' | sed 's/,//g' | awk '{print "CSRI",$4,$5,$5*100/($4+$5)"%"}' >> vtune_log/log_$filename.txt
amplxe-cl -report hotspots -r ./vtune_result/vtune_$filename/vtune_esb/ | grep 'calcEsbBlock_dp' | sed 's/,//g'| awk '{print "ESB",$4,$5,$5*100/($4+$5)"%"}' >> vtune_log/log_$filename.txt
amplxe-cl -report hotspots -r ./vtune_result/vtune_$filename/vtune_mkl/ | grep 'avx512_mic_dcsr0ng__c__mvout_par'| sed 's/,//g' |awk '{print "MKL",$6,$7,$7*100/($6+$7)"%"}' >> vtune_log/log_$filename.txt

echo $filename >> vtune_log/log_all.txt
cat vtune_log/log_$filename.txt >> vtune_log/log_all.txt

echo "===============    The result is stored in vtune_log/log_"$filename".txt    ================"
cat vtune_log/log_$filename.txt
