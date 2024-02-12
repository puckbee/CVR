# 转化率

Parallelized and vectorized SpMV on Intel Xeon Phi (Knights Landing). <br> This is the artifact of our CGO'2018 paper [ CVR: Efficient Vectorization of SpMV on X86 Processors ]. <br> You can find a copy of the paper in this repository named 'CVR-Efficient Vectorization of SpMV on X86 Processors.pdf'

# 建造

CVR can be built simply with 'make', while the resulted binariy file is 'spmv.cvr'.

```
Step: make
```

# 数据集准备和执行

我们的 CVR 实现支持矩阵市场格式的稀疏矩阵，这是 SuiteSparse Matrix Collection 中的默认格式之一。我们论文中使用的大多数数据集都可以在这两个集合中找到：

1. [SuiteSparse Matrix Collection](https://sparse.tamu.edu) （以前称为佛罗里达大学稀疏矩阵集合）。
2. [斯坦福大学大型网络数据集](http://snap.stanford.edu/data/)（SNAP）。

这里我们以web-Google为例来展示如何使用CVR：

```
step 1: ./run_sample.sh
```

CVR 接受三个参数： 文件路径；线程数；迭代次数。<br>在run_sample.sh中，有这样一个命令：

**numactl --membind=1 ./spmv.cvr [文件路径] [numThreads] [numIterations]**

**示例：numactl --membind=1 ./spmv.cvr dataset/web-Google.mtx 68 1000**

CVR 将在几秒内打印两次：[预处理时间]和[SpMV 执行时间]。<br> 【预处理时间】是将CSR格式的稀疏矩阵转换为CVR格式的时间。<br> [SpMV 执行时间] 是运行 CVR 格式的 SpMV 1000 次迭代的平均时间。请注意，1000 可以通过更改“迭代次数”来更改<br>

# 将 CVR 与其他格式/解决方案进行比较

MKL、CSR-I 和 ESB 都依赖于 MKL。<br>请确保 MKL 已安装并且环境变量 $MKL_ROOT 已设置。<br>

我们尝试了每种格式/解决方案的各种线程数和参数，并选择实现最佳性能的配置。<br>您可以尝试在run_comparison.sh中设置不同的线程数，稍后我们将详细说明如何执行此操作。<br>但如果你只是想重现web-Google的实验结果，这三个步骤绝对可以满足你的需求。<br>

```
Step 1: cd ./solutions_for_comparison

Step 2: ./build.sh        // build all formats/ solutions

Step 3: ./run_comparison.sh ../dataset/web-Google.mtx                           // run all formats/solutions
(a)     ./run_comparison.sh ../dataset/web-Google.mtx  | grep 'Pre-processing'  // get the Pre-processing time.
(b)     ./run_comparison.sh ../dataset/web-Google.mtx  | grep 'SpMV Execution'  // get the SpMV execution time.
(c)     ./run_comparison.sh ../dataset/web-Google.mtx  | grep 'Throughput'      // get the Throughput(GFlops).
```

我们将详细说明如何使用每种格式/解决方案，以便您可以更改配置以满足您自己的要求。

### 企业社会责任5

**numactl --membind=1 ./bin/spmv.csr5 [文件路径] [numThreads] [numIterations]**

**示例：numactl --membind=1 ./spmv.csr5 ../dataset/web-Google.mtx 204 1000**

### 肝细胞癌

VHCC 有很多参数。由于块的宽度和高度非常固定为（512,8192），因此我们在此仅提供面板的数量。

**numactl --membind=1 ./bin/spmv.vhcc [文件路径] [numThreads] [numIterations] [numPanels]**

**示例：numactl --membind=1 ./spmv.vhcc ../dataset/web-Google.mtx 272 1000 1**

### 企业社会责任-I

**numactl --membind=1 ./bin/spmv.csr [文件路径] [numThreads] [numIterations]**

**示例：numactl --membind=1 ./spmv.csr ../dataset/web-Google.mtx 272 1000**

### 企业服务总线

ESB 有不同的调度策略：静态和动态。 1 为静态； 2 为动态；两者均为 3。<br>

**numactl --membind=1 ./bin/spmv.esb [文件路径] [numThreads] [numIterations] [schedule_policy]**

**示例：numactl --membind=1 ./spmv.esb ../dataset/web-Google.mtx 272 1000 3**

### MKL

**numactl --membind=1 ./bin/spmv.mkl [文件路径] [numThreads] [numIterations]**

**示例：numactl --membind=1 ./spmv.mkl ../dataset/web-Google.mtx 272 1000**

# 缓存性能分析（附加）

依赖项：Vtune

```
Step 1: cd ./solutions_for_comparison
	
Step 2: ./build.sh                 // If it has not been built yet

Step 3: ./run_locality.sh [filepath][nT_CVR][nT_CSR5][nT_VHCC][nPanels][nT_CSRI][nT_ESB][schedule_ESB][nT_MKL]
        ./run_locality.sh ../dataset/web-Google.mtx 68 204 272 1 272 272 1 272

Note that 'nT' stands for numThreads, while 'nPanels' stands for numPanels of VHCC.
```

# 笔记

我们仅修改了 CSR5 和 VHCC 的源代码来格式化输出消息。<br> CSR5[ICS'15]的源代码，请参考（https://github.com/bhSPARSE/Benchmark_SpMV_using_CSR5）<br> VHCC[CGO'15]的源代码请参考(https://github.com/vhccspmv/vhcc)<br>

我们提供 ESB、CSR-I 和 MKL 的执行文件而不是源代码。如果您想查看代码，请给我发电子邮件（xiebiwei at ict.ac.cn）。我无法将代码放在 github 上，因为我不确定其许可证。请参阅[MKL 稀疏包](https://software.intel.com/en-us/articles/intel-math-kernel-library-inspector-executor-sparse-blas-routines)了解更多信息。
