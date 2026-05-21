# J-Llama

[![Java](https://img.shields.io/badge/Java-17%2B-blue.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

[English](#english) | [中文](#chinese-version)

---

## English <a id="english"></a>
J-Llama is a pure Java inference engine for the Llama 2 architecture. 
It has **zero third-party dependencies** (no PyTorch, TensorFlow, etc.). The core logic is built entirely from scratch using native Java arrays and fundamental math operators.

The primary goal of this project is to deeply understand the Transformer architecture by building it from the ground up, while exploring the engineering limits of Java in dense matrix computations and off-heap memory management.

### Core Design

* **Zero-Copy Loading**: Utilizes Java NIO `MappedByteBuffer` to map model weights directly into off-heap system memory. Loading a 15M parameter model takes only ~100ms with zero OOM risk.
* **Memory Pooling (Zero GC)**: Designed a global `RunState` and KV Cache memory pool for high-frequency activation matrices. Multi-dimensional tensors are flattened into continuous 1D `float[]` arrays to maximize CPU L1/L2 cache hit rates and eliminate Young GC pauses.
* **Multi-threaded MatMul**: Leverages JUC (`ThreadPoolExecutor` + `CountDownLatch`) to chunk the core bottleneck (Matrix Multiplication) across available CPU cores, achieving lock-free concurrency.
* **Native Transformer**: Pure Java implementation of RoPE (Rotary Positional Embedding), SwiGLU activation, and Multi-Head Attention.

### Quick Start

**1. Prepare Model Files**
The parsing logic is compatible with the [llama2.c](https://github.com/karpathy/llama2.c) weight format. Create a `models` directory in the root and place the following files inside:
* `stories15M.bin` (Model weights)
* `tokenizer.bin` (BPE Tokenizer)

**2. Compile and Run**
Zero dependencies, simply compile with JDK:
```bash
javac -d bin src/main/java/com/kirisu/llama2/*.java
java -cp bin com.kirisu.llama2.Llama2Runner

Benchmark
Environment: Standard Desktop CPU

Memory Footprint: < 100 MB

Generation Speed: ~56 tokens/s (Tested)

Sample Output:

Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the sky. It was the sun! She thought it was so pretty...

## 中文 <a id="chinese-version"></a>
J-Llama 是一个用纯 Java 手写的 Llama 2 推理引擎。
项目零第三方依赖（无 PyTorch/TensorFlow 等框架），核心逻辑完全基于原生 Java 数组和底层数学算子实现。

写这个项目的初衷是为了拒绝“调包”，从代码层面吃透 Transformer 架构；同时也借此验证 Java 在密集型矩阵计算和堆外内存管理上的工程极限。

核心设计
零拷贝 (Zero-Copy) 加载：使用 NIO MappedByteBuffer 将权重文件直接映射到系统内存，绕过 JVM 堆区。15M 参数规模加载耗时 ~100ms，无 OOM 风险。

内存池化无 GC：针对推理过程中的激活矩阵，设计了全局 RunState 和 KV Cache 内存池。将多维张量展开为一维连续 float[] 以提高 CPU L1/L2 缓存命中率，避免高频 Young GC 停顿。

多线程算子压榨：基于 JUC (ThreadPoolExecutor + CountDownLatch)，将核心瓶颈 MatMul（矩阵乘法）按 CPU 核心数进行 Chunk 切块，实现无锁并发。

Transformer 原生复现：纯代码实现了 RoPE (旋转位置编码)、SwiGLU 激活函数以及 Multi-Head Attention。

快速开始
1. 准备模型文件
本项目解析逻辑兼容 llama2.c 的权重格式。
请在根目录下创建 models 文件夹，并放入以下文件：

stories15M.bin (模型权重)

tokenizer.bin (词表文件)

2. 编译与运行
零依赖，无需 Maven/Gradle，直接使用 JDK 编译：
javac -d bin src/main/java/com/kirisu/llama2/*.java
java -cp bin com.kirisu.llama2.Llama2Runner
性能测试 (Benchmark)
测试环境：普通家用 CPU

内存占用：< 100 MB

生成速度：~56 tokens/s (实测)

输出演示：Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big, red ball in the sky. It was the sun! She thought it was so pretty...
