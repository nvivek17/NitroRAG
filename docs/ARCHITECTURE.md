***
# NitroRAG: Engineering Design Review

This document outlines the architecture, design decisions, trade-offs, and empirical evaluations of the NitroRAG system. It is structured as an engineering design document to provide transparency into the iteration process from naive implementations to the final architecture.

## 📑 Contents
1. [Motivation & Scope](#1-motivation--scope)
2. [System Walkthrough](#2-system-walkthrough)
3. [Subsystem I: Vector Database (JAG)](#3-subsystem-i-vector-database-jag)
4. [Subsystem II: Paged Memory Manager](#4-subsystem-ii-paged-memory-manager)
5. [Subsystem III: Inference & Scheduler](#5-subsystem-iii-inference--scheduler)
6. [Subsystem IV: Networking](#6-subsystem-iv-networking)
7. [Alternatives Considered](#7-alternatives-considered)
8. [Why Not...?](#8-why-not)
9. [Empirical Benchmarks](#9-empirical-benchmarks)
10. [Lessons Learned (Failures)](#10-lessons-learned)
11. [Future Research](#11-future-research)
12. [Appendix: Memory Layouts](#12-appendix-memory-layouts)

---

## 1. Motivation & Scope

NitroRAG was developed to explore the physical hardware constraints of serving LLMs and retrieving vectors on CPU architectures. 

**Design Principles:**
1. **Minimize Memory Movement:** CPU inference is primarily limited by memory bandwidth rather than arithmetic throughput.
2. **Avoid Unnecessary Allocations:** Static allocation leads to OOM; dynamic allocation leads to latency. Pre-allocate and map.
3. **Separate Retrieval from Inference:** Storage scaling and Compute scaling require independent process topologies.
4. **Measure Before Optimizing:** Rely on hardware profiling rather than theoretical assumptions.

**Scope Limitation:** NitroRAG is not intended to replace production systems like vLLM or Milvus. It is an exploration from first principles to understand the underlying mechanics of these systems without abstracted framework overhead.

---

## 2. System Walkthrough

<p align="center">
  <img src="../assets/architecture.svg"
       width="900">
</p>

### Query Lifecycle
1. **Client Gateway:** Natural language is parsed into structured metadata (e.g., `Category=Security`, `Year>2024`) and embedded.
2. **Retrieval Phase:** C++ Vector DB traverses a multiplexed graph, routing through attribute-specific edges.
3. **Prefix Caching:** The Inference Engine hashes the prompt against the Radix cache to locate shared physical memory blocks.
4. **Chunked Prefill:** The scheduler injects the prompt into the active batch, computing KV states via GEMM AVX2 operations.
5. **Decode:** The engine iterates token-by-token (GEMV), dynamically requesting 16-token physical pages until `<EOS>`.

---

## 3. Subsystem I: Vector Database (JAG)

*Relevant Files: `src/JAG_new.cpp`, `src/Server_VectorDB.cpp`*

<p align="center">
  <img src="../assets/Graph Routing.svg"
       width="900">
</p>

### Design Problem
Executing highly selective conjunctive queries (`Subset AND Range`) on approximate nearest neighbor (ANN) graphs without performing full database scans.

### Iteration 1: Filtered Vamana (OOD-DiskANN)
*   **Approach:** Implemented strict subset pruning: $L(p) \cap L(c) \subseteq L(v)$.
*   **Why it failed:** Under strict memory budgets ($R=64$), rare labels were starved of edges. The graph shattered, yielding <5% recall on uncorrelated datasets.
*   **Attempted Fix:** Designed a dynamic quota system that partitioned edges proportionally to label frequency. This fixed subset routing but completely failed when introducing continuous range variables.

### Iteration 2: Multiplexed Joint Attribute Graphs (JAG)
We abandoned strict boolean pruning for continuous navigational guidance.
1. **Z-Score Normalization:** The database runs an $O(N)$ Monte Carlo sampling phase during startup to compute the global standard deviations ($\sigma$) of all attributes. Distances are evaluated using a normalized L1 penalty: `(Hamming / sigma_subset)` for subset penalty and  `(RangeDist / sigma_range)` for range penalty.
2. **Heterogeneous Edge Multiplexing:** The degree budget ($R=64$) is explicitly partitioned into 3 lanes during `RobustPrune`: 32 Vector edges, 16 Subset edges, 16 Range edges.
**Why it failed:** Under strict memory budgets ($R=64$), edge partioning such as only 16 edges for subset penalty and 16 edges for range penalty is very low .This gave poor results than expected due to degree budget


### Chosen Design: Multiplexed Joint Attribute Graphs (JAG) by Mixed penalty

1. **Z-Score Normalization:** The database runs an $O(N)$ Monte Carlo sampling phase during startup to compute the global standard deviations ($\sigma$) of all attributes. Distances are evaluated using a normalized L1 penalty: `(Hamming / sigma_subset) + (RangeDist / sigma_range)`. Notice that here now we are combining penalty so there are no longer separate subset penalty and separate range penalty , they got merged into one mixed metadata penalty
2. **Heterogeneous Edge Multiplexing:** The degree budget ($R=64$) is explicitly partitioned into 2 lanes during `RobustPrune`: 32 Vector edges, 32 semantic edges.

### Trade-offs
*   **Pros:** Prevents local minima plateaus during `AND` queries. Provides a smooth routing gradient.
*   **Cons:** Doubles the indexing build time, as `RobustPrune` must be evaluated against three distinct comparators.
              Using such mixed penalty gives good results as expected for mixed queries but it performed worse when there are subset only queries and range only queries as compared to mixed queries.

---

## 4. Subsystem II: Paged Memory Manager

*Relevant Files: `src/PagedMemory.hpp`*
<p align="center">
  <img src="../assets/Paged Attention.svg"
       width="900">
</p>

### Design Problem
Standard LLM inference statically allocates KV-cache arrays up to `max_seq_len` for every request, leading to massive internal fragmentation and OOM crashes under high concurrency.

### Chosen Design: PagedAttention & Radix Hashing
1. **Block Allocator:** Pre-allocates a massive contiguous RAM pool at boot. Divides memory into 16-token physical blocks.
2. **Radix-Style Hash Router:** Implements a rolling FNV-1a 64-bit hash. `Hash = FNV(Parent_Hash ^ Tokens)`. Evaluates state in $O(1)$ time. 
3. **Dependency-Aware LRU Eviction:** Tracks `child_count` topology. When free memory drops below 5%, the OS bulk-evicts true leaf nodes (`ref_count == 0 && child_count == 0`), cascading upwards.

### Trade-offs
*   **Pros:** Memory consumption maps strictly to generated tokens, eliminating internal fragmentation. Enables zero-compute RAG via shared physical blocks.
*   **Cons:** Introduces address-translation pointer-chasing inside the inner Attention loop, slightly degrading raw single-thread decode compute latency.

---

## 5. Subsystem III: Inference & Scheduler

*Relevant Files: `src/Server_Inference.cpp`*

### Design Problem
Maximizing Arithmetic Intensity (Compute/Memory Bandwidth ratio) on CPU architecture. 

### Chosen Design
1. **Int8 Q8_0 Quantization:** Weights are quantized block-wise (32 weights + 1 FP32 scale) into 36-byte structs to ensure perfect L1 cache-line alignment.
2. **AVX2 Kernels:** Explicit `_mm256_fmadd_ps` intrinsics handle dequantization and accumulation entirely within CPU registers.
3. **Continuous Batching (Sarathi-style):** The orchestrator evaluates the queue at every token iteration. Completed sequences are evicted instantly. Waiting prompts are chunked (e.g., 256 tokens) and injected into the active batch, allowing decode tokens to piggyback on prefill memory fetches.

### Trade-offs
*   **Pros:** Prevents head-of-line blocking. Drastically improves aggregate throughput.
*   **Cons:** Chunking the prefill phase slightly increases the absolute Time-To-First-Token (TTFT) for the new user, in exchange for maintaining flat inter-token latency for existing users.

---

## 6. Networking

*Relevant Files: `src/NitroRPC.hpp`*

### Design Problem
Connecting the Vector DB and Inference Engine without introducing control-plane serialization latency.

### Chosen Design: Custom Binary RPC over POSIX TCP
We bypass HTTP/2 and JSON string-parsing entirely. The protocol transmits 4-byte size headers followed by raw contiguous byte arrays, mapping directly back to `std::vector` allocations in C++.

---

## 7. Alternatives Considered

| Component | Alternative | Why Rejected | Chosen Approach |
| :--- | :--- | :--- | :--- |
| **Networking** | gRPC / Protobuf | Heavy dependency footprint; unnecessary HTTP/2 serialization overhead for local C++ to C++ IPC. | **POSIX TCP Binary Sockets** |
| **File I/O** | `fread()` / Streams | Forces data copy from OS kernel buffer to user space, wasting RAM and latency. | **`mmap()` (Zero-copy)** |
| **Hash Algo** | SHA-256 | Cryptographic overhead; too slow for per-block routing evaluation. | **FNV-1a 64-bit** |
| **Concurrency** | `std::thread` | Requires manual thread-pool implementation and barrier synchronization management. | **OpenMP** |
| **Vector DB** | Multi-Index | Intersecting independent subset/range indices scales poorly for highly selective queries. | **Multiplexed JAG** |

---

## 8. Why Not...?

*   **Why not FAISS or HNSWLib?** Using pre-built libraries abstracts away the memory alignment and graph pruning challenges. The goal was to build systems intuition, not just an application.
*   **Why not vLLM / llama.cpp?** Both are exceptional, production-grade tools. NitroRAG serves as an educational microcosm to understand *why* those systems were built the way they were.
*   **Why not CUDA/GPU?** Hardware availability. Studying memory bandwidth limits on an AVX2 CPU forces a deeper understanding of cache locality without the abstraction of NVIDIA's proprietary compilers.

---

# NitroRAG: Empirical Benchmarks & Profiling

This document outlines the rigorous benchmarking methodology used to evaluate the NitroRAG architecture. All tests were conducted on bare-metal to observe true hardware limits, cache-locality behaviors, and memory fragmentation.

### 💻 Hardware & Software Setup
*   **CPU:** Intel Core i5-13420H (4 P-Cores, 4 E-Cores, 12 Logical Threads)
*   **RAM:** 16 GB DDR4 (8 GB hard-limit imposed for OOM testing)
*   **Compiler:** GCC 11.4 (`-O3 -march=native -ffast-math -fopenmp`)
*   **Model:** 110M Parameter Llama Architecture (Int8 Q8_0 Quantized)
*   **Dataset:** SIFT1M (128-dimensional)

---

## 1. Multi-Attribute Graph Recall (JAG)

**Purpose:** Evaluate graph navigability under complex conjunctive filters (`Subset AND Range`) at extreme selectivity (<0.1%).
**Methodology:** The dataset was tested under two conditions: Spatially Correlated metadata (easier routing) and Spatially Uncorrelated metadata (pure chaos).

### Correlated Metadata (The Stress Test)

**Build Times**
* Standard Vamana: 690.43s
* Filtered Vamana: 633.82s
* NitroRAG: 1052.32s

**Subset Filter Only (Avg Valid Docs: 1761.16 | Selectivity: 0.1761%)**

 In below benchmarks *vis* represents *Number of Visited Nodes*

| L_search | Std R@10 (%) | Std Lat (ms) | Std Vis | Filt R@10 (%) | Filt Lat (ms) | Filt Vis | Nitro R@10 (%) | Nitro Lat (ms) | Nitro Vis |
|---------:|-------------:|-------------:|---------:|--------------:|--------------:|---------:|---------------:|---------------:|-----------:|
| 10  | 0.23% | 0.542 | 795  | 0.83% | 0.214 | 75   | 41.43% | 0.155 | 502  |
| 20  | 0.43% | 0.488 | 1152 | 0.73% | 0.216 | 124  | 58.70% | 0.324 | 766  |
| 40  | 0.77% | 0.694 | 1765 | 0.77% | 0.310 | 208  | 76.13% | 0.443 | 1222 |
| 70  | 0.97% | 0.931 | 2591 | 1.57% | 0.408 | 314  | 84.30% | 0.693 | 1817 |
| 100 | 1.20% | 1.187 | 3332 | 1.77% | 0.469 | 409  | 90.90% | 0.928 | 2419 |
| 200 | 1.73% | 1.940 | 5468 | 2.17% | 0.808 | 692  | 97.83% | 1.614 | 4157 |
| 400 | 2.87% | 3.285 | 8907 | 2.53% | 1.249 | 1181 | 99.17% | 2.869 | 7406 |

---

**Range Filter Only (Avg Valid Docs: 9962.9268 | Selectivity: 0.9963%)**

| L_search | Std R@10 (%) | Std Lat (ms) | Std Vis | Filt R@10 (%) | Filt Lat (ms) | Filt Vis | Nitro R@10 (%) | Nitro Lat (ms) | Nitro Vis |
|---------:|-------------:|-------------:|---------:|--------------:|--------------:|---------:|---------------:|---------------:|-----------:|
| 10  | 0.63% | 0.229 | 795  | N/A | N/A | N/A | 48.67% | 0.178 | 462  |
| 20  | 1.33% | 0.409 | 1152 | N/A | N/A | N/A | 64.53% | 0.282 | 670  |
| 40  | 3.47% | 0.746 | 1765 | N/A | N/A | N/A | 78.57% | 0.456 | 1046 |
| 70  | 6.47% | 0.925 | 2591 | N/A | N/A | N/A | 87.60% | 0.700 | 1570 |
| 100 | 9.17% | 1.175 | 3332 | N/A | N/A | N/A | 92.77% | 0.850 | 2076 |
| 200 | 17.97% | 1.994 | 5468 | N/A | N/A | N/A | 97.27% | 1.592 | 3621 |
| 400 | 38.43% | 3.246 | 8907 | N/A | N/A | N/A | 99.20% | 2.658 | 6295 |

---
**Mixed Filter (Subset + Range) (Avg Valid Docs: 378.4600 | Selectivity: 0.0378%)**

| L_search | Std R@10 (%) | Std Lat (ms) | Std Vis | Filt R@10 (%) | Filt Lat (ms) | Filt Vis | Nitro R@10 (%) | Nitro Lat (ms) | Nitro Vis |
|---------:|-------------:|-------------:|---------:|--------------:|--------------:|---------:|---------------:|---------------:|-----------:|
| 10  | 0.00% | 0.208 | 795  | 0.33% | 0.075 | 70   | 70.67% | 0.173 | 468  |
| 20  | 0.00% | 0.404 | 1152 | 1.40% | 0.128 | 125  | 85.17% | 0.271 | 667  |
| 40  | 0.03% | 0.730 | 1765 | 3.70% | 0.246 | 209  | 94.53% | 0.476 | 1030 |
| 70  | 0.10% | 0.912 | 2591 | 6.47% | 0.294 | 307  | 97.90% | 0.652 | 1512 |
| 100 | 0.23% | 1.168 | 3332 | 9.00% | 0.380 | 397  | 98.97% | 0.791 | 1959 |
| 200 | 0.37% | 1.964 | 5468 | 19.10% | 0.645 | 651  | 99.50% | 1.396 | 3340 |
| 400 | 0.83% | 3.234 | 8907 | 39.50% | 1.125 | 1080 | 99.90% | 2.466 | 5753 |


---

### Uncorrelated Metadata (The Stress Test)

**Build Times**
* Standard Vamana: 677.0174s
* Filtered Vamana: 533.2127s
* NitroRAG: 1155.5845s

**Subset Filter Only (Avg Valid Docs: 1512.7933 | Selectivity: 0.1513%)**

| L_search | Std R@10 (%) | Std Lat (ms) | Std Vis | Filt R@10 (%) | Filt Lat (ms) | Filt Vis | Nitro R@10 (%) | Nitro Lat (ms) | Nitro Vis |
|---------:|-------------:|-------------:|---------:|--------------:|--------------:|---------:|---------------:|---------------:|-----------:|
| 10  | 0.10% | 1.055 | 797  | 2.30% | 0.133 | 92   | 55.40% | 0.115 | 544  |
| 20  | 0.20% | 0.591 | 1149 | 2.57% | 0.136 | 142  | 77.33% | 0.254 | 819  |
| 40  | 0.47% | 0.814 | 1763 | 2.87% | 0.231 | 229  | 93.27% | 0.417 | 1266 |
| 70  | 0.97% | 1.006 | 2590 | 3.07% | 0.329 | 343  | 99.00% | 0.631 | 1831 |
| 100 | 1.60% | 1.228 | 3332 | 3.00% | 0.387 | 459  | 99.47% | 0.827 | 2329 |
| 200 | 3.07% | 1.949 | 5466 | 3.23% | 0.696 | 798  | 99.87% | 1.422 | 3868 |
| 400 | 4.83% | 3.080 | 8906 | 3.53% | 1.146 | 1424 | 100.00% | 2.619 | 6965 |

---

**Range Filter Only (Avg Valid Docs: 9985.8662 | Selectivity: 0.9986%)**
| L_search | Std R@10 (%) | Std Lat (ms) | Std Vis | Filt R@10 (%) | Filt Lat (ms) | Filt Vis | Nitro R@10 (%) | Nitro Lat (ms) | Nitro Vis |
|---------:|-------------:|-------------:|---------:|--------------:|--------------:|---------:|---------------:|---------------:|-----------:|
| 10  | 0.83  | 0.237 | 797  | N/A | N/A | N/A | 37.27 | 0.187 | 543  |
| 20  | 1.87  | 0.378 | 1149 | N/A | N/A | N/A | 52.30 | 0.317 | 811  |
| 40  | 3.77  | 0.570 | 1763 | N/A | N/A | N/A | 68.43 | 0.458 | 1299 |
| 70  | 7.23  | 0.803 | 2590 | N/A | N/A | N/A | 80.47 | 0.694 | 1953 |
| 100 | 10.43 | 1.070 | 3332 | N/A | N/A | N/A | 87.20 | 0.913 | 2590 |
| 200 | 20.60 | 1.813 | 5466 | N/A | N/A | N/A | 95.27 | 1.654 | 4509 |
| 400 | 39.77 | 3.099 | 8906 | N/A | N/A | N/A | 98.60 | 3.042 | 7763 |

---

**Mixed Filter (Subset + Range) (Avg Valid Docs: 371.7633 | Selectivity: 0.0372%)**

| L_search | Std R@10 (%) | Std Lat (ms) | Std Vis | Filt R@10 (%) | Filt Lat (ms) | Filt Vis | Nitro R@10 (%) | Nitro Lat (ms) | Nitro Vis |
|---------:|-------------:|-------------:|---------:|--------------:|--------------:|---------:|---------------:|---------------:|-----------:|
| 10  | 0.00% | 0.259 | 797  | 0.93% | 0.146 | 89   | 74.57% | 0.201 | 486  |
| 20  | 0.07% | 0.366 | 1149 | 2.07% | 0.118 | 139  | 87.63% | 0.283 | 681  |
| 40  | 0.27% | 0.642 | 1763 | 4.27% | 0.169 | 218  | 96.03% | 0.384 | 1051 |
| 70  | 0.37% | 0.832 | 2590 | 7.10% | 0.268 | 322  | 98.90% | 0.616 | 1557 |
| 100 | 0.50% | 1.112 | 3332 | 10.17% | 0.318 | 418  | 99.57% | 0.809 | 2011 |
| 200 | 0.97% | 1.803 | 5466 | 19.77% | 0.578 | 722  | 99.87% | 1.349 | 3385 |
| 400 | 1.90% | 3.057 | 8906 | 38.67% | 0.981 | 1278 | 99.90% | 2.452 | 6010 |

*Interpretation: OOD-DiskANN (Filtered Vamana) exhibits severe graph shattering on uncorrelated datasets due to strict boolean pruning constraints, capping out at 38% recall. NitroRAG's heterogeneous edge multiplexing and Z-Score Normalized L1 Penalty maintains 99.9% connectivity across all dimensions, reaching the ground truth while visiting significantly fewer nodes than standard post-filtering.*

---

## 2. Memory Scaling & Fragmentation (PagedAttention)

**Purpose:** Observe memory exhaustion boundaries for 4096-context length models under concurrent load.
**Methodology:** Compare theoretical static allocation (reserving `max_seq_len` arrays) against physical block allocation via the Paged Memory OS.

| Users | Naive Alloc (MB) | Paged Used (MB) | Paged Alloc (MB) | Fragmentation |
|-------|-----------------:|----------------:|-----------------:|--------------:|
| 1     | 288.00           | 7.03            | 7.88             | 0.84          |
| 4     | 1152.00          | 30.23           | 32.62            | 2.39          |
| 16    | 4608.00          | 154.69          | 163.12           | 8.44          |
| 32    | OOM (>8GB)       | 399.38          | 416.25           | 16.88         |
| 64    | OOM (>8GB)       | 1158.75         | 1192.50          | 33.75         |

*Interpretation: Static allocation scales linearly with context window limits, exhausting system memory rapidly. Paged allocation scales sub-linearly with actual tokens generated. Fragmentation remains mathematically bound to a maximum of 15 tokens per sequence.*

---

## 3. OpenMP Thread Scaling (The Memory Wall)

**Purpose:** Observe physical vs. logical core saturation during Decode (GEMV) workloads.

| Threads | Decode Tokens/sec |
|---------|------------------:|
| 1       | 71.56             |
| 2       | 116.18            |
| 4       | 135.28            |
| 8       | 149.41            |
| 10      | 154.75            |
| 12      | 150.14            |
| 14      | 32.17             |
| 16      | 26.22             |

*Interpretation: Throughput scales linearly up to four threads, saturating the physical P-Cores. Additional threads provide diminishing returns due to memory bandwidth saturation. The substantial performance regression observed beyond 12 threads is due to OS context switching and E-Core barrier synchronization overhead (Hardware Oversubscription).*

---

## 4. Radix Prefix Cache Effectiveness

**Purpose:** Evaluate RAG context processing latency on highly shared enterprise documents.

| Status     | Prefill Latency (ms) | Peak RAM Used (MB) |
|------------|---------------------:|-------------------:|
| Cache Miss | 730.3662970000       | 18.00              |
| Cache Hit  | 0.0045660000         | 18.00              |

*Interpretation: Mapping logical page tables to existing physical blocks via FNV-1a hashing provides a near instantaneous (~4µs) cache hit, bypassing GEMM arithmetic entirely and facilitating O(1) prompt processing for shared contexts.*

---

## 5. Continuous Batching Throughput

**Purpose:** Evaluate the iteration-level scheduler's ability to maximize arithmetic intensity.

In below benchmarks *TTFT* represents *Avg Time-To-First-Token (TTFT)*

| Concurrent Users | Aggregate Throughput (tok/s) | Avg TTFT (ms) |
|------------------|-----------------------------:|--------------:|
| 1                | 22.30                        | 735.52        |
| 4                | 49.79                        | 1258.45       |
| 8                | 130.69                       | 767.35        |
| 16               | 167.96                       | 1068.71       |
| 32               | 169.14                       | 1557.10       |

*Interpretation: By injecting chunked prefills into the active decode batch, the scheduler prevents queue head-of-line blocking. The system achieves a nearly 8x increase in aggregate throughput while keeping TTFT stable under moderate load, only climbing as the hardware reaches maximum FLOP saturation at 32 users.*

### ⚠️ Threats to Validity
*   **Scale:** Benchmarks were performed on a 110M parameter model. Memory bandwidth characteristics will shift significantly on 7B+ parameter models.
*   **Topology:** OpenMP thread scaling is highly specific to the hybrid P-Core/E-Core architecture of the test machine.
*   **Single Node:** All microservices were tested over `localhost`. Physical network latency was not modeled.

---

## 10. Lessons Learned & Mistakes

1. **Lexicographical Gradient Plateaus:** Initially, I attempted to route queries using a strict tuple `MAX(Subset_Tier, Range_Tier)`. This shattered the graph, as local improvements in a single dimension were hidden. Transitioning to a Z-Score normalized sum provided the necessary gradient.
2. **OpenMP Thread Explosion:** Placing `#pragma omp` strictly inside inner attention loops caused catastrophic synchronization overhead. I learned to utilize `collapse(2)` on outer batch/head loops to amortize thread spin-up costs.
3. **The Empty Graph Cold-Start:** Initializing Vamana with 0 edges (per the original pseudocode) created isolated cliques during early insertion, permanently destroying long-range highways. I implemented a label-aware randomized initialization to bootstrap connectivity.

---

## 11. Future Research

*   **Product Quantization (PQ):** Investigating the compression of the Vamana graph to 16-byte sub-spaces, allowing massive graphs to remain resident in RAM.
*   **Direct I/O Reranking:** Implementing `pread()` system calls to fetch uncompressed Float32 vectors directly from NVMe SSDs to mitigate PQ accuracy loss.
*   **Speculative Decoding:** Implementing a draft-model continuous batching loop to accelerate single-user token generation.

---

## 12. Appendix: Memory Layouts

**Int8 Q8_0 Block Alignment**

```cpp
const int Q_BLOCK_SIZE = 32;
// 36 bytes per block. 
// Guarantees scale factor is pulled into L1 cache identically with the 32 weights.
struct BlockQ8_0 {
    float scale;
    int8_t weights[Q_BLOCK_SIZE];
};
```

**Paged KV Block Topology**
```cpp
struct KVBlock {
    int physical_id;
    int num_tokens; 
    int ref_count;    // For Prefix Sharing

    // Topology Tracking (For O(1) Eviction)
    int parent_id;    
    int child_count;  

    float* k_ptr; 
    float* v_ptr;
};
```

***
