%IMAGE_DESCRIPTION: bjad%

# Inference Acceleration for Large Language Models on CPUs 

{jithinvg, dittops, adarsh.ms}@bud.studio 

## Abstract 

In recent years, large language models have demonstrated remarkable performance across various natural language processing (NLP) tasks. However, deploying these models for realworld applications often requires efficient inference solutions to handle the computational demands. In this paper, we explore the utilization of CPUs for accelerating the inference of large language models. Specifically, we introduce a parallelized approach to enhance throughput by 1) Exploiting the parallel processing capabilities of modern CPU architectures, 2) Batching the inference request. Our evaluation shows the accelerated inference engine gives an **18-22x** improvement in the generated token per sec. The improvement is more with longer sequence and larger models. In addition to this, we can also run multiple workers in the 

| Model | Configuration | Processed tokens/s | Generated tokens/s |
|---|---|---:|---:|
| bigcode/starcoderbase-3b | without bud engine | 63.29 | 9.31 |
| bigcode/starcoderbase-3b | with bud engine | 992.36 | 167.58 |
| bigcode/starcoderbase-3b | with bud engine multiworker | 3969.44 | 670.32 |
| codellama/CodeLlama-7b-hf | without bud engine multiworker | 32.17 | 4.36 |
| codellama/CodeLlama-7b-hf | with bud engine | 593.74 | 93.69 |
| codellama/CodeLlama-7b-hf | with bud engine multiworker | 1852.32 | 305.30 |

Figure 1: Improvement of token/s with the use of Bud Inference engine with 32 vCPU on 4th Gen Intel® Xeon® Scalable Processors 

same machine with NUMA node isolation to further improvement in tokens/s. Table 2, we have received **4x** additional improvement with 4 workers. This would also make Gen-AI based products and companies’ environment friendly, our estimates shows that CPU usage for Inference could reduce the power consumption of LLMs by **48.9%** (1252 W for A100 with AMD EPYC 7V13 vs 613 W for Intel® Xeon® Gold 6538N) while providing production ready throughput & latency. 

## 1. Introduction 

The widespread integration of large language models (LLMs) across diverse applications, ranging from code generation to writing tasks, has surged in recent times, creating a heightened demand for more efficient inference solutions. Enterprises are increasingly leveraging LLMs to enhance various internal processes. However, the cost associated with their utilization remains a significant concern due to the reliance on GPUs for inference. [1] 

The current sequential approach employed by LLMs involves generating one token at a time based on the input prompt and the preceding tokens, persisting until a stop token or the predetermined maximum tokens are reached for each request. This method, while functional, restricts the optimal utilization of available resources [2], [3]. Elevating the tokens generated per second is crucial in mitigating the expenses associated with running LLM-enabled applications. While batching requests can increase throughput, achieving superior batching necessitates optimized memory utilization corresponding to the batch. 

| No: of requests | bigcode/starcoderbase-3b | budecosystem/code-millennials-13b |
|---|---|---|
| 0 | ~0 | ~0 |
| 100 | ~170 | ~50 |
| 200 | ~220 | ~80 |
| 300 | ~230 | ~90 |
| 400 | ~240 | ~100 |
| 500 | ~250 | ~110 |

Figure 2: This shows the tokens/s increases with the number of parallel requests increases due to the better utilization of the memory 

To address this challenge, we introduce an inference engine designed to enhance memory utilization by partitioning available memory into a series of tiles. This engine efficiently allocates incoming requests and assigns their respective tokens to specific memory tiles based on availability. By overseeing the computation and response delivery, the inference engine optimizes memory allocation per token, significantly enhancing performance for batching multiple requests. 

|**Model**|**Number of Request**|**Without Bud Inference**|**With Bud Inference**|
|---|---|---|---|
|bigcode/starcoderbase-3b|100|9.31|167.58|
|codellama/CodeLlama-7b-hf|100|4.36|93.69|



Table 1: Token generation speed for various models on 4th Gen Intel® Xeon® Scalable Processors with 32 vCPU and 100 requests 

In pursuit of cost-effective inference, we explore the feasibility of performing the same computations on Intel® Xeon® Scalable Processors. Leveraging optimizations like AVX and AMX, specifically tailored for Intel® Xeon® Scalable Processors [4], holds the promise of higher throughput in token generation. AVX facilitates parallel execution of operations on broader data vectors, while AMX focuses on optimizing matrix multiplication operations inherent in the core computations of transformer-based language models. The combined effect of these extensions enhances the utilization of the computational power of Intel® Xeon® Scalable Processors, resulting in superior throughput for CPU-based inference tasks. 

This paper focuses on harnessing the computational prowess of CPUs to accelerate the inference process. Through the strategic application of parallelization techniques and the utilization of the inference engine for batching, our goal is to enhance throughput, minimize latency, and render large language models more practical for real-time applications. 

## 3. Inference Acceleration 

Effective memory utilization stands out as a primary challenge in the current CPU-based inference systems. This challenge predominantly arises from the KV (Key-Value) cache, which dynamically adjusts memory utilization in response to the number of tokens [5], [6]. Consequently, the system experiences inefficiencies in memory allocation and grapples with both internal and external CPU memory fragmentation issues. Current systems reserve the required CPU memory for storing the KV cache of a request based on the max token length. But this will lead to internal fragmentation as the actual token generation might be less than the max token length. The reserved memory won’t be fully utilized. A request with a smaller token length won’t be able to utilize this unused reserved memory. Also, the max length for different request would be different which lead to external memory fragmentation. This limitation is there in GPU inference systems as well, as mentioned in paged attention [7]. 

