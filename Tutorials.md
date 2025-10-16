# Tutorials

Below are the tutorials for writing various basic operations with Trition.

## Vector Additon

In this tutorial, you will write a simple vector addition using Triton.

In doing so, you will learn about:

- The basic programming model of Triton.
- The *triton.jit* decorator, which is used to define Triton kernels.
- The best parctices for validating and benchmarking your custom ops against native reference implementations.

### Compute Kernel

```python 
import torch  
import triton  
import triton.language as tl  
  
DEVICE = triton.runtime.driver.active.get_active_torch_device()  
  
  
@triton.jit  
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):  
    '''  
    add kernel    :param x_ptr: pointer to first input vector    :param y_ptr: pointer to second input vector    :param output_ptr: pointer to output vector    :param n_elements: size of the vector    :param BLOCK_SIZE: number of elements each program should process    :return:  
    '''  
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0  
  
    # This progrom will process inputs that are offset from the initial data.    # For instance, if you have a vector of length 256 and block_size of 64, the programs    # would each access the elements [0:64,64:128, 128:192, 192:256].    # Note that offsets is a list of pointers.    
    
    block_start = pid * BLOCK_SIZE  
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  
  
    # Create a mask to guard memory oprations against out-of-bounds accesses.  
    mask = offsets < n_elements  
  
    # Load x and y from DRAM, masking out any extra elements in case the input is not a multiple of the block size.  
    x = tl.load(x_ptr + offsets, mask=mask)  
    y = tl.load(y_ptr + offsets, mask=mask)  
    output = x + y  
    tl.store(output_ptr + offsets, output, mask=mask)
```

Let's also declare a helper function to **allocate the z tensor** and **enqueue the above kernel with appropriate grid/block sizes**:

```python
def add(x: torch.Tensor, y: torch.Tensor):  
    # We need to preallocate the output.  
    output = torch.empty_like(x)  
    if not (x.device == DEVICE and y.device == DEVICE and output.device == DEVICE):  
        raise "x, y, output is not at same device"  
    n_elements = output.numel()  
    # The SPMD (Single Program Multiple Data) launch grid denotes the number of kernel instances that run in parallel.  
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].    
    # In this case, we use a 1D grid where the size is the number of blocks:    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)  
    # NOTE:  
    # - Each torch.tensor object is implicitly converted into a pointer to its first element.    # - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.    # - Don't forget to pass meta-parameters as keywords arguments.    
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)  
    # We return a handle to z, but since `torch.cuda.synchronize()` hasn't been called, the kernel is still running asynchronously at this point.  
    return output
```

We can now use the above function to compute the element-wise sum of two *torch.tensor* objects and test its correctness:

```python
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
```

Out:
```python
tensor([1.3713, 1.3076, 0.4940,  ..., 0.6724, 1.2141, 0.9733], device='cuda:0')
tensor([1.3713, 1.3076, 0.4940,  ..., 0.6724, 1.2141, 0.9733], device='cuda:0')
The maximum difference between torch and triton is 0.0
```

### Benchmarks

We can now benchmark our custom op on vectors of increasing sizes to get a sense of how it does relative to PyTorch. To make things easier, Triton has a set of built-in utilities that allow us to concisely plot performance of our custom ops for different problem sizes.

```python
@triton.testing.perf_report(  
    triton.testing.Benchmark(  
        x_names=['size'],  
        x_vals=[2 ** i for i in range(12, 28, 1)],  
        x_log=True,  
        line_arg='provider',  
        line_vals=['triton', 'torch'],  
        line_names=['Triton', 'Torch'],  
        styles=[('blue', '-'), ('green', '-')],  
        ylabel='GB/s',  
        plot_name='vector-add-performance',  
        args={},  
    ))  
def benchmark(size, provider):  
    x = torch.rand(size, device=DEVICE, dtype=torch.float32)  
    y = torch.rand(size, device=DEVICE, dtype=torch.float32)  
    quantiles = [0.5, 0.2, 0.8]  
    if provider == 'torch':  
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)  
    if provider == 'triton':  
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)  
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)  
    return gbps(ms), gbps(min_ms), gbps(max_ms)  
```

We can now run the decorated function above. Pass *print_data=True* to see the performance number, *show_plots=True* to plot them, and *save_path='path_to_results'* to save them to disk along with raw CSV data:

```python
benchmark.run(print_data=True, show_plots=True)
```

![performance](assets/vector_additon_performance.png)

## Fused Softmax

In this tutorial, you will write a fused softmax operation that is significantly than PyTorch's native op for a particular class of matrices: those whose rows can fit in the GPU's SRAM.

In doing so, you will learn about:

- The benefits of kernel fusion (内核融合) for bandwidth-bound (带宽受限) operations.
- Reduction operators in Triton.

### Motivations

Custom GPU kernels for elementwise additons are educationally valuable but won't get you very far in practice. Let us consider instead the case of a simple (numerically stabilized) softmax operation:

```python
import torch  
import triton  
import triton.language as tl  
from triton.runtime import driver  
  
DEVICE = triton.runtime.driver.active.get_active_torch_device()  
  
  
def is_hip():  
    return triton.runtime.driver.active.get_current_target().backend == 'hip'  
  
  
def is_cdna():  
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942',  
                                                                                   'gfx90a', 'gfx908')  
  
  
def naive_softmax(x):  
    '''Compute row-wise softmax of X using native pytorch  
  
    We subtract the maximum element in order to avoid overflows. Softmax is invariant(不变的) to this shift.  
    '''  
    # read MN elements; write M elements  
    x_max = x.max(dim=1).values  
    # read MN + M elements; write MN elements  
    z = x - x_max[:, None]  
    # read MN elements; write MN elements  
    numerator = torch.exp(z)  
    # read MN elements; write M elements  
    denominator = numerator.sum(dim=1)  
    # read MN + M elements; write MN elements  
    ret = numerator / denominator[:, None]  
    # in total: read 5MN + 2M elements; wrote 3MN + 2M elements  
    return ret
```

Computing `y = naive_softmax(x)` for $x \in R^{M \times N}$ requires reading $5MN + 2M$ elements from DRAM and write back $3MN + 2M$ elements. This is obviously wasteful; we'd prefer to have a custom "fused" kenel that only reads X once and does all the necessary computations on-chip. Doing so would require reading and writing back only $MN$ bytes, so we could expect a theoretical speed-up of ~4x. The `torch.jit.script` flags aims to perform this kind of "kernel fusion" automatically but, as we will see later, ti is still far from ideal.

### Compute Kernel

Our softmax kernel works as follows: each program loads a set of rows of the input matrix X strided by number of programs, normalizes it and writes back the result to the output Y.

Note that one important limitation of Triton is that each block must have a power-of-two number of elements, so we need to internally "pad" each row and guard the memory operations properly if we want to handle any possible input shapes:

```python
@triton.jit  
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr,  
                   num_stages: tl.constexpr):  
    # starting row of the program  
    row_start = tl.program_id(0)  
    row_step = tl.num_programs(0)  
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):  
        # The stride represents how much we need to increase the pointer to advance 1 row  
        row_start_ptr = input_ptr + row_idx * input_row_stride  
        # The block size is the next power of two greater than n_cols, so we can fit each row in a single block  
        col_offsets = tl.arange(0, BLOCK_SIZE)  
        input_ptrs = row_start_ptr + col_offsets  
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be bigger than n_cols  
        mask = col_offsets < n_cols  
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))  
        # Subtract maximum for numerical stability  
        row_minus_max = row - tl.max(row, axis=0)  
        # Note that exponentiation in Triton is fast but approximate(近似的)  
        numerator = tl.exp(row_minus_max)  
        denominator = tl.sum(numerator, axis=0)  
        softmax_output = numerator / denominator  
        # Write back output to DRAM  
        output_row_start_ptr = output_ptr + row_idx * output_row_stride  
        output_ptrs = output_row_start_ptr + col_offsets  
        tl.store(output_ptrs, softmax_output, mask=mask)
```

We can create a helper function that enqueues the kernel and its arguments for any given input tensor.

```python
import torch
import triton
import triton.language as tl
from triton.language.semantic import num_programs
from triton.runtime import driver

DEVICE = triton.runtime.driver.active.get_active_torch_device()

properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties['multiprocessor_count']
NUM_REGS = properties['max_num_regs']
SIZE_SMEM = properties['max_shared_mem']
WARP_SIZE = properties['warpSize']
target = triton.runtime.driver.active.get_current_target()
kernels = {}


def softmax(x):
    n_rows, n_cols = x.shape

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Another trick we can use is to ask the compiler to use more threads per row by
    # increasing the number of warps(`num_warps`) over which each row is distributed.
    # You can set this value with a auto way
    num_warps = 8

    # Number of software pipelining stages.
    num_stages = 4 if SIZE_SMEM > 200000 else 2

    # Allocate output
    y = torch.empty_like(x)

    # pre-compile kernel to get register usage and compute thread occupancy
    kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                   num_stages=num_stages, num_warps=num_warps, grid=(1,))
    kernel._init_handles()
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared

    if is_hip():
        # NUM_REGS represents the number of regular purpose registers. On CDNA architectures(AMD) this is half of all registers available.
        # However, this is not always the case. In most cases all registers can be used as regular purpose registers.
        NUM_GPRS = NUM_REGS
        if is_cdna():
            NUM_GPRS = NUM_REGS * 2

        # MAX_NUM_THREADS represents maximum number of resident threads per multi-processor.
        # When we divide this number with WARP_SIZE we get maximum number of waves that can execute on a CU(multi-processor) in parallel.
        MAX_NUM_THREADS = properties['max_threads_per_sm']
        max_num_waves = MAX_NUM_THREADS // WARP_SIZE
        occupancy = min(NUM_GPRS // WARP_SIZE // n_regs, max_num_waves) // num_warps
    else:
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    occupancy = min(occupancy, SIZE_SMEM // size_smem)
    num_programs = NUM_SM * occupancy

    num_programs = min(num_programs, n_rows)

    # Create a number of persistent programs.
    kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages)
    return y
```

### Unit Test

We can test our kernel on a matrix with an irregular number of rows and columns.

```python
torch.manual_seed(0)
x = torch.randn(1823, 781, device=DEVICE)
y_triton = softmax(x)
y_torch = torch.softmax(x, axis=1)
assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
```

### Benchmark

```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 100)],  # different possible values for `x_name`
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch', 'naive_softmax'],  # possible values for `line_arg``
        line_names=["Triton", "Torch", "Naive Softmax"],  # label name for the lines
        styles=[('blue', '-'), ('green', '-'), ('red', '-')],  # line styles
        ylabel="GB/s",  # label name for the y-axis
        plot_name="softmax-performance",  # name for the plot. Used also as a file name for saving the plot.
        args={'M': 4096},  # values for function arguments not in `x_names` and `y_name`
    ))
def benchmark(M, N, provider):
    x = torch.randn(M, N, device=DEVICE, dtype=torch.float32)
    stream = getattr(torch, DEVICE.type).Stream()
    getattr(torch, DEVICE.type).set_stream(stream)
    if provider == 'torch':
        ms = triton.testing.do_bench(lambda: torch.softmax(x, axis=-1))
    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: softmax(x))
    if provider == 'naive_softmax':
        ms = triton.testing.do_bench(lambda: naive_softmax(x))
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)


benchmark.run(show_plots=True, print_data=True)
```

![02 fused softmax](assets/sphx_glr_02-fused-softmax_001.png)

## Matrix Multiplication

This tutorial will write a very short high-performance FP16 matrix multiplication kernel that achieves performance on par with cuBLAS or rocBLAS.

- Block-level matrix multiplications
- Multi-dimensional pointer arithmetic
- Program re-ordering for improved L2 cache hit rate
- Automatic performance tuning

### Motivations

Matrix multiplications are a key building block of most modern high-performance computing systems. They are notoriously hard to optimize, hence their implementation is generally done by hardware vendors themselves as part of so-called “kernel libraries”(e.g., cuBLAS).

Unfortunately, these libraries are often propritary and cannot be easily customized to accommodate the needs of modern deep learning workloads. 

