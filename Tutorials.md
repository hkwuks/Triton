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

![[vector_additon_performance.png]]