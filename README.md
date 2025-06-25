# RCCL Test Example

This is a simple program that demonstrates the use of RCCL (ROCm Collective Communications Library) for multi-GPU collective operations.

## Overview

The program implements three key collective communication operations:
1. **AllReduce**: Performs a sum reduction across all participating GPUs
2. **AllGather**: Gathers data from all GPUs and distributes the complete dataset to each GPU
3. **Broadcast**: Broadcasts a value from rank 0 to all other ranks

## Prerequisites

- ROCm 5.0 or later with HIP
- RCCL (ROCm Collective Communications Library)
- MPI implementation (OpenMPI or MPICH)
- Multiple AMD GPUs (or single GPU for testing)

## Building

The project uses CMake for building. You can build it in several ways:

### Option 1: Manual CMake build
```bash
# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm .

# Build
cmake --build build

# Run
mpirun -np 2 ./rccl_test
```

## Running

### Basic Usage
```bash
# Run with 2 MPI processes
cd build && mpirun -np 2 ./rccl_test
```

### Environment Variables

You may need to set these environment variables:
```bash
export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
export HIP_VISIBLE_DEVICES=0,1,2,3  # Specify which GPUs to use
```

### Example Output

```
Process 0 of 2 starting
Process 1 of 2 starting
Rank 0 using GPU 0
Rank 1 using GPU 1
Rank 0 initialized with data starting from 0
Rank 1 initialized with data starting from 100
AllReduce results (first 5 elements):
  Element 0: 100 (expected: 100)
  Element 1: 102 (expected: 102)
  Element 2: 104 (expected: 104)
  Element 3: 106 (expected: 106)
  Element 4: 108 (expected: 108)
AllGather results (first element from each rank):
  From rank 0: 0
  From rank 1: 1
Broadcasting value 42 from rank 0
Rank 0 received broadcast value: 42
Rank 1 received broadcast value: 42
Rank 0 completed successfully
Rank 1 completed successfully
```

## Code Structure

### Key Components

1. **MPI Initialization**: Sets up multi-process communication
2. **HIP Device Setup**: Assigns GPUs to MPI ranks
3. **RCCL Communicator**: Creates communication context for collective operations
4. **Memory Management**: Allocates and manages GPU memory
5. **Collective Operations**: Demonstrates AllReduce, AllGather, and Broadcast
6. **Error Handling**: Comprehensive error checking with custom macros

### Collective Operations Explained

- **AllReduce**: Each rank contributes data, and the result (sum) is available on all ranks
- **AllGather**: Each rank contributes data, and the concatenated data from all ranks is available on all ranks
- **Broadcast**: One rank (root) sends data to all other ranks

## Troubleshooting

### Common Issues

1. **"RCCL not found"**: Ensure ROCm is properly installed and RCCL libraries are in the library path
2. **"No GPUs available"**: Check that AMD GPUs are detected with `rocm-smi`
3. **MPI errors**: Verify MPI installation and that mpirun is in PATH
4. **Memory errors**: Ensure sufficient GPU memory is available

### Debug Tips

- Use `rocm-smi` to check GPU status
- Run with fewer processes if you have limited GPUs
- Check CMake configuration: `cmake --build build --target help`
- Enable HIP debugging: `export HIP_VISIBLE_DEVICES=0` for single GPU testing
- For debugging: `./build.sh -t Debug` then `cd build && mpirun -np 1 gdb ./hello_cuda`

## Extending the Example

This example can be extended to:
- Add more collective operations (Reduce, Scatter, etc.)
- Implement custom data types
- Add performance benchmarking
- Test with different data sizes
- Implement fault tolerance mechanisms

## References

- [RCCL Documentation](https://rccl.readthedocs.io/)
- [ROCm Documentation](https://rocmdocs.amd.com/)
- [HIP Programming Guide](https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-GUIDE.html)
