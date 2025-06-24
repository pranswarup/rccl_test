#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <iostream>
#include <vector>
#include <mpi.h>
#include <cassert>

#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << " - " << hipGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

#define RCCL_CHECK(call) \
    do { \
        ncclResult_t err = call; \
        if (err != ncclSuccess) { \
            std::cerr << "RCCL error at " << __FILE__ << ":" << __LINE__ << " - " << ncclGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

#define MPI_CHECK(call) \
    do { \
        int err = call; \
        if (err != MPI_SUCCESS) { \
            std::cerr << "MPI error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(1); \
        } \
    } while(0)

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_CHECK(MPI_Init(&argc, &argv));
    
    int rank, size;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
    
    std::cout << "Process " << rank << " of " << size << " starting" << std::endl;
    
    // Set HIP device based on MPI rank
    int deviceCount;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));
    int device = rank % deviceCount;
    HIP_CHECK(hipSetDevice(device));
    
    std::cout << "Rank " << rank << " using GPU " << device << std::endl;
    
    // Initialize RCCL
    ncclUniqueId commId;
    if (rank == 0) {
        RCCL_CHECK(ncclGetUniqueId(&commId));
    }
    
    // Broadcast the communicator ID to all processes
    MPI_CHECK(MPI_Bcast(&commId, sizeof(commId), MPI_BYTE, 0, MPI_COMM_WORLD));
    
    // Create RCCL communicator
    ncclComm_t comm;
    RCCL_CHECK(ncclCommInitRank(&comm, size, commId, rank));
    
    // Create HIP stream
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    
    // Prepare data for collective operations
    const int dataSize = 1024;
    std::vector<float> hostData(dataSize);
    
    // Initialize data on each rank
    for (int i = 0; i < dataSize; i++) {
        hostData[i] = rank * 100.0f + i;
    }
    
    // Allocate device memory
    float* deviceData;
    float* deviceResult;
    HIP_CHECK(hipMalloc(&deviceData, dataSize * sizeof(float)));
    HIP_CHECK(hipMalloc(&deviceResult, dataSize * sizeof(float)));
    
    // Copy data to device
    HIP_CHECK(hipMemcpy(deviceData, hostData.data(), dataSize * sizeof(float), hipMemcpyHostToDevice));
    
    std::cout << "Rank " << rank << " initialized with data starting from " << hostData[0] << std::endl;
    
    // Perform AllReduce operation (sum across all ranks)
    RCCL_CHECK(ncclAllReduce(deviceData, deviceResult, dataSize, ncclFloat, ncclSum, comm, stream));
    
    // Synchronize
    HIP_CHECK(hipStreamSynchronize(stream));
    
    // Copy result back to host
    std::vector<float> hostResult(dataSize);
    HIP_CHECK(hipMemcpy(hostResult.data(), deviceResult, dataSize * sizeof(float), hipMemcpyDeviceToHost));
    
    // Verify results (first few elements)
    if (rank == 0) {
        std::cout << "AllReduce results (first 5 elements):" << std::endl;
        for (int i = 0; i < 5; i++) {
            float expected = 0.0f;
            for (int r = 0; r < size; r++) {
                expected += r * 100.0f + i;
            }
            std::cout << "  Element " << i << ": " << hostResult[i] << " (expected: " << expected << ")" << std::endl;
            if (abs(hostResult[i] - expected) > 1e-5) {
                std::cerr << "ERROR: Mismatch at element " << i << std::endl;
            }
        }
    }
    
    // Additional collective operation: AllGather
    std::vector<float> gatherHostData(dataSize);
    for (int i = 0; i < dataSize; i++) {
        gatherHostData[i] = rank + i * 0.1f;
    }
    
    float* deviceGatherInput;
    float* deviceGatherOutput;
    HIP_CHECK(hipMalloc(&deviceGatherInput, dataSize * sizeof(float)));
    HIP_CHECK(hipMalloc(&deviceGatherOutput, dataSize * size * sizeof(float)));
    
    HIP_CHECK(hipMemcpy(deviceGatherInput, gatherHostData.data(), dataSize * sizeof(float), hipMemcpyHostToDevice));
    
    // Perform AllGather operation
    RCCL_CHECK(ncclAllGather(deviceGatherInput, deviceGatherOutput, dataSize, ncclFloat, comm, stream));
    
    // Synchronize
    HIP_CHECK(hipStreamSynchronize(stream));
    
    // Copy AllGather result back to host
    std::vector<float> hostGatherResult(dataSize * size);
    HIP_CHECK(hipMemcpy(hostGatherResult.data(), deviceGatherOutput, dataSize * size * sizeof(float), hipMemcpyDeviceToHost));
    
    // Print AllGather results (first element from each rank)
    if (rank == 0) {
        std::cout << "AllGather results (first element from each rank):" << std::endl;
        for (int r = 0; r < size; r++) {
            std::cout << "  From rank " << r << ": " << hostGatherResult[r * dataSize] << std::endl;
        }
    }
    
    // Broadcast operation
    float broadcastValue = 42.0f;
    if (rank == 0) {
        std::cout << "Broadcasting value " << broadcastValue << " from rank 0" << std::endl;
    }
    
    float* deviceBroadcast;
    HIP_CHECK(hipMalloc(&deviceBroadcast, sizeof(float)));
    
    if (rank == 0) {
        HIP_CHECK(hipMemcpy(deviceBroadcast, &broadcastValue, sizeof(float), hipMemcpyHostToDevice));
    }
    
    // Perform Broadcast operation
    RCCL_CHECK(ncclBcast(deviceBroadcast, 1, ncclFloat, 0, comm, stream));
    
    // Synchronize
    HIP_CHECK(hipStreamSynchronize(stream));
    
    // Copy broadcast result back to host
    float receivedValue;
    HIP_CHECK(hipMemcpy(&receivedValue, deviceBroadcast, sizeof(float), hipMemcpyDeviceToHost));
    
    std::cout << "Rank " << rank << " received broadcast value: " << receivedValue << std::endl;
    
    // Cleanup
    HIP_CHECK(hipFree(deviceData));
    HIP_CHECK(hipFree(deviceResult));
    HIP_CHECK(hipFree(deviceGatherInput));
    HIP_CHECK(hipFree(deviceGatherOutput));
    HIP_CHECK(hipFree(deviceBroadcast));
    HIP_CHECK(hipStreamDestroy(stream));
    
    // Finalize RCCL
    RCCL_CHECK(ncclCommDestroy(comm));
    
    std::cout << "Rank " << rank << " completed successfully" << std::endl;
    
    // Finalize MPI
    MPI_CHECK(MPI_Finalize());
    
    return 0;
}