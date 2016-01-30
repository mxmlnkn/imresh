/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2016 Maximilian Knespel
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "cudaVectorReduce.hpp"

#include <cassert>
#include <cstdint>    // uint64_t
#include <limits>     // numeric_limits
#include <cuda.h>     // atomicCAS, atomicAdd
#include <cufft.h>    // cufftComplex, cufftDoubleComplex
#include "libs/cudacommon.h"
/**
 * Gives only compile errors, e.g.
 *    ptxas fatal   : Unresolved extern function '_ZN6imresh10algorithms4cuda10SumFunctorIfEclEff'
 * so I justd copy-pasted the functors here ...
 **/
//#include "algorithms/cuda/cudaVectorReduce.hpp" // maxFunctor, atomicFunc


namespace benchmark
{
namespace imresh
{
namespace algorithms
{
namespace cuda
{


    template<class T_PREC, class T_FUNC>
    __device__ inline void atomicFunc
    (
        T_PREC * rdpTarget,
        T_PREC rValue,
        T_FUNC f
    );

    template<class T_FUNC>
    __device__ inline void atomicFunc
    (
        float * rdpTarget,
        float rValue,
        T_FUNC f
    );

    template<class T_FUNC>
    __device__ inline void atomicFunc
    (
        double * rdpTarget,
        double rValue,
        T_FUNC f
    );

    /**
     * simple functors to just get the sum of two numbers. To be used
     * for the binary vectorReduce function to make it a vectorSum or
     * vectorMin or vectorMax
     **/
    template<class T> struct SumFunctor {
        __device__ __host__ inline T operator() ( T a, T b )
        { return a+b; }
    };
    template<class T> struct MinFunctor {
        __device__ __host__ inline T operator() ( T a, T b )
        { if (a<b) return a; else return b; } // std::min not possible, can't call host function from device!
    };
    template<class T> struct MaxFunctor {
        __device__ __host__ inline T operator() ( T a, T b )
        { if (a>b) return a; else return b; }
    };
    template<> struct MaxFunctor<float> {
        __device__ __host__ inline float operator() ( float a, float b )
        { return fmax(a,b); }
    };


    template<class T_FUNC>
    __device__ inline void atomicFunc
    (
        float * const rdpTarget,
        const float rValue,
        T_FUNC f
    )
    {
        uint32_t assumed;
        uint32_t old = * (uint32_t*) rdpTarget;

        do
        {
            assumed = old;
            old = atomicCAS( (uint32_t*) rdpTarget, assumed,
                __float_as_int( f( __int_as_float(assumed), rValue ) ) );
        }
        while ( assumed != old );
    }

    template<class T_FUNC>
    __device__ inline void atomicFunc
    (
        double * const rdpTarget,
        const double rValue,
        T_FUNC f
    )
    {
        using ull = unsigned long long int;
        ull assumed;
        ull old = * (ull*) rdpTarget;
        do
        {
            assumed = old;
            old = atomicCAS( (ull*) rdpTarget, assumed,
                __double_as_longlong( f( __longlong_as_double(assumed), rValue ) ) );
        }
        while ( assumed != old );
    }


    template<>
    __device__ inline void atomicFunc<int,MaxFunctor<int>>
    (
        int * const rdpTarget,
        const int rValue,
        MaxFunctor<int> f
    )
    {
        atomicMax( rdpTarget, rValue );
    }


    SumFunctor<float > sumFunctorf;
    MinFunctor<float > minFunctorf;
    MaxFunctor<float > maxFunctorf;
    SumFunctor<double> sumFunctord;
    MinFunctor<double> minFunctord;
    MaxFunctor<double> maxFunctord;


    template<class T_PREC, class T_FUNC>
    __global__ void kernelVectorReduceShared
    (
        T_PREC const * const rdpData,
        unsigned int const rnData,
        T_PREC * const rdpResult,
        T_FUNC f,
        T_PREC const rInitValue
    )
    {
        assert( blockDim.y == 1 );
        assert( blockDim.z == 1 );
        assert( gridDim.y  == 1 );
        assert( gridDim.z  == 1 );

        const int32_t nTotalThreads = gridDim.x * blockDim.x;
        int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        assert( i < nTotalThreads );

        T_PREC localReduced = T_PREC(rInitValue);
        for ( ; i < rnData; i += nTotalThreads )
            localReduced = f( localReduced, rdpData[i] );

        __shared__ T_PREC smReduced;
        /* master thread of every block shall set shared mem variable to 0 */
        __syncthreads();
        if ( threadIdx.x == 0 )
            smReduced = T_PREC(rInitValue);
        __syncthreads();

        atomicFunc( &smReduced, localReduced, f );

        __syncthreads();
        if ( threadIdx.x == 0 )
            atomicFunc( rdpResult, smReduced, f );
    }


    /**
     * benchmarks suggest that this kernel is twice as fast as
     * kernelVectorReduceShared
     **/
    template<class T_PREC, class T_FUNC>
    __global__ void kernelVectorReduceSharedMemoryWarps
    (
        T_PREC const * const rdpData,
        unsigned int const rnData,
        T_PREC * const rdpResult,
        T_FUNC f,
        T_PREC const rInitValue
    )
    {
        assert( blockDim.y == 1 );
        assert( blockDim.z == 1 );
        assert( gridDim.y  == 1 );
        assert( gridDim.z  == 1 );

        const int32_t nTotalThreads = gridDim.x * blockDim.x;
        int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        assert( i < nTotalThreads );

        T_PREC localReduced = T_PREC(rInitValue);
        for ( ; i < rnData; i += nTotalThreads )
            localReduced = f( localReduced, rdpData[i] );

        /**
         * reduce per warp:
         * With __shfl_down we can read the register values of other lanes in
         * a warp. In the first iteration lane 0 will add to it's value the
         * value of lane 16, lane 1 from lane 17 and so in.
         * In the next step lane 0 will add the result from lane 8.
         * In the end lane 0 will have the reduced value.
         * @see http://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
         **/
        constexpr int warpSize = 32;
        const int32_t laneId = threadIdx.x % warpSize;
        for ( int32_t warpDelta = warpSize / 2; warpDelta > 0; warpDelta /= 2)
            localReduced = f( localReduced, __shfl_down( localReduced, warpDelta ) );

        __shared__ T_PREC smReduced;
        /* master thread of every block shall set shared mem variable to 0 */
        __syncthreads();
        if ( threadIdx.x == 0 )
            smReduced = T_PREC(rInitValue);
        __syncthreads();

        if ( laneId == 0 )
            atomicFunc( &smReduced, localReduced, f );

        __syncthreads();
        if ( threadIdx.x == 0 )
            atomicFunc( rdpResult, smReduced, f );
    }


    template<class T_PREC, class T_FUNC>
    T_PREC cudaReduceSharedMemory
    (
        T_PREC const * const rdpData,
        unsigned int const rnElements,
        T_FUNC f,
        T_PREC const rInitValue,
        cudaStream_t rStream
    )
    {
        /* the more threads we have the longer the reduction will be
         * done inside shared memory instead of global memory */
        const unsigned nThreads = 256;
        const unsigned nBlocks = 256;
        assert( nBlocks < 65536 );

        T_PREC reducedValue;
        T_PREC * dpReducedValue;
        T_PREC initValue = rInitValue;

        CUDA_ERROR( cudaMalloc( (void**) &dpReducedValue, sizeof(T_PREC) ) );
        CUDA_ERROR( cudaMemcpyAsync( dpReducedValue, &initValue, sizeof(T_PREC),
                                     cudaMemcpyHostToDevice, rStream ) );

        /* memcpy is on the same stream as kernel will be, so no synchronize needed! */
        kernelVectorReduceShared<<< nBlocks, nThreads, 0, rStream >>>
            ( rdpData, rnElements, dpReducedValue, f, rInitValue );

        CUDA_ERROR( cudaStreamSynchronize( rStream ) );
        CUDA_ERROR( cudaMemcpyAsync( &reducedValue, dpReducedValue, sizeof(T_PREC),
                                     cudaMemcpyDeviceToHost, rStream ) );
        CUDA_ERROR( cudaStreamSynchronize( rStream) );
        CUDA_ERROR( cudaFree( dpReducedValue ) );

        return reducedValue;
    }


    template<class T_PREC, class T_FUNC>
    T_PREC cudaReduceSharedMemoryWarps
    (
        T_PREC const * const rdpData,
        unsigned int const rnElements,
        T_FUNC f,
        T_PREC const rInitValue,
        cudaStream_t rStream
    )
    {
        const unsigned nThreads = 256;
        const unsigned nBlocks = 256;
        assert( nBlocks < 65536 );

        T_PREC reducedValue;
        T_PREC * dpReducedValue;
        T_PREC initValue = rInitValue;

        CUDA_ERROR( cudaMalloc( (void**) &dpReducedValue, sizeof(T_PREC) ) );
        CUDA_ERROR( cudaMemcpyAsync( dpReducedValue, &initValue, sizeof(T_PREC),
                                     cudaMemcpyHostToDevice, rStream ) );

        /* memcpy is on the same stream as kernel will be, so no synchronize needed! */
        kernelVectorReduceSharedMemoryWarps<<< nBlocks, nThreads, 0, rStream >>>
            ( rdpData, rnElements, dpReducedValue, f, rInitValue );

        CUDA_ERROR( cudaStreamSynchronize( rStream ) );
        CUDA_ERROR( cudaMemcpyAsync( &reducedValue, dpReducedValue, sizeof(T_PREC),
                                     cudaMemcpyDeviceToHost, rStream ) );
        CUDA_ERROR( cudaStreamSynchronize( rStream) );
        CUDA_ERROR( cudaFree( dpReducedValue ) );

        return reducedValue;
    }


    template<class T_PREC>
    T_PREC cudaVectorMaxSharedMemory
    (
        T_PREC const * const rdpData,
        unsigned int const rnElements,
        cudaStream_t rStream
    )
    {
        MaxFunctor<T_PREC> maxFunctor;
        return cudaReduceSharedMemory( rdpData, rnElements, maxFunctor,
                                       std::numeric_limits<T_PREC>::lowest(),
                                       rStream );
    }


    template<class T_PREC>
    T_PREC cudaVectorMaxSharedMemoryWarps
    (
        T_PREC const * const rdpData,
        unsigned int const rnElements,
        cudaStream_t rStream
    )
    {
        MaxFunctor<T_PREC> maxFunctor;
        return cudaReduceSharedMemoryWarps( rdpData, rnElements, maxFunctor,
                                            std::numeric_limits<T_PREC>::lowest(),
                                            rStream );
    }


    /**
     * @see cudaKernelCalculateHioError
     **/
    template<class T_COMPLEX>
    __global__ void cudaKernelCalculateHioErrorBitPacked
    (
        T_COMPLEX  const * const __restrict__ rdpgPrime,
        int        const * const __restrict__ rdpIsMasked,
        unsigned int const rnData,
        float * const __restrict__ rdpTotalError,
        float * const __restrict__ rdpnMaskedPixels
    )
    {
        /**
         * @see http://www.pixel.io/blog/2012/4/19/does-anyone-actually-use-cudas-built-in-warpsize-variable.html
         * warpSize will be read with some assembler isntruction, therefore
         * it is not known at compile time, meaning some optimizations like
         * loop unrolling won't work. That's the reason for this roundabout way
         **/
        constexpr int cWarpSize = 32;
        static_assert( cWarpSize == 8 * sizeof( rdpIsMasked[0] ), "" );
        assert( cWarpSize  == warpSize );
        assert( blockDim.x == cWarpSize );
        assert( blockDim.y == 1 );
        assert( blockDim.z == 1 );
        assert( gridDim.y  == 1 );
        assert( gridDim.z  == 1 );

        const int nTotalThreads = gridDim.x * blockDim.x;
        int i = blockIdx.x * blockDim.x + threadIdx.x;

        float localTotalError    = 0;
        float localnMaskedPixels = 0;
        for ( ; i < rnData; i += nTotalThreads )
        {
            const auto re = rdpgPrime[i].x;
            const auto im = rdpgPrime[i].y;

            const bool shouldBeZero = rdpIsMasked[i];
            assert( rdpIsMasked[i] >= 0.0 and rdpIsMasked[i] <= 1.0 );

            localTotalError    += shouldBeZero * ( re*re+im*im );
            localnMaskedPixels += shouldBeZero;
        }

        const int laneId = threadIdx.x % cWarpSize;
        #pragma unroll
        for ( int warpDelta = cWarpSize / 2; warpDelta > 0; warpDelta /= 2 )
        {
            localTotalError    += __shfl_down( localTotalError   , warpDelta );
            localnMaskedPixels += __shfl_down( localnMaskedPixels, warpDelta );
        }

        if ( laneId == 0 )
        {
            atomicAdd( rdpTotalError   , localTotalError    );
            atomicAdd( rdpnMaskedPixels, localnMaskedPixels );
        }
    }


    /* implicit template instantiations */

    float _instantiateAllTemplatesCudaVectorReduceBenchmark( void )
    {
        return
        cudaReduceSharedMemory( (float*) NULL, 0, sumFunctorf, 0.0f ) +
        cudaReduceSharedMemory( (float*) NULL, 0, minFunctorf, 0.0f ) +
        cudaReduceSharedMemory( (float*) NULL, 0, maxFunctorf, 0.0f ) +
        cudaReduceSharedMemory( (double*) NULL, 0, sumFunctord, 0.0 ) +
        cudaReduceSharedMemory( (double*) NULL, 0, minFunctord, 0.0 ) +
        cudaReduceSharedMemory( (double*) NULL, 0, maxFunctord, 0.0 ) +
        cudaReduceSharedMemoryWarps( (float*) NULL, 0, sumFunctorf, 0.0f ) +
        cudaReduceSharedMemoryWarps( (float*) NULL, 0, minFunctorf, 0.0f ) +
        cudaReduceSharedMemoryWarps( (float*) NULL, 0, maxFunctorf, 0.0f ) +
        cudaReduceSharedMemoryWarps( (double*) NULL, 0, sumFunctord, 0.0 ) +
        cudaReduceSharedMemoryWarps( (double*) NULL, 0, minFunctord, 0.0 ) +
        cudaReduceSharedMemoryWarps( (double*) NULL, 0, maxFunctord, 0.0 ) +
        cudaVectorMaxSharedMemory( (float*) NULL, 0 ) +
        cudaVectorMaxSharedMemory( (double*) NULL, 0 ) +
        cudaVectorMaxSharedMemoryWarps( (float*) NULL, 0 ) +
        cudaVectorMaxSharedMemoryWarps( (double*) NULL, 0 ) +
        0;
    }


    template
    float cudaVectorMaxSharedMemory<float>
    (
        float const * const rdpData,
        unsigned int const rnElements,
        cudaStream_t rStream
    );

    template
    float cudaVectorMaxSharedMemoryWarps<float>
    (
        float const * const rdpData,
        unsigned int const rnElements,
        cudaStream_t rStream
    );


    template
    __global__ void cudaKernelCalculateHioErrorBitPacked
    <cufftComplex>
    (
        cufftComplex const * rdpgPrime,
        int          const * rdpIsMasked,
        unsigned int const rnData,
        float * rdpTotalError,
        float * rdpnMaskedPixels
    );


} // namespace cuda
} // namespace algorithms
} // namespace imresh
} // namespace benchmark