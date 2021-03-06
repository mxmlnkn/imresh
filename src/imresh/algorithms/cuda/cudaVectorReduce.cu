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
#include <limits>     // lowest
#include <cmath>
#include <cuda.h>     // atomicCAS
#include <cufft.h>    // cufftComplex, cufftDoubleComplex
#include "libs/cudacommon.h"


namespace imresh
{
namespace algorithms
{
namespace cuda
{

    SumFunctor<float > sumFunctorf;
    MinFunctor<float > minFunctorf;
    MaxFunctor<float > maxFunctorf;
    SumFunctor<double> sumFunctord;
    MinFunctor<double> minFunctord;
    MaxFunctor<double> maxFunctord;


    template<class T_PREC, class T_FUNC>
    __device__ inline void atomicFunc
    (
        T_PREC * const rdpTarget,
        const T_PREC rValue,
        T_FUNC f
    )
    {
        /* atomicCAS only is defined for int and long long int, thats why we
         * need these roundabout casts */
        int assumed;
        int old = * (int*) rdpTarget;

        /* atomicCAS returns the value with which the current value 'assumed'
         * was compared. If the value changed between reading out to assumed
         * and calculating the reduced value and storing it back, then we
         * need to call this function again. (I hope the GPU has some
         * functionality to prevent synchronized i.e. neverending races ... */
        do
        {
            assumed = old;

            /* If the reduced value doesn't change, then we don't need to hinder
             * other threads with atomicCAS. This additional check may prove a
             * bottleneck, if this is rarely the case, e.g. for sum and no 0s or
             * for max and an ordered list, where the largest is the last
             * element. In tests this more often slowed down the calculation */
            //if ( f( __int_as_float(assumed), rValue ) == assumed )
            //    break;

            /* compare and swap after the value was read with assumend, return
             * old value, if assumed isn't anymore the value at rdpTarget,
             * then we will have to try again to write it */
            old = atomicCAS( (int*) rdpTarget, assumed,
                __float_as_int( f( __int_as_float(assumed), rValue ) ) );
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


    /*
    // seems to work for testVectorReduce, but it shouldn't oO, maybe just good numbers, or because this is only for max, maybe it wouldn't work for min, because the maximum is > 0 ... In the end it isn't faster than atomicCAS and it doesn't even use floatAsOrderdInt yet, which would make use of bitshift, subtraction and logical or, thereby decreasing performance even more: http://stereopsis.com/radix.html
    template<>
    __device__ inline void atomicFunc<float,MaxFunctor<float>>
    (
        float * const rdpTarget,
        const float rValue,
        MaxFunctor<float> f
    )
    {
        atomicMax( (int*)rdpTarget, __float_as_int(rValue) );
    }*/


    template<class T_PREC, class T_FUNC>
    __global__ void kernelVectorReduceShared
    (
        const T_PREC * const rdpData,
        const unsigned rnData,
        T_PREC * const rdpResult,
        T_FUNC f,
        const T_PREC rInitValue
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
        const T_PREC * const rdpData,
        const unsigned rnData,
        T_PREC * const rdpResult,
        T_FUNC f,
        const T_PREC rInitValue
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
    __global__ void kernelVectorReduceWarps
    (
        const T_PREC * const rdpData,
        const unsigned rnData,
        T_PREC * const rdpResult,
        T_FUNC f,
        const T_PREC rInitValue
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

        /* reduce per warp (warpSize == 32 assumed) */
        const int32_t laneId = threadIdx.x % 32;
        #pragma unroll
        for ( int32_t warpDelta = 32 / 2; warpDelta > 0; warpDelta /= 2)
            localReduced = f( localReduced, __shfl_down( localReduced, warpDelta ) );

        if ( laneId == 0 )
            atomicFunc( rdpResult, localReduced, f );
    }


    template<class T_PREC, class T_FUNC>
    T_PREC cudaReduce
    (
        const T_PREC * const rdpData,
        const unsigned rnElements,
        T_FUNC f,
        const T_PREC rInitValue,
        cudaStream_t rStream
    )
    {
        const unsigned nThreads = 128;
        //const unsigned nBlocks  = ceil( (float) rnElements / nThreads );
        //printf( "nThreads = %i, nBlocks = %i\n", nThreads, nBlocks );
        const unsigned nBlocks = 288;
        /* 256*256 = 65536 concurrent threads should fill most modern graphic
         * cards. E.g. GTX 760 can only handle 12288 runnin concurrently,
         * everything else will be run after some threads finished. The
         * number of kernels is only 384, because of oversubscription with
         * warps */
        assert( nBlocks < 65536 );

        T_PREC reducedValue;
        T_PREC * dpReducedValue;
        T_PREC initValue = rInitValue;

        CUDA_ERROR( cudaMalloc( (void**) &dpReducedValue, sizeof(float) ) );
        CUDA_ERROR( cudaMemcpyAsync( dpReducedValue, &initValue, sizeof(float), cudaMemcpyHostToDevice, rStream ) );

        /* memcpy is on the same stream as kernel will be, so no synchronize needed! */
        kernelVectorReduceWarps<<< nBlocks, nThreads, 0, rStream >>>
            ( rdpData, rnElements, dpReducedValue, f, rInitValue );

        CUDA_ERROR( cudaStreamSynchronize( rStream ) );
        CUDA_ERROR( cudaMemcpyAsync( &reducedValue, dpReducedValue, sizeof(float), cudaMemcpyDeviceToHost, rStream ) );
        CUDA_ERROR( cudaStreamSynchronize( rStream) );
        CUDA_ERROR( cudaFree( dpReducedValue ) );

        return reducedValue;
    }


    template<class T_PREC, class T_FUNC>
    T_PREC cudaReduceSharedMemory
    (
        const T_PREC * const rdpData,
        const unsigned rnElements,
        T_FUNC f,
        const T_PREC rInitValue,
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

        CUDA_ERROR( cudaMalloc( (void**) &dpReducedValue, sizeof(float) ) );
        CUDA_ERROR( cudaMemcpyAsync( dpReducedValue, &initValue, sizeof(float), cudaMemcpyHostToDevice, rStream ) );

        /* memcpy is on the same stream as kernel will be, so no synchronize needed! */
        kernelVectorReduceShared<<< nBlocks, nThreads, 0, rStream >>>
            ( rdpData, rnElements, dpReducedValue, f, rInitValue );

        CUDA_ERROR( cudaStreamSynchronize( rStream ) );
        CUDA_ERROR( cudaMemcpyAsync( &reducedValue, dpReducedValue, sizeof(float), cudaMemcpyDeviceToHost, rStream ) );
        CUDA_ERROR( cudaStreamSynchronize( rStream) );
        CUDA_ERROR( cudaFree( dpReducedValue ) );

        return reducedValue;
    }


    template<class T_PREC, class T_FUNC>
    T_PREC cudaReduceSharedMemoryWarps
    (
        const T_PREC * const rdpData,
        const unsigned rnElements,
        T_FUNC f,
        const T_PREC rInitValue,
        cudaStream_t rStream
    )
    {
        const unsigned nThreads = 256;
        const unsigned nBlocks = 256;
        assert( nBlocks < 65536 );

        T_PREC reducedValue;
        T_PREC * dpReducedValue;
        T_PREC initValue = rInitValue;

        CUDA_ERROR( cudaMalloc( (void**) &dpReducedValue, sizeof(float) ) );
        CUDA_ERROR( cudaMemcpyAsync( dpReducedValue, &initValue, sizeof(float), cudaMemcpyHostToDevice, rStream ) );

        /* memcpy is on the same stream as kernel will be, so no synchronize needed! */
        kernelVectorReduceSharedMemoryWarps<<< nBlocks, nThreads, 0, rStream >>>
            ( rdpData, rnElements, dpReducedValue, f, rInitValue );

        CUDA_ERROR( cudaStreamSynchronize( rStream ) );
        CUDA_ERROR( cudaMemcpyAsync( &reducedValue, dpReducedValue, sizeof(float), cudaMemcpyDeviceToHost, rStream ) );
        CUDA_ERROR( cudaStreamSynchronize( rStream) );
        CUDA_ERROR( cudaFree( dpReducedValue ) );

        return reducedValue;
    }


    template<class T_PREC>
    T_PREC cudaVectorMin
    (
        const T_PREC * const rdpData,
        const unsigned rnElements,
        cudaStream_t rStream
    )
    {
        MinFunctor<T_PREC> minFunctor;
        return cudaReduce( rdpData, rnElements, minFunctor, std::numeric_limits<T_PREC>::max(), rStream );
    }


    template<class T_PREC>
    T_PREC cudaVectorMax
    (
        const T_PREC * const rdpData,
        const unsigned rnElements,
        cudaStream_t rStream
    )
    {
        MaxFunctor<T_PREC> maxFunctor;
        return cudaReduce( rdpData, rnElements, maxFunctor, std::numeric_limits<T_PREC>::lowest(), rStream );
    }


    template<class T_PREC>
    T_PREC cudaVectorSum
    (
        const T_PREC * const rdpData,
        const unsigned rnElements,
        cudaStream_t rStream
    )
    {
        SumFunctor<T_PREC> sumFunctor;
        return cudaReduce( rdpData, rnElements, sumFunctor, T_PREC(0), rStream );
    }


    /* These functions only persist for benchmarking purposes to show that
     * the standard version is the fastest */

    template<class T_PREC>
    T_PREC cudaVectorMaxSharedMemory
    (
        const T_PREC * const rdpData,
        const unsigned rnElements,
        cudaStream_t rStream
    )
    {
        MaxFunctor<T_PREC> maxFunctor;
        return cudaReduceSharedMemory( rdpData, rnElements, maxFunctor, std::numeric_limits<T_PREC>::lowest(), rStream );
    }

    template<class T_PREC>
    T_PREC cudaVectorMaxSharedMemoryWarps
    (
        const T_PREC * const rdpData,
        const unsigned rnElements,
        cudaStream_t rStream
    )
    {
        MaxFunctor<T_PREC> maxFunctor;
        return cudaReduceSharedMemoryWarps( rdpData, rnElements, maxFunctor, std::numeric_limits<T_PREC>::lowest(), rStream );
    }


    /**
     * "For the input-output algorithms the error E_F is
     *  usually meaningless since the input g_k(X) is no longer
     *  an estimate of the object. Then the meaningful error
     *  is the object-domain error E_0 given by Eq. (15)."
     *                                      (Fienup82)
     * Eq.15:
     * @f[ E_{0k}^2 = \sum\limits_{x\in\gamma} |g_k'(x)^2|^2 @f]
     * where \gamma is the domain at which the constraints are
     * not met. SO this is the sum over the domain which should
     * be 0.
     *
     * Eq.16:
     * @f[ E_{Fk}^2 = \sum\limits_{u} |G_k(u) - G_k'(u)|^2 / N^2
                    = \sum_x |g_k(x) - g_k'(x)|^2 @f]
     **/
    template< class T_COMPLEX, class T_MASK_ELEMENT >
    __global__ void cudaKernelCalculateHioError
    (
        const T_COMPLEX * const rdpgPrime,
        const T_MASK_ELEMENT * const rdpIsMasked,
        const unsigned rnData,
        const bool rInvertMask,
        float * const rdpTotalError,
        float * const rdpnMaskedPixels
    )
    {
        assert( blockDim.y == 1 );
        assert( blockDim.z == 1 );
        assert( gridDim.y  == 1 );
        assert( gridDim.z  == 1 );

        const int32_t nTotalThreads = gridDim.x * blockDim.x;
        int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        assert( i < nTotalThreads );

        float localTotalError    = 0;
        float localnMaskedPixels = 0;
        for ( ; i < rnData; i += nTotalThreads )
        {
            const auto & re = rdpgPrime[i].x;
            const auto & im = rdpgPrime[i].y;

            /* only add up norm where no object should be (rMask == 0) */
            /* note: invert   + masked   -> unmasked  <=> 1 ? 1 -> 0
             *       noinvert + masked   -> masked    <=> 0 ? 1 -> 1
             *       invert   + unmasked -> masked    <=> 1 ? 0 -> 1
             *       noinvert + unmasked -> unmasked  <=> 0 ? 0 -> 0
             *   => ? is xor    => no thread divergence
             */
            assert( rdpIsMasked[i] == 0 or rdpIsMasked[i] == 1 );
            const bool shouldBeZero = rInvertMask xor (bool) rdpIsMasked[i];
            assert( rdpIsMasked[i] >= 0.0 and rdpIsMasked[i] <= 1.0 );
            //float shouldBeZero = rInvertMask + ( 1-2*rInvertMask )*rdpIsMasked[i];
            /*
            float shouldBeZero = rdpIsMasked[i];
            if ( rInvertMask )
                shouldBeZero = 1 - shouldBeZero;
            */

            localTotalError    += shouldBeZero * ( re*re+im*im );
            localnMaskedPixels += shouldBeZero;
        }

        /* reduce per warp (warpSize == 32 assumed) */
        const int32_t laneId = threadIdx.x % 32;
        #pragma unroll
        for ( int32_t warpDelta = 32 / 2; warpDelta > 0; warpDelta /= 2 )
        {
            localTotalError    += __shfl_down( localTotalError   , warpDelta );
            localnMaskedPixels += __shfl_down( localnMaskedPixels, warpDelta );
        }
        SumFunctor<float> sum;
        if ( laneId == 0 )
        {
            atomicFunc( rdpTotalError   , localTotalError   , sum );
            atomicFunc( rdpnMaskedPixels, localnMaskedPixels, sum );
        }
    }


    template<class T_COMPLEX, class T_MASK_ELEMENT>
    float calculateHioError
    (
        const T_COMPLEX * const & rdpData,
        const T_MASK_ELEMENT * const & rdpIsMasked,
        const unsigned & rnElements,
        const bool & rInvertMask,
        cudaStream_t rStream
    )
    {
        const unsigned nThreads = 256;
        //const unsigned nBlocks  = ceil( (float) rnElements / nThreads );
        const unsigned nBlocks  = 256;
        assert( nBlocks < 65536 );

        float     totalError,     nMaskedPixels;
        float * dpTotalError, * dpnMaskedPixels;

        CUDA_ERROR( cudaMalloc( (void**) &dpTotalError   , sizeof(float) ) );
        CUDA_ERROR( cudaMalloc( (void**) &dpnMaskedPixels, sizeof(float) ) );
        CUDA_ERROR( cudaMemsetAsync( dpTotalError   , 0, sizeof(float), rStream ) );
        CUDA_ERROR( cudaMemsetAsync( dpnMaskedPixels, 0, sizeof(float), rStream ) );

        /* memset is on the same stream as kernel will be, so no synchronize needed! */
        cudaKernelCalculateHioError<<< nBlocks, nThreads, 0, rStream >>>
            ( rdpData, rdpIsMasked, rnElements, rInvertMask, dpTotalError, dpnMaskedPixels );
        CUDA_ERROR( cudaStreamSynchronize( rStream ) );

        CUDA_ERROR( cudaMemcpyAsync( &totalError   , dpTotalError   , sizeof(float), cudaMemcpyDeviceToHost, rStream ) );
        CUDA_ERROR( cudaMemcpyAsync( &nMaskedPixels, dpnMaskedPixels, sizeof(float), cudaMemcpyDeviceToHost, rStream ) );
        CUDA_ERROR( cudaStreamSynchronize( rStream ) );

        CUDA_ERROR( cudaFree( dpTotalError    ) );
        CUDA_ERROR( cudaFree( dpnMaskedPixels ) );

        return sqrtf(totalError) / nMaskedPixels;
    }


    /* explicit instantiations */

    template
    float cudaVectorMin<float>
    (
        const float * const rdpData,
        const unsigned rnElements,
        cudaStream_t rStream
    );
    template
    double cudaVectorMin<double>
    (
        const double * const rdpData,
        const unsigned rnElements,
        cudaStream_t rStream
    );


    template
    float cudaVectorMax<float>
    (
        const float * const rdpData,
        const unsigned rnElements,
        cudaStream_t rStream
    );
    template
    double cudaVectorMax<double>
    (
        const double * const rdpData,
        const unsigned rnElements,
        cudaStream_t rStream
    );


    template
    float cudaVectorSum<float>
    (
        const float * const rdpData,
        const unsigned rnElements,
        cudaStream_t rStream
    );
    template
    double cudaVectorSum<double>
    (
        const double * const rdpData,
        const unsigned rnElements,
        cudaStream_t rStream
    );

    template
    __global__ void cudaKernelCalculateHioError
    <cufftComplex, float>
    (
        const cufftComplex * const rdpgPrime,
        const float * const rdpIsMasked,
        const unsigned rnData,
        const bool rInvertMask,
        float * const rdpTotalError,
        float * const rdpnMaskedPixels
    );


    template
    float calculateHioError
    <cufftComplex, float>
    (
        const cufftComplex * const & rdpData,
        const float * const & rdpIsMasked,
        const unsigned & rnElements,
        const bool & rInvertMask,
        cudaStream_t rStream
    );


    template
    float cudaVectorMaxSharedMemory<float>
    (
        const float * const rdpData,
        const unsigned rnElements,
        cudaStream_t rStream
    );

    template
    float cudaVectorMaxSharedMemoryWarps<float>
    (
        const float * const rdpData,
        const unsigned rnElements,
        cudaStream_t rStream
    );


} // namespace cuda
} // namespace algorithms
} // namespace imresh
