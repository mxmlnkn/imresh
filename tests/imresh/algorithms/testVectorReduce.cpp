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


#include "testVectorReduce.hpp"

#include <iostream>
#include <iomanip>
#include <cassert>
#include <cstdlib>          // srand, rand
#include <cstdint>          // uint32_t, uint64_t
#include <chrono>
#include <limits>
#include <vector>
#include <cmath>
#include <cfloat>           // FLT_MAX
#include <bitset>
#include <cuda_to_cupla.hpp>
#include "libs/cufft_to_cupla.hpp"  // cufftComplex
#ifdef USE_FFTW
#   include <fftw3.h>
#   include "libs/hybridInputOutput.hpp"
#endif
#include "algorithms/vectorReduce.hpp"
#include "algorithms/cuda/cudaVectorReduce.hpp"
#include "benchmark/imresh/algorithms/cuda/cudaVectorReduce.hpp"
#include "libs/cudacommon.hpp"
#include "benchmarkHelper.hpp"


namespace imresh
{
namespace algorithms
{


    unsigned int constexpr nRepetitions = 10;


    template<class T_PREC>
    bool compareFloat( const char * file, int line, T_PREC a, T_PREC b, T_PREC marginFactor = 1.0 )
    {
        auto const max = std::max( std::abs(a), std::abs(b) );
        if ( max == 0 )
            return true; // both are 0 and therefore equal
        auto const relErr = fabs( a - b ) / max;
        auto const maxRelErr = marginFactor * std::numeric_limits<T_PREC>::epsilon();
        if ( not ( relErr <= maxRelErr ) )
            printf( "[%s:%i] relErr: %f > %f :maxRelErr!\n", file, line, relErr, maxRelErr );
        return relErr <= maxRelErr;
    }


    void testVectorReduce( void )
    {
        using namespace std::chrono;
        using namespace benchmark::imresh::algorithms::cuda;
        using namespace imresh::algorithms;
        using namespace imresh::algorithms::cuda;
        using namespace imresh::libs;

        const unsigned nMaxElements = 16*1024*1024; // 64*1024*1024;  // ~4000x4000 pixel
        auto pData = new float[nMaxElements];

        srand(350471643);
        for ( unsigned i = 0; i < nMaxElements; ++i )
            pData[i] = ( (float) rand() / RAND_MAX ) - 0.5f;
        float * dpData;
        mallocCudaArray( &dpData, nMaxElements );
        CUDA_ERROR( cudaMemcpy( dpData, pData, nMaxElements*sizeof(dpData[0]), cudaMemcpyHostToDevice ) );

        /* Test for array of length 1 */
        assert( vectorMin( pData, 1 ) == pData[0] );
        assert( vectorMax( pData, 1 ) == pData[0] );
        assert( vectorSum( pData, 1 ) == pData[0] );
        assert( cudaVectorMin( CudaKernelConfig(), dpData, 1 ) == pData[0] );
        assert( cudaVectorMax( CudaKernelConfig(), dpData, 1 ) == pData[0] );
        assert( cudaVectorSum( CudaKernelConfig(), dpData, 1 ) == pData[0] );

        /* do some checks with longer arrays and obvious results */
        float obviousMaximum = 7.37519;
        float obviousMinimum =-7.37519;
        /* in order to filter out page time outs or similarily long random wait
         * times, we repeat the measurement nRepetitions times and choose the
         * shortest duration measured */

        using clock = std::chrono::high_resolution_clock;

        std::cout << "# Timings are in milliseconds, but note that measurements are repeated " << nRepetitions << " times, meaning they take that much longer than the value displayed" << std::endl;
        std::cout <<
/*            (1)        (2)       (3)       (4)       (5)       (6)       (7)       (8)       (9)     */
"# vector :          | local   |         | local + |         |         |         | minimum | minimum |\n"
"# length :          | reduce+ | ibid.   | shared+ |  ibid.  |         | #pragma |         | #pragma |\n"
"#        : global   | global  | pointer | global  |(old warp| chosen  | omp     | chosen_ | omp     |\n"
"#        : atomic   | atomic  | arithm. | atomic  | reduce )|  one    | reduce  | one     | reduce  |\n"
"---------:----------+---------+---------+---------+---------+---------+---------+---------+---------+"
        << std::endl;

        using namespace imresh::tests;
        for ( auto nElements : getLogSpacedSamplingPoints( 2, nMaxElements, 50 ) )
        {
            std::cout << std::setw(8) << nElements << " : ";
            float milliseconds, minTime;
            decltype( clock::now() ) clock0, clock1;

            int iObviousValuePos = rand() % nElements;
            // std::cout << "iObviousValuePos = " << iObviousValuePos << "\n";
            // std::cout << "nElements        = " << nElements << "\n";

            /* Maximum */
            pData[iObviousValuePos] = obviousMaximum;
            CUDA_ERROR( cudaMemcpy( dpData, pData, nElements*sizeof(dpData[0]), cudaMemcpyHostToDevice ) );

            #define TIME_KERNEL( FUNC, OBVIOUS_VALUE )                          \
            {                                                                \
                minTime = FLT_MAX;                                           \
                for ( unsigned iRepetition = 0; iRepetition < nRepetitions;  \
                      ++iRepetition )                                        \
                {                                                            \
                    clock0 = clock::now();                                   \
                    auto cudaReduced = FUNC( CudaKernelConfig(), pData,      \
                                             nElements );                    \
                    clock1 = clock::now();                                   \
                    auto seconds = duration_cast<duration<float>>(           \
                                        clock1 - clock0 );                   \
                    minTime = std::fmin( minTime, seconds.count() * 1000 );  \
                    assert( cudaReduced == OBVIOUS_VALUE );                  \
                }                                                            \
                std::cout << std::setw(8) << minTime << " |" << std::flush;  \
            }

            if ( nElements < 1e6 )
                TIME_KERNEL( cudaVectorMaxGlobalAtomic2, obviousMaximum ) /* (1) */
            else
                std::cout << std::setw(8) << -1 << " |" << std::flush;
            TIME_KERNEL( cudaVectorMaxGlobalAtomic     , obviousMaximum ) /* (2) */
            TIME_KERNEL( cudaVectorMaxPointer          , obviousMaximum ) /* (3) */
            TIME_KERNEL( cudaVectorMaxSharedMemory     , obviousMaximum ) /* (4) */
            TIME_KERNEL( cudaVectorMaxSharedMemoryWarps, obviousMaximum ) /* (5) */
            TIME_KERNEL( cudaVectorMax                 , obviousMaximum ) /* (6) */

            /* time CPU */
            #define TIME_CPU( FUNC, OBVIOUS_VALUE )                          \
            {                                                                \
                minTime = FLT_MAX;                                           \
                for ( unsigned iRepetition = 0; iRepetition < nRepetitions;  \
                      ++iRepetition )                                        \
                {                                                            \
                    clock0 = clock::now();                                   \
                    auto cpuMax = FUNC( pData, nElements );                  \
                    clock1 = clock::now();                                   \
                    auto seconds = duration_cast<duration<float>>(           \
                                        clock1 - clock0 );                   \
                    minTime = std::fmin( minTime, seconds.count() * 1000 );  \
                    assert( cpuMax == OBVIOUS_VALUE );                       \
                }                                                            \
                std::cout << std::setw(8) << minTime << " |" << std::flush;  \
            }
            TIME_CPU( vectorMax, obviousMaximum )        /* (7) */

            /* Minimum */
            pData[iObviousValuePos] = obviousMinimum;
            CUDA_ERROR( cudaMemcpy( dpData, pData, nElements*sizeof(dpData[0]), cudaMemcpyHostToDevice ) );

            TIME_KERNEL( cudaVectorMin, obviousMinimum ) /* (8) */
            TIME_CPU( vectorMin, obviousMinimum )        /* (9) */

            /* set obvious value back to random value */
            pData[iObviousValuePos] = (float) rand() / RAND_MAX;
            std::cout << "\n";

            #undef TIME_KERNEL
            #undef TIME_CPU
        }

        CUDA_ERROR( cudaFree( dpData ) );
        delete[] pData;
    }



    template<class T_MASK, class T_PACKED>
    __attribute__(( optimize("unroll-loops") ))
    void unpackBitMask
    (
        T_MASK         * const __restrict__ rMask,
        T_PACKED const * const __restrict__ rPackedBits,
        unsigned int const nElements
    )
    {
        auto const nElem = rMask + nElements;
        auto constexpr nBits = sizeof( T_PACKED ) * 8u;
        auto iPacked = rPackedBits;

        for ( auto iElem = rMask; iElem < nElem; ++iPacked )
        {
            auto bitMask = T_PACKED(0x01) << ( nBits-1 );

            for ( auto iBit = 0u; iBit < nBits; ++iBit, ++iElem )
            {
                if ( iElem >= nElem )
                    break;

                assert( bitMask != T_MASK(0) );
                assert( iElem < rMask + nElements );
                assert( iPacked < rPackedBits + ceilDiv( nElements, nBits ) );

                *iElem = T_MASK( (*iPacked & bitMask) != 0 );
                bitMask >>= 1;
            }
        }
    }

    inline uint32_t bfe
    (
        uint32_t src,
        uint32_t offset,
        uint32_t nBits
    )
    {
        return ( ( uint32_t(0xFFFFFFFF) >> (32-nBits) ) << offset ) & src;
    }

    void testUnpackBitMask( void )
    {
        uint32_t packed = 0x33333333;
        constexpr auto nElements = 8 * sizeof( packed );
        bool unpacked[ nElements ];
        unpacked[ nElements-2 ] = 1;
        unpacked[ nElements-1 ] = 0;
        unpackBitMask( unpacked, &packed, nElements-2 );

        for ( auto i = 0u; i < (nElements-2)/2; ++i )
        {
            assert( unpacked[2*i+0] == i % 2 );
            assert( unpacked[2*i+1] == i % 2 );
        }
        assert( unpacked[ nElements-2 ] == 1 );
        assert( unpacked[ nElements-1 ] == 0 );
    }

    void testCalculateHioError( void )
    {
        using namespace std::chrono;
        using namespace benchmark::imresh::algorithms::cuda;   // cudaCalculateHioErrorBitPacked
        using namespace imresh::algorithms::cuda;   // cudaKernelCalculateHioError
        using namespace imresh::libs;               // calculateHioError, mallocCudaArray
        using namespace imresh::tests;              // getLogSpacedSamplingPoints

        const unsigned nMaxElements = 16*1024*1024;  // ~4000x4000 pixel

        /* allocate */
        cufftComplex * dpData, * pData;
        unsigned char * dpIsMaskedChar, * pIsMaskedChar;
        float         * dpIsMasked    , * pIsMasked;
        unsigned      * dpBitMasked   , * pBitMasked;
        auto const nBitMaskedElements = ceilDiv( nMaxElements, 8 * sizeof( dpBitMasked[0] ) );
        mallocCudaArray( &dpIsMaskedChar, nMaxElements       );
        mallocCudaArray( &dpData        , nMaxElements       );
        mallocCudaArray( &dpIsMasked    , nMaxElements       );
        mallocCudaArray( &dpBitMasked   , nBitMaskedElements );
        pData         = new cufftComplex [ nMaxElements ];
        pIsMaskedChar = new unsigned char[ nMaxElements ];
        pIsMasked     = new float        [ nMaxElements ];
        pBitMasked    = new unsigned[ nBitMaskedElements ];
        /* allocate result buffer for reduced values of calculateHioError
         * kernel call */
        float nMaskedPixels = 0;
        float totalError = 0;

        /* initialize mask randomly */
        assert( sizeof(int) == 4 );
        srand(350471643);
        for ( auto i = 0u; i < nBitMaskedElements; ++i )
            pBitMasked[i] = rand() % UINT_MAX;
        unpackBitMask( pIsMasked, pBitMasked, nMaxElements );
        for ( auto i = 0u; i < nMaxElements; ++i )
        {
            pIsMaskedChar[i] = pIsMasked[i];
            assert( pIsMaskedChar[i] == 0 or pIsMaskedChar[i] == 1 );
        }
        /* visual check if bitpacking works */
        std::cout << "[unpacked] ";
        for ( int i = 0; i < 32; ++i )
            std::cout << pIsMasked[i];
        std::cout << std::endl;
        std::cout << "[  packed] " << std::bitset<32>( pBitMasked[0] ) << "\n";
        std::cout << "[     bfe] ";
        for ( int i = 0; i < 32; ++i )
        {
            unsigned int nBits = 32;
            bool const shouldBeZero = bfe(
                    pBitMasked[ i/nBits ], nBits-1 - i, 1 );
            std::cout << ( shouldBeZero ? '1' : '0' );
        }
        std::cout << std::endl;


        struct {
            float a = 3.0f;
            float b = 4.0f;
            float c = 5.0f;
        } pythagoreanTriple;
        /* initialize masked data with Pythagorean triple 3*3 + 4*4 = 5*5
         * and unmaksed with random values */
        for ( auto i = 0u; i < nMaxElements; ++i )
        {
            if ( pIsMasked[i] )
            {
                pData[i].x = pythagoreanTriple.a;
                pData[i].y = pythagoreanTriple.b;
            }
            else
            {
                pData[i].x = (float) rand() / RAND_MAX;
                pData[i].y = (float) rand() / RAND_MAX;
            }
        }
        /* if calculateHioError works correctly then we simply get
         * #masked * 5 as the mean complex norm error */

        /* push to GPU */
        CUDA_ERROR( cudaMemcpy( dpData     , pData     , nMaxElements * sizeof( pData    [0] ), cudaMemcpyHostToDevice ) );
        CUDA_ERROR( cudaMemcpy( dpIsMasked , pIsMasked , nMaxElements * sizeof( pIsMasked[0] ), cudaMemcpyHostToDevice ) );
        CUDA_ERROR( cudaMemcpy( dpBitMasked, pBitMasked, nBitMaskedElements * sizeof( pBitMasked[0] ), cudaMemcpyHostToDevice ) );
        CUDA_ERROR( cudaMemcpy( dpIsMaskedChar, pIsMaskedChar, nMaxElements * sizeof( pIsMaskedChar[0] ), cudaMemcpyHostToDevice ) );

        std::cout << "test with randomly masked pythagorean triples";
        /* - because the number of elements we include only increases, the
         *   number of found masked elements should also only increase.
         *   from iteration to iteration
         * - calculateHioError sums up the complex norm of masked pixels,
         *   the complex norm for the pythagorean triple 3,4,5 should be 5
         *   Check if totalError is nMaskedPixels * 5 */
        float nLastMaskedPixels = 0; /* found masked pixels in last iteration (with less nElements) */
        for ( auto nElements : getLogSpacedSamplingPoints( 2, nMaxElements, 50 ) )
        {
            std::cout << "." << std::flush;
            nLastMaskedPixels = nMaskedPixels;

            cudaCalculateHioError(
                CudaKernelConfig(3,256),
                dpData, dpIsMasked, nElements, false /* don't invert mask */,
                &totalError, &nMaskedPixels
            );
            /* Calculation done, now check if everything is correct */
            if ( totalError < 16777216 ) // float values higher 16777216 round to multiple of 2
            {
                assert( nLastMaskedPixels <= nMaskedPixels );
                assert( (unsigned) totalError % (int) pythagoreanTriple.c == 0 );
                assert( nMaskedPixels * pythagoreanTriple.c == totalError );
                assert( nMaskedPixels <= nElements );
            }

            /* check char version */
            cudaCalculateHioError(
                CudaKernelConfig(3,256),
                dpData, dpIsMaskedChar, nElements, false /* don't invert mask */,
                &totalError, &nMaskedPixels
            );
            /* Calculation done, now check if everything is correct */
            if ( totalError < 16777216 ) // float values higher than 16777216 round to multiple of 2
            {
                assert( nLastMaskedPixels <= nMaskedPixels );
                assert( (unsigned) totalError % (int) pythagoreanTriple.c == 0 );
                assert( nMaskedPixels * pythagoreanTriple.c == totalError );
                assert( nMaskedPixels <= nElements );
            }

            /* check packed bit version */
            cudaCalculateHioErrorBitPacked(
                CudaKernelConfig(1,32),
                dpData, dpBitMasked, nElements, false /* don't invert mask */,
                &totalError, &nMaskedPixels
            );
            /* Calculation done, now check if everything is correct */
            if ( totalError < 16777216 ) // float vlaues higher round to multiple of 2
            {
                if ( ( not ( (unsigned) totalError % (int) pythagoreanTriple.c == 0 ) ) ||
                     ( not ( nLastMaskedPixels <= nMaskedPixels ) ) ||
                     ( not ( nMaskedPixels * pythagoreanTriple.c == totalError ) )
                   )
                {
                    std::cout << "nElements        : " << nElements         << std::endl
                              << "nLastMaskedPixels: " << nLastMaskedPixels << std::endl
                              << "nMaskedPixels    : " << nMaskedPixels     << std::endl
                              << "totalError       : " << totalError        << std::endl;
                }
                assert( nMaskedPixels <= nElements );
                assert( (unsigned) totalError % (int) pythagoreanTriple.c == 0 );
                assert( nLastMaskedPixels <= nMaskedPixels );
                assert( nMaskedPixels * pythagoreanTriple.c == totalError );
            }
            else
            {
                /* no use continuing this loop if we can't assert anything */
                break;
            }

            #ifdef USE_FFTW
                static_assert( sizeof( cufftComplex ) == sizeof( fftwf_complex ), "" );

                /* now compare with CPU version which should give the exact same
                 * result, as there should be no floating point rounding errors
                 * for relatively short array ( < 1e6 ? ) */
                float nMaskedPixelsCpu, totalErrorCpu;
                calculateHioError( (fftwf_complex*) pData, pIsMasked, nElements, /* is inverted:  */ false, &totalErrorCpu, &nMaskedPixelsCpu );

                /* when rounding errors occur the order becomes important */
                if ( totalError < 16777216 )
                {
                    assert( compareFloat( __FILE__, __LINE__, totalError, totalErrorCpu, sqrtf(nElements) ) );
                    assert( nMaskedPixelsCpu == nMaskedPixels );
                    assert( nMaskedPixelsCpu <= nElements );
                }
            #endif
        }
        std::cout << "OK\n";

        /* benchmark with random numbers */

        for ( auto i = 0u; i < nBitMaskedElements; ++i )
        {
            pData[i].x = (float) rand() / RAND_MAX;
            pData[i].y = (float) rand() / RAND_MAX;
        }
        CUDA_ERROR( cudaMemcpy( dpData, pData, nMaxElements * sizeof( pData[0] ), cudaMemcpyHostToDevice ) );

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        using clock = std::chrono::high_resolution_clock;

        std::cout << "time in milliseconds calcHioError which sums up the norm of masked complex values:\n";
        /*           "  524287 :  3.41546 | 2.45486 | 11.3163 |0.974992 */
        std::cout << "  vector : mask in  | mask in | mask bit| CPU not |\n"
                  << "  length : uint32_t | uint8_t | packed  | alpaka  |\n"
                  << "---------:----------+---------+--------+----------+"
                  << std::endl;
        for ( auto nElements : getLogSpacedSamplingPoints( 2, nMaxElements, 50 ) )
        {
            std::cout << std::setw(8) << nElements << " : ";
            float milliseconds, minTime;
            decltype( clock::now() ) clock0, clock1;

            float error;
            #define TIME_GPU( FUNC, MASK, CONFIG )                            \
            minTime = FLT_MAX;                                                \
            for ( auto iRepetition = 0u; iRepetition < nRepetitions;          \
                  ++iRepetition )                                             \
            {                                                                 \
                cudaEventRecord( start );                                     \
                error = FUNC( CONFIG, dpData, MASK, nElements );              \
                cudaEventRecord( stop );                                      \
                cudaEventSynchronize( stop );                                 \
                cudaEventElapsedTime( &milliseconds, start, stop );           \
                minTime = fmin( minTime, milliseconds );                      \
                assert( error <= nElements );                                 \
            }                                                                 \
            std::cout << std::setw(8) << minTime << " |" << std::flush;

            TIME_GPU( cudaCalculateHioError, dpIsMasked, CudaKernelConfig() )
            auto unpackedError = error;
            TIME_GPU( cudaCalculateHioError, dpIsMaskedChar, CudaKernelConfig() ) // sets error
            compareFloat( __FILE__, __LINE__, unpackedError, error, sqrtf(nElements) );
            if ( nElements < 1e6 )
            {
                TIME_GPU( cudaCalculateHioErrorBitPacked, dpBitMasked, CudaKernelConfig( 0,32 ) ) // sets error
                compareFloat( __FILE__, __LINE__, unpackedError, error, sqrtf(nElements) );
            }
            else
                std::cout << std::setw(8) << -1 << " |" << std::flush;

            #ifdef USE_FFTW
                /* time CPU */
                minTime = FLT_MAX;
                for ( auto iRepetition = 0u; iRepetition < nRepetitions;
                      ++iRepetition )
                {
                    clock0 = clock::now();
                    error = calculateHioError( (fftwf_complex*) pData, pIsMasked, nElements );
                    clock1 = clock::now();
                    auto seconds = duration_cast<duration<float>>( clock1 - clock0 );
                    minTime = std::fmin( minTime, seconds.count() * 1000 );
                    assert( error <= nElements );
                }
            #endif
            std::cout << std::setw(8) << minTime << "\n" << std::flush;
        }

        /* free */
        CUDA_ERROR( cudaFree( dpData          ) );
        CUDA_ERROR( cudaFree( dpIsMasked      ) );
        CUDA_ERROR( cudaFree( dpBitMasked     ) );
        delete[] pData;
        delete[] pIsMasked;
        delete[] pBitMasked;
    }


} // namespace algorithms
} // namespace imresh