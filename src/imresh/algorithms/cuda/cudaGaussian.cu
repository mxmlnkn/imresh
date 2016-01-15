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


#include "algorithms/cuda/cudaGaussian.h"
#include "libs/cudacommon.h"


/* doesn't work inside namespaces */
constexpr unsigned nMaxWeights = 50; // need to assert whether this is enough
constexpr unsigned nMaxKernels = 20; // ibid
__constant__ float gdpGaussianWeights[ nMaxWeights*nMaxKernels ];


namespace imresh
{
namespace algorithms
{
namespace cuda
{

    #define DEBUG_CUDAGAUSSIAN_CPP 0


    template<class T>
    __device__ inline T * ptrMin ( T * const a, T * const b )
    {
        return a < b ? a : b;
    }
    /**
     * Provides a class for a moving window type 2d cache
     **/
    template<class T_PREC>
    struct Cache1d
    {
        T_PREC const * const & data;
        unsigned const & nData;

        T_PREC * const & smBuffer; /**< pointer to allocated buffer, will not be allocated on constructor because this class needs to be trivial to work on GPU */
        unsigned const & nBuffer;

        unsigned const & nThreads;
        unsigned const & nKernelHalf;

        __device__ inline T_PREC & operator[]( unsigned i ) const
        {
            return smBuffer[i];
        }

        #ifndef NDEBUG
        #if DEBUG_CUDAGAUSSIAN_CPP == 1
            __device__ void printCache( void ) const
            {
                if ( threadIdx.x != 0 or blockIdx.x != 0 )
                    return;
                for ( unsigned i = 0; i < nBuffer; ++i )
                {
                    printf( "% 3i :", i );
                    printf( "%11.6f\n", smBuffer[i] );
                }
            }
        #endif
        #endif

        __device__ inline void initializeCache( void ) const
        {
            #ifndef NDEBUG
            #if DEBUG_CUDAGAUSSIAN_CPP == 1
                /* makes it easier to see if we cache the correct data */
                if ( threadIdx.x == 0 )
                    memset( smBuffer, 0, nBuffer*sizeof( smBuffer[0] ) );
                __syncthreads();
            #endif
            #endif

            /* In the first step initialize the left border to the same values (extend)
             * It's problematic to use threads for this for loop, because it is not
             * guaranteed, that blockDim.x >= N */
            /**
             * benchmark ImageSize 1024x1024
             *    parallel    : 1.55ms
             *    non-parallel: 1.59ms
             * the used register count is equal for both versions.
             **/
            #if false
                for ( T_PREC * target = smBuffer + nThreads + threadIdx.x;
                      target < smBuffer + nBuffer; target += nThreads )
                {
                    *target = leftBorderValue;
                }
            #else
                if ( threadIdx.x == 0 )
                for ( unsigned iB = nThreads; iB < nBuffer; ++iB )
                {
                    const int signedDataIndex = int(iB-nThreads) - (int)nKernelHalf;
                    /* periodic */
                    //unsigned cappedIndex = signedDataIndex % signedDataIndex;
                    //if ( cappedIndex < 0 ) cappedIndex += rImageWidth;
                    /* extend */
                    const unsigned cappedIndex = min( nData-1, (unsigned) max( 0, signedDataIndex ) );
                    smBuffer[iB] = data[cappedIndex];
                }
            #endif

            #ifndef NDEBUG
            #if DEBUG_CUDAGAUSSIAN_CPP == 1
                if ( threadIdx.x == 0 and blockIdx.x == 0 )
                {
                    printf("Copy some initial data to buffer:\n");
                    printCache();
                }
            #endif
            #endif
        }

        __device__ inline void loadCacheLine( T_PREC const * const curDataRow ) const
        {
            /* move last N elements to the front of the buffer */
            __syncthreads();
            /**
             * all of these do the same, but benchmarks suggest that the
             * last version which looks the most complicated is the fastest:
             * imageSize 1024x1024, compiled with -O0:
             *   - memcpy            : 1.09ms
             *   - parallel          : 0.78ms
             *   - pointer arithmetic: 0.89ms
             * not that the memcpy version only seems to work on the GPU, on
             * the CPU it lead to wrong results I guess because we partially
             * write to memory we also read, or it was some kind of other
             * error ...
             **/
            #if true
                /* eliminating the variable i doesn't make it faster ... */
                for ( unsigned i = threadIdx.x, i0 = 0;
                      i0 + nThreads < nBuffer;
                      i += nThreads, i0 += nThreads )
                {
                    if ( i+nThreads < nBuffer )
                        smBuffer[i] = smBuffer[i+nThreads];
                    __syncthreads();
                }
            #elif false
                if ( threadIdx.x == 0 )
                    memcpy( smBuffer, &smBuffer[ nThreads ], nKernelHalf*sizeof(T_PREC) );
            #else
                /* this version may actually be wrong, because some threads
                 * could be already in next iteration, thereby overwriting
                 * some values which still are to be moved! */
                /*{
                    T_PREC * iTarget = smBuffer + threadIdx.x;
                    T_PREC const * iSrc = iTarget + nThreads;
                    for ( ; iSrc < smBuffer + nBufferSize;
                          iTarget += nThreads, iSrc += nThreads )
                    {
                        *iTarget = *iSrc;
                    }
                }*/
            #endif

            #ifndef NDEBUG
            #if DEBUG_CUDAGAUSSIAN_CPP == 1
                if ( threadIdx.x == 0 and blockIdx.x == 0 )
                {
                    printf( "Shift buffer by %i elements:\n", nThreads );
                    printCache();
                }
            #endif
            #endif

            /* Load nThreads new data elements into buffer. */
            //for ( unsigned iRowBuf = nRowsBuffer - nThreads; iRowBuf < nRowsBuffer; ++iRowBuf )
            //const unsigned newRow = min( rnDataY-1, (unsigned) max( 0,
            //    (int)iRow - (int)nKernelHalf + (int)iRowBuf ) );
            //const unsigned iBuf = nKernelHalf + threadIdx.x;
            const unsigned iBuf = nBuffer - nThreads + threadIdx.x;
            /* If data end reached, fill buffer with last data element */
            assert( curDataRow - nKernelHalf + iBuf == curDataRow + nKernelHalf + threadIdx.x );
            T_PREC const * const datum = ptrMin( data + nData-1,
                curDataRow - nKernelHalf + iBuf );
            assert( iBuf < nBuffer );
            __syncthreads();
            smBuffer[ iBuf ] = *datum;
            __syncthreads();

            #ifndef NDEBUG
            #if DEBUG_CUDAGAUSSIAN_CPP == 1
                if ( threadIdx.x == 0 and blockIdx.x == 0 )
                {
                    printf( "Load %i new elements:\n", nThreads );
                    printCache();
                }
            #endif
            #endif
        }
    };

    /**
     * Choose the buffer size, so that in every step rnThreads data values
     * can be saved back and newly loaded. As we need N neighbors left and
     * right for the calculation of one value, especially at the borders,
     * this means, the buffer size needs to be rnThreads + 2*N elements long:
     * @verbatim
     *                                                   kernel
     * +--+--+--+--+--+--+--+--+--+--+--+--+        +--+--+--+--+--+
     * |xx|xx|  |  |  |  |  |  |  |  |yy|yy|        |  |  |  |  |  |
     * +--+--+--+--+--+--+--+--+--+--+--+--+        +--+--+--+--+--+
     * <-----><---------------------><----->        <-------------->
     *   N=2       rnThreads = 8      N=2             rnWeights = 5
     *                                              <----->  <----->
     *                                                N=2      N=2
     * @endverbatim
     * Elements marked with xx and yy can't be calculated, the other elements
     * can be calculated in parallel.
     *
     * In the first step the elements marked with xx are copie filled with
     * the value in the element right beside it, i.e. extended borders.
     *
     * In the step thereafter especially the elements marked yy need to be
     * calculated (if the are not already on the border). To calculate those
     * we need to move yy and N=2 elements to the left to the beginning of
     * the buffer and fill the rest with new data from rData:
     * @verbatim
     *               ((bufferSize-1)-(2*N-1)
     *                           |
     * <------------ bufferSize -v--------->
     * +--+--+--+--+--+--+--+--+--+--+--+--+
     * |xx|xx|  |  |  |  |  |  |  |  |yy|yy|
     * +--+--+--+--+--+--+--+--+--+--+--+--+
     *                         <----------->
     * <----------->                2*N=4
     *       ^                        |
     *       |________________________|
     *
     * +--+--+--+--+--+--+--+--+--+--+--+--+
     * |vv|vv|yy|yy|  |  |  |  |  |  |ww|ww|
     * +--+--+--+--+--+--+--+--+--+--+--+--+
     * <-----><---------------------><----->
     *   N=2       rnThreads = 8      N=2
     * @endverbatim
     * All elements except those marked vv and ww can now be calculated
     * in parallel. The elements marked yy are the old elements from the right
     * border, which were only used readingly up till now. The move of the
     * 2*N elements may be preventable by using a modulo address access, but
     * a move in shared memory / cache is much faster than waiting for the
     * rest of the array to be filled with new data from global i.e. uncached
     * memory.
     *
     * @param[in] blockDim.x number of threads will be interpreted as how many
     *            values are to be calculated in parallel. The internal buffer
     *            stores then blockDim.x + 2*N values per step
     * @param[in] blockDim.y number of rows to blur. threadIdx.y == 0 will blur
     *            rdpData[ 0...rImageWidth-1 ], threadIdx.y == 1 the next
     *            rImageWidth elements. Beware that you must not start more
     *            threads in y direction than the image has rows, else a
     *            segfault will occur!
     * @param[in] N kernel half size, meaning the kernel is supposed to be
     *            2*N+1 elements long. N can also be interpreted as the number
     *            of neighbors in each direction needed to calculate one value.
     **/
    template<class T_PREC>
    __global__ void cudaKernelApplyKernelSharedWeights
    (
        /* You can't pass by reference to a kernel !!! compiles, but gives weird errors ... */
        T_PREC * const rdpData,
        const unsigned rImageWidth,
        T_PREC const * const rdpWeights,
        const unsigned nKernelHalf
    )
    {
        assert( blockDim.y == 1 and blockDim.z == 1 );
        assert(  gridDim.y == 1 and  gridDim.z == 1 );

        /* If more than 1 block, then each block works on a separate line.
         * Each line borders will be extended. So mutliple blocks can't be used
         * to blur one very very long line even faster! */
        const int & nThreads = blockDim.x;
        T_PREC * const data = &rdpData[ blockIdx.x * rImageWidth ];

        /* manage dynamically allocated shared memory */
        /* @see http://stackoverflow.com/questions/27570552/ */
        extern __shared__ __align__( sizeof(T_PREC) ) unsigned char dynamicSharedMemory[];
        T_PREC * const smBlock = reinterpret_cast<T_PREC*>( dynamicSharedMemory );

        const unsigned nWeights = 2*nKernelHalf+1;
        const unsigned nBufferSize = nThreads + 2*nKernelHalf;
        T_PREC * const smWeights = smBlock;
        T_PREC * const smBuffer  = &smBlock[ nWeights ];
        __syncthreads();

        Cache1d<T_PREC> buffer{ data, rImageWidth, smBuffer, nBufferSize, blockDim.x, nKernelHalf };

        /* cache the weights to shared memory. Benchmarks imageSize 1024x1024
         * parallel (pointer arithmetic) : 0.95ms
         * parallel                      : 1.1ms
         * memcpy                        : 1.57ms
        */
        #if true
            {
                T_PREC * target    = smWeights  + threadIdx.x;
                T_PREC const * src = rdpWeights + threadIdx.x;
                for ( ; target < smWeights + nWeights;
                     target += blockDim.x, src += blockDim.x )
                {
                    *target = *src;
                }
            }
        #elif false
            for ( unsigned iWeight = threadIdx.x; iWeight < nWeights; ++iWeight )
                smWeights[iWeight] = rdpWeights[iWeight];
        #else
            if ( threadIdx.x == 0 )
                memcpy( smWeights, rdpWeights, sizeof(T_PREC)*nWeights );
        #endif


        #ifndef NDEBUG
        #if DEBUG_CUDAGAUSSIAN_CPP == 1
            if ( blockIdx.x == 0 and threadIdx.x == 0 )
            {
                printf( "================ cudaGaussianApplyKernel ================\n" );
                printf( "\gridDim = (%i,%i,%i), blockDim = (%i,%i,%i)\n",
                        gridDim.x, gridDim.y, gridDim.z,
                        blockDim.x, blockDim.y, blockDim.z );
                printf( "rImageWidth = %u\n", rImageWidth );
                printf( "\nConvolve Kernel : \n");
                for ( unsigned iW = 0; iW < nWeights; ++iW )
                    printf( "%10.6f ", smWeights[iW] );
                printf( "\n" );

                printf( "\nInput Vector to convolve horizontally : \n" );
                for ( unsigned i = 0; i < rImageWidth; ++i )
                    printf( "%10.6f ", data[ i ] );
                printf( "\n" );
            }
        #endif
        #endif

        buffer.initializeCache();

        /* Loop over buffers. If rnData == rnThreads then the buffer will
         * exactly suffice, meaning the loop will only be run 1 time.
         * The for loop break condition is the same for all threads, so it is
         * safe to use __syncthreads() inside */
        for ( T_PREC * curDataRow = data; curDataRow < data + rImageWidth; curDataRow += nThreads )
        {
            buffer.loadCacheLine( curDataRow );
            #ifndef NDEBUG
            #if DEBUG_CUDAGAUSSIAN_CPP == 1
                if ( rImageWidth == 2 )
                    return; //assert(false);
            #endif
            #endif

            /* calculated weighted sum on inner points in buffer, but only if
             * the value we are at is actually needed: */
            const unsigned iBuf = nKernelHalf + threadIdx.x;
            if ( &curDataRow[iBuf-nKernelHalf] < &data[rImageWidth] )
            {
                T_PREC sum = T_PREC(0);
                /* this for loop is done by each thread and should for large
                 * enough kernel sizes sufficiently utilize raw computing power */
                for ( T_PREC * w = smWeights, * x = &buffer[iBuf-nKernelHalf];
                      w < &smWeights[nWeights]; ++w, ++x )
                {
                    sum += (*w) * (*x);
                }
                /* write result back into memory (in-place). No need to wait for
                 * all threads to finish, because we write into global memory,
                 * to values we already buffered into shared memory! */
                curDataRow[iBuf-nKernelHalf] = sum;
            }
        }
    }


    /* new version using constant memory */

    template<class T_PREC>
    __global__ void cudaKernelApplyKernel
    (
        /* You can't pass by reference to a kernel !!! compiles, but gives weird errors ... */
        T_PREC * const rdpData,
        unsigned const rnDataX,
        unsigned const rnDataY,
        T_PREC * const rWeights,
        unsigned const rnWeights
    )
    {
        assert( blockDim.y == 1 and blockDim.z == 1 );
        assert(  gridDim.y == 1 and  gridDim.z == 1 );
        assert( rnWeights >= 1 );
        const unsigned nKernelHalf = (rnWeights-1) / 2;

        const int & nThreads = blockDim.x;
        T_PREC * const data = &rdpData[ blockIdx.x * rnDataX ];

        /* @see http://stackoverflow.com/questions/27570552/ */
        extern __shared__ __align__( sizeof(T_PREC) ) unsigned char sm[];
        T_PREC * const smBuffer = reinterpret_cast<T_PREC*>( sm );
        const unsigned nBufferSize = nThreads + 2*nKernelHalf;
        __syncthreads();

        Cache1d<T_PREC> buffer{ data, rnDataX, smBuffer, nBufferSize, blockDim.x, nKernelHalf };
        buffer.initializeCache(); /* loads first set of data */

        /* The for loop break condition is the same for all threads in a block,
         * so it is safe to use __syncthreads() inside */
        for ( T_PREC * curDataRow = data; curDataRow < data + rnDataX; curDataRow += nThreads )
        {
            buffer.loadCacheLine( curDataRow );

            /* calculated weighted sum on inner points in buffer, but only if
             * the value we are at is actually needed: */
            const unsigned iBuf = nKernelHalf + threadIdx.x;
            if ( &curDataRow[iBuf-nKernelHalf] < &data[rnDataX] )
            {
                T_PREC sum = T_PREC(0);
                for ( T_PREC * w = rWeights, * x = &buffer[iBuf-nKernelHalf];
                      w < rWeights + rnWeights; ++w, ++x )
                {
                    sum += (*w) * (*x);
                }
                /* write result back into memory (in-place). No need to wait for
                 * all threads to finish, because we write into global memory,
                 * to values we already buffered into shared memory! */
                curDataRow[iBuf-nKernelHalf] = sum;
            }
        }
    }



    template<class T_PREC>
    void cudaGaussianBlurHorizontal
    (
        T_PREC * const & rdpData,
        const unsigned & rnDataX,
        const unsigned & rnDataY,
        const double & rSigma
    )
    {
        static unsigned firstFree = 0;
        static float kernelSigmas[ nMaxKernels ];
        static float kernelSizes [ nMaxKernels ];

        /* look if we already have that kernel buffered */
        unsigned iKernel = 0;
        for ( ; iKernel < firstFree; ++iKernel )
            if ( kernelSigmas[ iKernel ] == rSigma )
                break;

        T_PREC * dpWeights = NULL;
        unsigned kernelSize = 0;
        /* if not found, then calculate and save it */
        if ( iKernel == firstFree )
        {
            /* calc kernel */
            T_PREC pKernel[nMaxWeights];
            kernelSize = libs::calcGaussianKernel( rSigma, (T_PREC*) pKernel, nMaxWeights );
            assert( kernelSize <= nMaxWeights );

            /* if buffer full, then delete buffer */
            if ( firstFree == nMaxKernels )
            {
                #ifndef NDEBUG
                    std::cout << "Warning, couldn't find sigma in kernel buffer and no space to store it. Clearing buffer!\n";
                #endif
                firstFree = 0;
                iKernel = 0;
            }

            /* remember sigma */
            kernelSigmas[ iKernel ] = rSigma;
            kernelSizes [ iKernel ] = kernelSize;
            ++firstFree;

            /* upload to GPU */
            CUDA_ERROR( cudaGetSymbolAddress( (void**) &dpWeights, gdpGaussianWeights ) );
            dpWeights += iKernel * nMaxWeights;
            CUDA_ERROR( cudaMemcpy( dpWeights, pKernel,
                kernelSize * sizeof( pKernel[0] ), cudaMemcpyHostToDevice ) );
        }
        else
        {
            CUDA_ERROR( cudaGetSymbolAddress( (void**) &dpWeights, gdpGaussianWeights ) );
            dpWeights += iKernel * nMaxWeights;
            kernelSize = kernelSizes[ iKernel ];
        }

        /* the image must be at least nThreads threads wide, else many threads
         * will only sleep. The number of blocks is equal to the image height.
         * Every block works on 1 image line. The number of Threads is limited
         * by the hardware to be e.g. 512 or 1024. The reason for this is the
         * limited shared memory size! */
        const unsigned nThreads = 16;
        const unsigned nBlocks  = rnDataY;
        const unsigned bufferSize = nThreads + kernelSize-1;

        cudaKernelApplyKernel<<<
            nBlocks,nThreads,
            sizeof(T_PREC) * bufferSize
        >>>( rdpData, rnDataX, rnDataY, dpWeights, kernelSize );
        CUDA_ERROR( cudaDeviceSynchronize() );
    }


    template<class T_PREC>
    void cudaGaussianBlurHorizontalSharedWeights
    (
        T_PREC * const & rdpData,
        const unsigned & rnDataX,
        const unsigned & rnDataY,
        const double & rSigma
    )
    {
        const int nKernelElements = 64;
        T_PREC pKernel[64];
        const int kernelSize = libs::calcGaussianKernel( rSigma, (T_PREC*) pKernel, nKernelElements );
        assert( kernelSize <= nKernelElements );

        /* upload kernel to GPU */
        T_PREC * dpKernel;
        CUDA_ERROR( cudaMalloc( &dpKernel, sizeof(T_PREC)*kernelSize ) );
        CUDA_ERROR( cudaMemcpy(  dpKernel, pKernel, sizeof(T_PREC)*kernelSize, cudaMemcpyHostToDevice ) );

        /* the image must be at least nThreads threads wide, else many threads
         * will only sleep. The number of blocks is equal to the image height.
         * Every block works on 1 image line. The number of Threads is limited
         * by the hardware to be e.g. 512 or 1024. The reason for this is the
         * limited shared memory size! */
        const unsigned nThreads = 16;
        const unsigned nBlocks  = rnDataY;
        const unsigned N = (kernelSize-1)/2;
        const unsigned bufferSize = nThreads + 2*N;

        cudaKernelApplyKernelSharedWeights<<<
            nBlocks,nThreads,
            sizeof(T_PREC)*( kernelSize + bufferSize )
        >>>( rdpData, rnDataX, dpKernel, N );
        CUDA_ERROR( cudaDeviceSynchronize() );

        CUDA_ERROR( cudaFree( dpKernel ) );
    }


    /**
     * Calculates the weighted sum vertically i.e. over the rows.
     *
     * In order to make use of Cache Lines blockDim.x columns are always
     * calculated in parallel. Furthermore to increase parallelism blockIdx.y
     * threads can calculate the values for 1 column in parallel:
     * @verbatim
     *                gridDim.x=3
     *               <---------->
     *               blockDim.x=4
     *                    <-->
     *            I  #### #### ## ^
     *            m  #### #### ## | blockDim.y
     *            a  #### #### ## v    = 3
     *            g  #### #### ## ^
     *            e  #### #### ## | blockDim.y
     *                            v    = 3
     *               <---------->
     *               imageWidth=10
     * @endverbatim
     * The blockIdx.y threads act as a sliding window. Meaning in the above
     * example y-thread 0 and 1 need to calculate 2 values per kernel run,
     * y-thread 2 only needs to calculate 1 calue, because the image height
     * is not a multiplie of blockIdx.y
     *
     * Every block uses a shared memory buffer which holds roughly
     * blockDim.x*blockDim.y elements. In order to work on wider images the
     * kernel can be called with blockDim.x != 0
     *
     * @see cudaKernelApplyKernel @see gaussianBlurVertical
     *
     * @param[in] N kernel half size. rdpWeights is assumed to be 2*N+1
     *            elements long.
     * @param[in] blockDim.x number of columns to calculate in parallel.
     *            this should be a value which makes full use of a cache line,
     *            i.e. 32 Warps * 4 Byte = 128 Byte for a NVidia GPU (2016)
     * @param[in] blockDim.y number of rows to calculate in parallel.
     *            This value shouldn't be too small, because else we are only
     *            moving the buffer date to and fro without doing much
     *            calculation. That happens because of the number of neighbors
     *            N needed to calculate 1 value. If the buffer is 2*N+1
     *            elements large ( blockDim.y == 1 ), then we can only
     *            calculate 1 value with that buffer data.
     *            @todo implement buffer index modulo instead of shifting the
     *                  values in memory
     **/
    template<class T_PREC>
    __global__ void cudaKernelApplyKernelVertically
    (
        T_PREC * const rdpData,
        const unsigned rnDataX,
        const unsigned rnDataY,
        T_PREC const * const rdpWeights,
        const unsigned N
    )
    {
        assert( blockDim.z == 1 );
        assert( gridDim.y == 1 and  gridDim.z == 1 );

        /* the shared memory buffer dimensions */
        const unsigned nColsCacheLine = blockDim.x;
        const unsigned nRowsCacheLine = blockDim.y + 2*N;

        /* Each block works on a separate group of columns */
        T_PREC * const data = &rdpData[ blockIdx.x * blockDim.x ];
        /* the rightmost block might not be full. In that case we need to mask
         * those threads working on the columns right of the image border */
        const bool iAmSleeping = blockIdx.x * blockDim.x + threadIdx.x >= rnDataX;

        /* The dynamically allocated shared memory buffer will fit the weights and
         * the values to calculate + the 2*N neighbors needed to calculate them */
        extern __shared__ __align__( sizeof(T_PREC) ) unsigned char dynamicSharedMemory[];
        T_PREC * const smBlock = reinterpret_cast<T_PREC*>( dynamicSharedMemory );

        const unsigned nWeights   = 2*N+1;
        const unsigned nBufferSize = nColsCacheLine * nRowsCacheLine;
        T_PREC * const smWeights  = smBlock;
        T_PREC * const smBuffer  = &smBlock[ nWeights ];
        __syncthreads();

        /* cache the weights to shared memory */
        if ( threadIdx.x == 0 )
            memcpy( smWeights, rdpWeights, sizeof(rdpWeights[0])*nWeights );

        /**
         * @verbatim
         *                        (shared memory)
         *                         kernel (size 3)
         *                      +------+-----+-----+
         *                      | w_-1 | w_0 | w_1 |
         *                      +------+-----+-----+
         *       (global memory)
         *       data to convolve
         *    +------+------+------+------+    (should be a multiple of
         *    | a_00 | a_01 | a_02 | a_02 |   cache line wide i.e. nRows)
         *    +------+------+------+------+        (shared memory)
         *    | a_10 | a_11 | a_12 | a_12 |         result buffer
         *    +------+------+------+------+        +------+------+
         *    | a_20 | a_21 | a_22 | a_22 |        | b_00 | b_01 |
         *    +------+------+------+------+        +------+------+
         *    | a_30 | a_31 | a_32 | a_32 |        | b_10 | b_11 |
         *    +------+------+------+------+        +------+------+
         *    | a_40 | a_41 | a_42 | a_42 |        | b_20 | b_21 |
         *    +------+------+------+------+        +------+------+
         *    | a_50 | a_51 | a_52 | a_52 |
         *    +------+------+------+------+
         *
         *        b_0x = w_-1*a_1x + w_0*a_2x + w_1*a_3x
         *        b_1x = w_-1*a_2x + w_0*a_3x + w_1*a_4x
         *        b_1x = w_-1*a_3x + w_0*a_4x + w_1*a_5x
         *        b_1x = w_-1*a_3x + w_0*a_4x + w_1*a_5x
         * @endverbatim
         * In order to reduce global memory accesses, we can reorder the
         * calculation of b_ij so that we can cache one row of a_ij and basically
         * broadcast ist to b_ij:
         *
         *  a) cache a_1x  ->  b_0x += w_-1*a_1x
         *  b) cache a_2x  ->  b_0x += w_0*a_2x, b_1x += w_-1*a_2x
         *  c) cache a_3x  ->  b_0x += w_1*a_3x, b_1x += w_0*a_3x, b_2x += w_-1*a_3x
         *  d) cache a_4x  ->                    b_1x += w_1*a_1x, b_2x += w_0*a_4x
         *  e) cache a_5x  ->                                      b_2x += w_1*a_5x
         *
         * The buffer size needs at least kernelSize rows. If it's equal to kernel
         * size rows, then in every step one row will be completed calculating,
         * meaning it can be written back.
         * This enables us to use a round-robin like calculation:
         *   - after step c we can write-back b_0x to a_3x, we don't need a_3x
         *     anymore after this step.
         *   - because the write-back needs time the next calculation in d should
         *     write to b_1x. This is the case
         *   - the last operation in d would then be an addition to b_3x == b_0x
         *     (use i % 3)
         **/

        /* In the first step extend upper border. Write them into the N elements
         * before the lower border-N, beacause in the first step in the loop
         * these elements will be moved to the upper border, see below. */
        T_PREC * const smTargetRow = &smBuffer[ nBufferSize - 2*N*nColsCacheLine ];
        if ( threadIdx.y == 0 and not iAmSleeping )
        {
            const T_PREC upperBorderValue = data[ threadIdx.x ];
            for ( unsigned iB = 0; iB < N*nColsCacheLine; iB += nColsCacheLine )
                smTargetRow[ iB+threadIdx.x ] = upperBorderValue;
        }


        /* Loop over and calculate the rows. If rnDataY == blockDim.y, then the
         * buffer will exactly suffice, meaning the loop will only be run 1 time */
        for ( T_PREC * curDataRow = data; curDataRow < &data[rnDataX*rnDataY];
              curDataRow += blockDim.y * rnDataX )
        {
            /* move last N rows to the front of the buffer */
            __syncthreads();
            /* @todo: memcpy doesnt respect iAmSleeping yet!
            assert( smTargetRow + N*nColsCacheLine < smBuffer[ bufferSize ] );
            if ( threadIdx.y == 0 and threadIdx.x == 0 )
                memcpy( smBuffer, smTargetRow, N*nColsCacheLine*sizeof(smBuffer[0]) );
            */
            /* memcpy version above parallelized. @todo: benchmark what is faster! */
            if ( threadIdx.y == 0 and not iAmSleeping )
            {
                for ( unsigned iB = 0; iB < N*nColsCacheLine; iB += nColsCacheLine )
                {
                    const unsigned iBuffer = iB + threadIdx.x;
                        assert( iBuffer < nBufferSize );
                    smBuffer[ iBuffer ] = smTargetRow[ iBuffer ];
                }
            }

            /* Load blockDim.y + N rows into buffer.
             * If data end reached, fill buffer rows with last row
             *   a) Load blockDim.y rows in parallel */
            T_PREC * const pLastData = &data[ (rnDataY-1)*rnDataX + threadIdx.x ];
            const unsigned iBuf = /*skip first N rows*/ N * nColsCacheLine
                                + threadIdx.y * nColsCacheLine + threadIdx.x;
            __syncthreads();
            if ( not iAmSleeping )
            {
                T_PREC * const datum = ptrMin(
                    &curDataRow[ threadIdx.y * rnDataX + threadIdx.x ],
                    pLastData
                );
                assert( iBuf < nBufferSize );
                smBuffer[iBuf] = *datum;
            }
            /*   b) Load N rows by master threads, because nThreads >= N is not
             *      guaranteed. */
            if ( not iAmSleeping and threadIdx.y == 0 )
            {
                for ( unsigned iBufRow = N+blockDim.y; iBufRow < nRowsCacheLine; ++iBufRow )
                {
                    T_PREC * const datum = ptrMin(
                        &curDataRow[ (iBufRow-N) * rnDataX + threadIdx.x ],
                        pLastData
                    );
                    const unsigned iBuffer = iBufRow*nColsCacheLine + threadIdx.x;
                        assert( iBuffer < nBufferSize );
                    smBuffer[ iBuffer ] = *datum;
                }
            }
            __syncthreads();

            /* calculated weighted sum on inner rows in buffer, but only if
             * the value we are at is inside the image */
            if ( ( not iAmSleeping )
                 and &curDataRow[ threadIdx.y*rnDataX ] < &rdpData[ rnDataX*rnDataY ] )
            {
                T_PREC sum = T_PREC(0);
                /* this for loop is done by each thread and should for large
                 * enough kernel sizes sufficiently utilize raw computing power */
                T_PREC * w = smWeights;
                T_PREC * x = &smBuffer[ threadIdx.y * nColsCacheLine + threadIdx.x ];
                for ( ; w < &smWeights[nWeights]; ++w, x += nColsCacheLine )
                {
                    assert( w < smWeights + nWeights );
                    assert( x < smBuffer + nBufferSize );
                    sum += (*w) * (*x);
                }
                /* write result back into memory (in-place). No need to wait for
                 * all threads to finish, because we write into global memory, to
                 * values we already buffered into shared memory! */
                curDataRow[ threadIdx.y * rnDataX + threadIdx.x ] = sum;
            }
        }

    }


    template<class T_PREC>
    void cudaGaussianBlurVertical
    (
        T_PREC * const & rdpData,
        const unsigned & rnDataX,
        const unsigned & rnDataY,
        const double & rSigma
    )
    {

        /* calculate Gaussian kernel */
        const int nKernelElements = 64;
        T_PREC pKernel[64];
        const int kernelSize = libs::calcGaussianKernel( rSigma, (T_PREC*) pKernel, nKernelElements );
        assert( kernelSize <= nKernelElements );
        assert( kernelSize % 2 == 1 );

        /* upload kernel to GPU */
        T_PREC * dpKernel;
        CUDA_ERROR( cudaMalloc( &dpKernel, sizeof(T_PREC)*kernelSize ) );
        CUDA_ERROR( cudaMemcpy(  dpKernel, pKernel, sizeof(T_PREC)*kernelSize, cudaMemcpyHostToDevice ) );

        /**
         * the image must be at least nThreadsX threads wide, else many threads
         * will only sleep. The number of blocks is ceil( image height / nThreadsX )
         * Every block works on nThreadsX image columns.
         * Those columns use nThreadsY threads to parallelize the calculation per
         * column.
         * The number of Threads is limited by the hardware to be e.g. 512 or 1024.
         * The reason for this is the limited shared memory size!
         * nThreadsX should be a multiple of a cache line / superword = 32 warps *
         * 1 float per warp = 128 Byte => nThreadsX = 32. For double 16 would also
         * suffice.
         */
        dim3 nThreads( 32, 256/32, 1 );
        dim3 nBlocks ( 1, 1, 1 );
        nBlocks.x = (unsigned) ceilf( (float) rnDataX / nThreads.x );
        const unsigned kernelHalfSize = (kernelSize-1)/2;
        const unsigned bufferSize     = nThreads.x*( nThreads.y + 2*kernelHalfSize );

        cudaKernelApplyKernelVertically<<<
            nBlocks,nThreads,
            sizeof( dpKernel[0] ) * ( kernelSize + bufferSize )
        >>>( rdpData, rnDataX, rnDataY, dpKernel, kernelHalfSize );
        CUDA_ERROR( cudaDeviceSynchronize() );

        CUDA_ERROR( cudaFree( dpKernel ) );
    }


    template<class T_PREC>
    void cudaGaussianBlur
    (
        T_PREC * const & rdpData,
        const unsigned & rnDataX,
        const unsigned & rnDataY,
        const double & rSigma
    )
    {
        cudaGaussianBlurHorizontal( rdpData,rnDataX,rnDataY,rSigma );
        cudaGaussianBlurVertical  ( rdpData,rnDataX,rnDataY,rSigma );
    }


    template<class T_PREC>
    void cudaGaussianBlurSharedWeights
    (
        T_PREC * const & rdpData,
        const unsigned & rnDataX,
        const unsigned & rnDataY,
        const double & rSigma
    )
    {
        cudaGaussianBlurHorizontalSharedWeights( rdpData,rnDataX,rnDataY,rSigma );
        cudaGaussianBlurVertical  ( rdpData,rnDataX,rnDataY,rSigma );
    }


    /* Explicitely instantiate certain template arguments to make an object file */

    template void cudaGaussianBlur<float>
    (
        float * const & rData,
        const unsigned & rnDataX,
        const unsigned & rnDataY,
        const double & rSigma
    );
    template void cudaGaussianBlur<double>
    (
        double * const & rData,
        const unsigned & rnDataX,
        const unsigned & rnDataY,
        const double & rSigma
    );


    template void cudaGaussianBlurSharedWeights<float>
    (
        float * const & rData,
        const unsigned & rnDataX,
        const unsigned & rnDataY,
        const double & rSigma
    );
    template void cudaGaussianBlurSharedWeights<double>
    (
        double * const & rData,
        const unsigned & rnDataX,
        const unsigned & rnDataY,
        const double & rSigma
    );

    /**
     * The following templates will be implicitely instantiated by the above
     * functions:
     *   - cudaApplyKernel
     *   - cudaGaussianBlurHorizontal
     *   - cudaGaussianBlurVertical
     **/


} // namespace cuda
} // namespace algorithms
} // namespace imresh
