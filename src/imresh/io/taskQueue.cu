/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2016 Philipp Trommler
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <functional>               // std::function
#ifdef IMRESH_DEBUG
#   include <iostream>              // std::cout, std::endl
#endif
#include <list>                     // std::list
#include <mutex>                    // std::mutex
#include <thread>                   // std::thread
#include <utility>                  // std::pair

#include "algorithms/cuda/cudaShrinkWrap.h"
#include "libs/cudacommon.h"        // CUDA_ERROR

namespace imresh
{
namespace io
{
    /**
     * Struct containing a CUDA stream with it's associated device.
     */
    struct stream
    {
        int device;
        cudaStream_t str;
    };

    /**
     * Mutex to coordinate device usage.
     */
    std::mutex mtx;
    /**
     * List where all streams are stored as imresh::io::stream structs.
     */
    std::list<stream> streamList;
    /**
     * List to store all created threads.
     */
    std::list<std::thread> threadPool;
    /**
     * Maximum size of the thread pool.
     *
     * This is determined while imresh::io::fillStreamList() as the number of
     * available streams.
     */
    unsigned int threadPoolMaxSize = 0;

    /**
     * Function to add a image processing task to the queue.
     *
     * This is called from taskQueue::addTask() as a thread to prevent blocking
     * and to ensure that all streams are filled with work. It selects the least
     * recently used stream from the streamList and fills it with new work (FIFO).
     *
     * A mutex ensures the correct work balancing over the CUDA streams.
     * However, this mutex doesn't include the call to the write out function.
     * If you need your write out function to be thread safe, you'll have to
     * use your own lock mechanisms inside of this function.
     *
     * @param _h_mem Pointer to the image data.
     * @param _size Size of the memory to be adressed.
     * @param _writeOutFunc A function pointer (std::function) that will be
     * used to handle the processed data.
     * @param _filename The filename to use to save the processed image. Note
     * that some write out functions will take a file extension (as '.png') and
     * some others may not.
     * @param _numberOfCycles Number of iterations to run shrink wrap for.
     * @param _numberOfHIOCycles Number of iterations to run the initial
     * hybrid input output for.
     * @param _targetError The target error to stop the program when reached.
     */
    void addTaskAsync(
        float* _h_mem,
        std::pair<unsigned int,unsigned int> _size,
        std::function<void(float*,std::pair<unsigned int,unsigned int>,
            std::string)> _writeOutFunc,
        std::string _filename,
        unsigned int _numberOfCycles,
        unsigned int _numberOfHIOCycles,
        float _targetError,
        float _HIOBeta,
        float _intensityCutOffAutoCorel,
        float _intensityCutOff,
        float _sigma0,
        float _sigmaChange
    )
    {
        // Lock the mutex so no other thread intermediatly changes the device
        // selection
        mtx.lock( );
        // Get the next device and stream to use
        auto strm = streamList.front( );
        streamList.pop_front( );
        streamList.push_back( strm );
        auto device = strm.device;
        auto str = strm.str;

        // Select device and copy memory
        CUDA_ERROR( cudaSetDevice( device ) );

#       ifdef IMRESH_DEBUG
            std::cout << "imresh::io::addTaskAsync(): Mutex locked, device and stream selected. Calling shrink-wrap."
                << std::endl;
#       endif

        // Call shrinkWrap in the selected stream on the selected device.
        imresh::algorithms::cuda::cudaShrinkWrap( _h_mem,
                                              _size.first,
                                              _size.second,
                                              str,
                                              device,
                                              _numberOfCycles,
                                              _targetError,
                                              _HIOBeta,
                                              _intensityCutOffAutoCorel,
                                              _intensityCutOff,
                                              _sigma0,
                                              _sigmaChange,
                                              _numberOfHIOCycles );

        mtx.unlock( );

#       ifdef IMRESH_DEBUG
            std::cout << "imresh::io::addTaskAsync(): CUDA work finished, mutex unlocked. Calling write out function."
                << std::endl;
#       endif

        _writeOutFunc( _h_mem, _size, _filename );
    }

    /**
     * Calls imresh::io::addTaskAsync() in a thread.
     */
    void addTask(
        float* _h_mem,
        std::pair<unsigned int,unsigned int> _size,
        std::function<void(float*,std::pair<unsigned int,unsigned int>,
            std::string)> _writeOutFunc,
        std::string _filename,
        unsigned int _numberOfCycles = 20,
        unsigned int _numberOfHIOCycles = 20,
        float _targetError = 0.00001f,
        float _HIOBeta = 0.9f,
        float _intensityCutOffAutoCorel = 0.04f,
        float _intensityCutOff = 0.2f,
        float _sigma0 = 3.0f,
        float _sigmaChange = 0.01f
    )
    {
        while( threadPool.size( ) >= threadPoolMaxSize )
        {
#           ifdef IMRESH_DEBUG
                std::cout << "imresh::io::addTask(): Too many active threads. Waiting for one of them to finish."
                    << std::endl;
#           endif
            if ( threadPool.front().joinable() )
                threadPool.front( ).join( );
            else
            {
#               ifdef IMRESH_DEBUG
                    std::cout << "[Warning] " << __FILE__ << " line " << __LINE__
                              << ": a thread from the thread pool is not joinable!\n";
#               endif
            }
            threadPool.pop_front( );
        }

#       ifdef IMRESH_DEBUG
            std::cout << "imresh::io::addTask(): Creating working thread."
                << std::endl;
#       endif

        threadPool.push_back( std::thread( addTaskAsync, _h_mem,
                                                         _size,
                                                         _writeOutFunc,
                                                         _filename,
                                                         _numberOfCycles,
                                                         _numberOfHIOCycles,
                                                         _targetError,
                                                         _HIOBeta,
                                                         _intensityCutOffAutoCorel,
                                                         _intensityCutOff,
                                                         _sigma0,
                                                         _sigmaChange ) );
    }

    /**
     * This function adds all streams to the stream list.
     *
     * To achieve that it iterates over all available devices and creates one
     * stream for each multiprocessor on each device. All these streams are
     * stored in the streamList as imresh::io::stream objects. If no streams are
     * found, the program aborts.
     */
    unsigned fillStreamList( )
    {
#       ifdef IMRESH_DEBUG
            std::cout << "imresh::io::fillStreamList(): Starting stream creation."
                << std::endl;
#       endif
        int deviceCount = 0;
        CUDA_ERROR( cudaGetDeviceCount( &deviceCount ) );

        if( deviceCount <= 0 )
        {
#           ifdef IMRESH_DEBUG
                std::cout << "imresh::io::fillStreamList(): No devices found. Aborting."
                    << std::endl;
#           endif
            exit( EXIT_FAILURE );
        }

        for( int i = 0; i < deviceCount; i++ )
        {
            cudaDeviceProp prop;
            CUDA_ERROR( cudaGetDeviceProperties( &prop, i ) );

            if( prop.multiProcessorCount <= 0 )
            {
#               ifdef IMRESH_DEBUG
                    std::cout << "imresh::io::fillStreamList(): Devices has no multiprocessors. Aborting."
                        << std::endl;
#               endif
                exit( EXIT_FAILURE );
            }

            for( int j = 0; j < prop.multiProcessorCount; j++ )
            {
                stream str;
                str.device = i;
                CUDA_ERROR( cudaStreamCreate( &str.str ) );
                streamList.push_back( str );
#               ifdef IMRESH_DEBUG
                    std::cout << "imresh::io::fillStreamList(): Created stream "
                        << j << " on device " << i << std::endl;
#               endif
            }
        }
#       ifdef IMRESH_DEBUG
            std::cout << "imresh::io::fillStreamList(): Finished stream creation."
                << std::endl;
#       endif

        return streamList.size( );
    }

    /**
     * Initializes the library.
     *
     * This is the _first_ call you should make in order to use this library.
     */
    void taskQueueInit( )
    {
        threadPoolMaxSize = fillStreamList( );
#       ifdef IMRESH_DEBUG
            std::cout << "imresh::io::taskQueueInit(): Finished initilization."
                << std::endl;
#       endif
    }

    /**
     * Deinitializes the library.
     *
     * This is the last call you should make in order to clear the library's
     * members.
     */
    void taskQueueDeinit( )
    {
        threadPoolMaxSize = 0;

        while( threadPool.size( ) > 0 )
        {
            threadPool.front( ).join( );
            threadPool.pop_front( );
        }

        while( streamList.size( ) > 0 )
        {
            CUDA_ERROR( cudaStreamDestroy( streamList.front( ).str ) );
            streamList.pop_front( );
        }

#       ifdef IMRESH_DEBUG
            std::cout << "imresh::io::taskQueueDeinit(): Finished deinitilization."
                << std::endl;
#       endif
    }
} // namespace io
} // namespace imresh
