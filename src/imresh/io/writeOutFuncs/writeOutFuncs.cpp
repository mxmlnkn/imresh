/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2016 Philipp Trommler, Maximilian Knespel
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


#include <algorithm>
#ifdef IMRESH_DEBUG
#   include <iostream>              // std::cout, std::endl
#endif
#ifdef USE_PNG
#   include <pngwriter.h>
#endif
#ifdef USE_SPLASH
#   include <splash/splash.h>
#endif
#include <string>                   // std::string
#include <utility>                  // std::pair
#include <cstddef>                  // NULL
#include <sstream>
#include <cassert>

#include "algorithms/vectorReduce.hpp" // vectorMax
#include "io/writeOutFuncs/writeOutFuncs.hpp"


namespace imresh
{
namespace io
{
namespace writeOutFuncs
{


    void justFree
    (
        float * _mem,
        std::pair<unsigned, unsigned> const _size,
        std::string const _filename
    )
    {
        if ( _mem != NULL )
        {
            free( _mem );
            _mem = NULL;
        }
#       ifdef IMRESH_DEBUG
            std::cout << "imresh::io::writeOutFuncs::justFree(): Freeing data ("
                << _filename << ")." << std::endl;
#       endif
    }

#   ifdef USE_PNG
        void writeOutPNG
        (
            float const * const _mem,
            std::pair<unsigned,unsigned> const _size,
            std::string const _filename
        )
        {
            auto const & Nx = _size.first;
            auto const & Ny = _size.second;

            pngwriter png( Nx, Ny, 0, _filename.c_str( ) );

            float max = algorithms::vectorMax( _mem, Nx * Ny );
            for( unsigned iy = 0; iy < Ny; ++iy )
            {
                for( unsigned ix = 0; ix < Nx; ++ix )
                {
                    auto const index = iy * Nx + ix;
                    assert( index < Nx * Ny );
                    const auto & value = _mem[index] / max;
                    if ( not ( value == value ) ) // isNaN
                        png.plot( ix, iy, 255, 0, 0 );
                    else
                        png.plot( ix, iy, value, value, value );
                }
            }

            png.close( );
#           ifdef IMRESH_DEBUG
                std::cout << "imresh::io::writeOutFuncs::writeOutPNG(): "
                             "Successfully written image data to PNG ("
                          << _filename << ")." << std::endl;
#           endif
        }
#   endif

#   ifdef USE_SPLASH
        void writeOutHDF5
        (
            float const * const _mem,
            std::pair<unsigned, unsigned> const _size,
            std::string const _filename
        )
        {
            splash::SerialDataCollector sdc( 0 );
            splash::DataCollector::FileCreationAttr fCAttr;
            splash::DataCollector::initFileCreationAttr( fCAttr );

            fCAttr.fileAccType = splash::DataCollector::FAT_CREATE;

            sdc.open( _filename.c_str( ), fCAttr );

            splash::ColTypeFloat cTFloat;
            splash::Dimensions size( _size.first, _size.second, 1 );

            sdc.write( 0,
                       cTFloat,
                       2,
                       splash::Selection( size ),
                       _filename.c_str( ),
                       _mem );

            sdc.close( );
#           ifdef IMRESH_DEBUG
                std::cout << "imresh::io::writeOutFuncs::writeOutHDF5(): "
                             "Successfully written image data to HDF5 ("
                          << _filename << "_0_0_0.h5)." << std::endl;
#           endif
        }
#   endif


} // namespace writeOutFuncs
} // namespace io
} // namespace imresh

