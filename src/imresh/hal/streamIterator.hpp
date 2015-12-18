/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015 Helmholtz-Zentrum Dresden - Rossendorf
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

#pragma once

#include <cuda_runtime.h>   // cudaStream_t
#include <iterator>         // std::iterator, std::forward_iterator_tag

namespace imresh
{
namespace hal
{
    /**
     * Convenient wrapper to iterate over all available devices.
     *
     * This can be handy while distributing work over the used GPUs.
     */
    class StreamIterator :
        public std::iterator<std::forward_iterator_tag, cudaStream_t>
    {
    public:
        StreamIterator( cudaStream_t* _item ) :
            item( _item )
        { }

        StreamIterator( const StreamIterator& _iter ) :
            item( _iter.item )
        { }

        StreamIterator& operator++( )
        {
            ++item;
            return *this;
        }

        bool operator==( const StreamIterator& rhs )
        {
            return item == rhs.item;
        }

        bool operator!=( const StreamIterator& rhs )
        {
            return item != rhs.item;
        }

        cudaStream_t& operator*( )
        {
            return *item;
        }

    private:
        cudaStream_t* item;
    };
} // namespace hal
} // namespace imresh
