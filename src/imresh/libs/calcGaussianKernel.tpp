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


#include "calcGaussianKernel.hpp"

#include <algorithm>    // std::min
#include <cassert>
#include <cmath>
#ifndef M_PI
#   define M_PI 3.141592653589793238462643383279502884
#endif
#include <cstddef>      // NULL


namespace imresh
{
namespace libs
{


    template<class T_Prec>
    int calcGaussianKernel
    (
        double       const rSigma   ,
        T_Prec *     const rWeights ,
        unsigned int const rnWeights,
        double       const rMinAbsoluteError
    )
    {
        assert( rWeights != NULL );
        assert( rMinAbsoluteError > 0 );

        /**
         * @todo inverfc, e.g. with minimax (port python version to C/C++)
         * the inverse erfc diverges at 0, this makes it hard to find a
         * a polynomial approximation there, but maybe I could rewrite
         * minimax algorithm to work with @f[ \sum a_n/x**n @f]
         * Anyway, the divergence is also bad for the kernel Size. In order
         * to reach floating point single precision of 1e-7 absolute error
         * the kernel size would be: 3.854659 ok, it diverges much slower
         * than I thought
         **/
        //const int nNeighbors = ceil( erfcinv( 2.0*rMinAbsoluteError ) - 0.5 );
        assert( rSigma >= 0 );
        const int nNeighbors = ceil( 2.884402748387961466 * rSigma - 0.5 );
        const int nWeights   = 2*nNeighbors + 1;
        assert( nWeights > 0 );
        if ( rWeights == NULL or (unsigned) nWeights > rnWeights )
            return nWeights;

        double sumWeightings = 0;
        /* Calculate the weightings. I'm not sure, if this is correct.
         * I mean it could be, that the weights are the integrated gaussian
         * values over the pixel interval, but I guess that would force
         * no interpolation. Depending on the interpolation it wouldn't even
         * be pixel value independent anymore, making this useless, so I guess
         * the normal distribution evaluated at -1,0,1 for a kernel size of 3
         * should be correct ??? */
        const double a =  1.0/( sqrt(2.0*M_PI)*rSigma );
        const double b = -1.0/( 2.0*rSigma*rSigma );
        for ( int i = -nNeighbors; i <= nNeighbors; ++i )
        {
            const T_Prec weight = T_Prec( a*exp( i*i*b ) );
            rWeights[nNeighbors+i] = weight;
            sumWeightings += weight;
        }

        /* scale up or down the kernel, so that the sum of the weights will be 1 */
        for ( int i = -nNeighbors; i <= nNeighbors; ++i )
            rWeights[nNeighbors+i] /= sumWeightings;

        return nWeights;
    }


    /**
     * Same as calcGaussianKernel, but creates 2D kernel of arbirtrary
     * size by padding it with zeros.
     *
     * For use as a Gaussian Blur Kernel rCenterX and rCenterY should be 0.
     */
    template<class T_Prec>
    void calcGaussianKernel2d
    (
        double       const rSigma    ,
        unsigned int const rCenterX  ,
        unsigned int const rCenterY  ,
        T_Prec *     const rWeights  ,
        unsigned int const rnWeightsX,
        unsigned int const rnWeightsY
    )
    {
        assert( rSigma >= 0 );
        assert( rnWeightsX >= 0 );
        assert( rnWeightsY >= 0 );
        assert( rCenterX <= rnWeightsX );
        assert( rCenterY <= rnWeightsY );
        assert( rWeights != NULL );

        auto const sigmaX = rSigma;
        auto const sigmaY = rSigma;

        double sumWeightings = 0;
        /* this normalization factor is not used, because it is only true
         * for a continuous Gaussian, but we evaluate it at discrete points */
        double const a  =  1.0/( 2.0 * M_PI * sigmaX * sigmaY );
        double const bx = -1.0/( 2.0 * sigmaX * sigmaX );
        double const by = -1.0/( 2.0 * sigmaY * sigmaY );

        for ( auto ix = 0u; ix < rnWeightsX; ++ix )
        for ( auto iy = 0u; iy < rnWeightsY; ++iy )
        {
            /* Only plot all values rnWeightsX/2 and rnWeightsY/2 around
             * the center and then periodically shift it.
             * E.g. if rCenterX=0 and and we look at pixel rnWeightsX-1, then
             * we find it mapped to
             */
            auto const dx = std::abs( ix - rCenterX );
            auto const dy = std::abs( iy - rCenterY );
            /* This is problematic to do with fmod, because of rounding errors
             * For 0 <= dx < rnWeightsX the maximum value of x will be
             * rnWeightsX / 2 */
            auto const x  = std::min( dx, std::abs( dx - rnWeightsX ) );
            auto const y  = std::min( dy, std::abs( dy - rnWeightsY ) );

            const T_Prec weight = T_Prec( a*exp( x*x*bx + y*y*by ) );

            sumWeightings += weight;
            rWeights[ iy * rnWeightsX + ix ] = weight;
        }

        for ( auto i = 0u; i <= rnWeightsX * rnWeightsY; ++i )
            rWeights[i] /= sumWeightings;
    }


} // namespace libs
} // namespace imresh