
#include "sdl/sdlcommon.h"
#include "tests/testSdlPlot.cpp"
#include "tests/testDcft.cpp"
#include "tests/testDft.h"
#include "tests/testFftw.h"
#include "tests/testGaussian.h"
#include "tests/testGaussian2d.h"
#include "tests/testMatrixInvertGaussJacobi.cpp"


int main(void)
{
    SDL_Window   * pWindow;
    SDL_Renderer * pRenderer;

    /* Initialize SDL Context */
    SDL_CHECK( SDL_Init( SDL_INIT_VIDEO ) )


    const int numdrivers = SDL_GetNumRenderDrivers();
    std::cout << "Render driver count: " << numdrivers << "\n";
    for ( int i = 0; i < numdrivers; i++ )
    {
        SDL_RendererInfo drinfo;
        SDL_GetRenderDriverInfo (i, &drinfo);
        std::cout << "Driver name ("<<i<<"): " << drinfo.name << " flags: ";
        if (drinfo.flags & SDL_RENDERER_SOFTWARE)      printf("Software ");
        if (drinfo.flags & SDL_RENDERER_ACCELERATED)   printf("Accelerated ");
        if (drinfo.flags & SDL_RENDERER_PRESENTVSYNC)  printf("VSync ");
        if (drinfo.flags & SDL_RENDERER_TARGETTEXTURE) printf("Textures ");
        printf("\n");
    }


    pWindow = SDL_CreateWindow( "Output",
        SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        1024, 640, SDL_WINDOW_SHOWN );
    SDL_CHECK( pWindow );

    pRenderer = SDL_CreateRenderer( pWindow, -1, SDL_RENDERER_ACCELERATED );
    SDL_CHECK( pRenderer );

    SDL_SetRenderDrawColor( pRenderer, 255,255,255,255 );
    SDL_RenderClear( pRenderer );
    SDL_RenderPresent( pRenderer );

    /* Do and plot tests */
    SDL_SetRenderDrawColor( pRenderer, 0,0,0,255 );
    //sdlcommon::test::testSdlPlot(pRenderer);
    imresh::test::testMatrixInvertGaussJacobi();
    //imresh::test::testDcft(pRenderer);
    //imresh::test::testGaussian(pRenderer);
    imresh::test::testGaussian2d(pRenderer);
    //imresh::test::testDft(pRenderer);
    //imresh::test::testFftw(pRenderer);
    //imresh::test::testFftw2d(pRenderer);


    //SDL_drawLineControl drawControl;

    /* Wait for key to quit */
    int mainProgrammRunning = 1;
    int renderTouched = 1;
    while (mainProgrammRunning)
    {
        /* Handle Keyboard and Mouse events */
        SDL_Event event;
        while ( SDL_PollEvent(&event) )
        {
            mainProgrammRunning &= not SDL_basicControl(event,pWindow,pRenderer);
            SDL_SetRenderDrawColor( pRenderer, 128,0,0,255 );
            //renderTouched |= drawControl(event, pRenderer);
        }

        if ( renderTouched )
        {
            renderTouched = 0;
            SDL_RenderPresent( pRenderer );
        }
        SDL_Delay(50 /*ms*/);
    }

	SDL_DestroyWindow( pWindow );
	SDL_Quit();
    return 0;
}