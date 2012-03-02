
Functional Model Based Guitar String Synthesizer
================================================

 1. Fetching Sources
 2. Compiling
 3. Using



Fetching Sources
----------------

To fetch the sources you need to have Git installed. The repository is located in `~/SHARED_FILES/werner/C` so to clone it you need to run

    $ git clone ~/SHARED_FILES/werner/C



Compiling
---------

The synthesizer is written in C++ and needs you to have libsndfile to be installed. To compile it run

    $ make



Using
-----

The executable will be named `build/iirfilter`. It will produce a WAV file named `filter.wav` wich can be played back by any media player software.
