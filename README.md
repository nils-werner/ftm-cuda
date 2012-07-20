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

The synthesizer is written in C and needs you to have libsndfile to be installed. To compile it run

    $ make

All necessary CUDA-paths are set in the Makefile. You may need to edit it in order to compile it successfully.



Using
-----

The executable will be named `build/iirfilter`. It will produce a WAV file named `filter.wav` wich can be played back by any media player software.

Settings may be changed using parameters. Use `build/iirfilter -h` to see what parameters are available.

Benchmarking
------------

Benchmarking is done using the bash-script `tools/bench.sh`. Again, see `tools/bench.sh -h` for all available parameters.

All benchmarks are saved in unique files so you can investigate past benchmarks and maybe compare them with newer ones.

A common mistake is to i.e. parameterize filters using `tools/bench.sh -f 128` while the correct syntax is `tools/bench.sh -f 128:128`. In the first
case, the script would iterate 1...128 while in the second one it would only pick 128. It's written in the -h docs though.
