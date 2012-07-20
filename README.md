Functional Model Based Guitar String Synthesizer
================================================

 1. Fetching Sources
 2. Compiling
 3. Using
 4. Benchmarking
 5. Visualizing



Fetching Sources
----------------

The sourcecode of this project is located in this folder. To fetch them you need to simply copy them on your harddrive.
Since this directory is also a Git repository you could alternatively use Git to clone it:

    $ git clone /path/to/CD/rom/C

It is definitely recommended to use Git instead of simply copying it since you will be able to inspect the changelog of the entire development process of this tool.
Also, you would be able to seamlessly start tracking your development in the very same repository.



Compiling
---------

The synthesizer is written in C and needs you to have libsndfile to be installed. To compile it run

    $ make

All necessary CUDA-paths are set in the Makefile. You may need to edit it in order to compile it successfully.



Using
-----

The resulting executable will be named `build/iirfilter`. It will produce a WAV file named `filter.wav` wich can be played back by any media player software.

Settings may be changed using parameters. Use `build/iirfilter -h` to see what parameters are available.



Benchmarking
------------

Benchmarking is done using the bash-script `tools/bench.sh`. Again, see `tools/bench.sh -h` for all available parameters.

All benchmarks are saved in unique files so you can investigate past benchmarks and maybe compare them with newer ones.

A common mistake is to i.e. parameterize filters using `tools/bench.sh -f 128` while the correct syntax is `tools/bench.sh -f 128:128`. In the first
case, the script would iterate 1...128 while in the second one it would only pick 128. It's written in the -h docs though.

Visualizing
-----------

Visualization of the benchmark data is done in Matlab. To use it, open `tools/plotbench.m` in Matlab but make sure your path is set to be the root of this project
(otherwise it won't find the data in `bench/*`). Upon running plotbench.m, Matlab will first ask what file to open, then what data to load and then what data to eliminate.

This elimiation process is important as data is being generated in 4 dimensions (blocksize, matrixblocksize, filters and chunksize) but Matlab is only capable
of visualizing 2 dimensions in one surface plot. So make sure to eliminate at least two fields (don't select the same field twice) and also make sure to eliminate
all fields with only one entry, otherwise you'd end up with just 1D of data in a 2D plot and Matlab will complain.

If one of the remaining fields is the chunksize-field, the roundtrip data will automatically be weighted to represend "multiples of playbackspeed".
