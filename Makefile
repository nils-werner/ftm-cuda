.PHONY: all clean preview

DEBUG = 0
LIBS = -lasound -lsndfile
OBJS = classes/Filter.o classes/Matrix.o classes/BlockDiagMatrix.o classes/Buffer.o
CPPFLAGS := -DDEBUG=$(DEBUG)

all: build/countcards build/countwave build/listpcm build/playback build/iirfilter build/matrixtest

clean:
	- rm -f build/*
	- rm -f classes/*.o
	- rm -f *.o

time: build/iirfilter
	time -p ./build/iirfilter

preview: time
	cvlc filter.wav vlc://quit

build/countwave: alsa/countwave.c
	gcc -o build/countwave alsa/countwave.c $(LIBS)

build/listpcm: alsa/listpcm.c
	gcc -o build/listpcm alsa/listpcm.c $(LIBS)

build/countcards: alsa/countcards.c
	gcc -o build/countcards alsa/countcards.c $(LIBS)

build/playback: alsa/playback.c
	gcc -o build/playback alsa/playback.c $(LIBS)

build/iirfilter: main.cpp main.h $(OBJS)
	g++ -o build/iirfilter main.cpp $(LIBS) $(CPPFLAGS)

build/matrixtest: matrixtest.cpp $(OBJS)
	g++ -o build/matrixtest matrixtest.cpp $(LIBS) $(CPPFLAGS)

%.o: %.cpp %.h
	g++ -c $(FLAGS) -o $@ $<
