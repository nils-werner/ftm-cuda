.PHONY: all clear

DEBUG = 0
LIBS = -lasound
OBJS = classes/Filter.o classes/Matrix.o classes/Buffer.o
CPPFLAGS := -DDEBUG=$(DEBUG)

all: bin/countcards bin/countwave bin/listpcm bin/playback bin/iirfilter bin/matrixtest

clear:
	- rm bin/*
	- rm classes/*.o
	- rm *.o

bin/countwave: alsa/countwave.c
	gcc -o bin/countwave alsa/countwave.c $(LIBS)

bin/listpcm: alsa/listpcm.c
	gcc -o bin/listpcm alsa/listpcm.c $(LIBS)

bin/countcards: alsa/countcards.c
	gcc -o bin/countcards alsa/countcards.c $(LIBS)

bin/playback: alsa/playback.c
	gcc -o bin/playback alsa/playback.c $(LIBS)

bin/iirfilter: main.cpp $(OBJS)
	g++ -o bin/iirfilter main.cpp $(LIBS) $(CPPFLAGS)

bin/matrixtest: matrixtest.cpp $(OBJS)
	g++ -o bin/matrixtest matrixtest.cpp $(LIBS) $(CPPFLAGS)

%.o: %.cpp %.h
	g++ -c $(FLAGS) -o $@ $<
