.PHONY: all clear

LIBS = -lasound

all: bin/countcards bin/countwave bin/listpcm bin/playback bin/iirfilter

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

bin/iirfilter: iirfilter.cpp classes/Matrix.o
	g++ -o bin/iirfilter iirfilter.cpp $(LIBS)

%o: %cpp
	g++ -c $(FLAGS) -o $@ $<