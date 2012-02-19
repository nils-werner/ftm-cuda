.PHONY: all clear

LIBS = -lasound

all: bin/countcards bin/countwave bin/listpcm bin/playback bin/iirfilter

clear:
	- rm bin/*

bin/countwave:
	gcc -o bin/countwave countwave.c $(LIBS)

bin/listpcm:
	gcc -o bin/listpcm listpcm.c $(LIBS)

bin/countcards:
	gcc -o bin/countcards countcards.c $(LIBS)

bin/playback:
	gcc -o bin/playback playback.c $(LIBS)

bin/iirfilter:
	gcc -o bin/iirfilter iirfilter.c $(LIBS)