LIBS = -lasound

all: countcards countwave listpcm

countwave:
	gcc -o bin/countwave countwave.c $(LIBS)

listpcm:
	gcc -o bin/listpcm listpcm.c $(LIBS)

countcards:
	gcc -o bin/countcards countcards.c $(LIBS)

clear:
	- rm bin/*