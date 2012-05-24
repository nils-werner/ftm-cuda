#!/bin/bash

unset IFS

MODES=( gpu cpu )
FILTERS=( 30 60 90 120 150 180 210 240 270 300 330 360 390 420 450 )
CHUNKSIZES=( 25 50 75 100 125 150 175 200 225 250 )
TRIES=( 1 2 3 4 5 )

rm bench.xml

echo "<?xml version=\"1.0\" ?>" >> bench.xml
echo "<benchmark>" >> bench.xml

for mode in ${MODES[@]}
do
	for filter in ${FILTERS[@]}
	do
		for chunk in ${CHUNKSIZES[@]}
		do
			for try in ${TRIES[@]}
			do
				echo $mode $filter $chunk
				./build/iirfilter $mode xml $filter $chunk >> bench.xml;
			done
		done
	done
done

echo "</benchmark>" >> bench.xml
