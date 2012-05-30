#!/bin/bash

unset IFS

MODES=( gpu cpu )
TRIES=( 1 )

case $1 in
	"filters")
		MODES=( gpu )
		FILTERS=( 30 90 150 210 270 330 390 450 500 550 600 650 700 750 )
		CHUNKSIZES=( 100 )
		;;
	"chunks")
		MODES=( gpu )
		FILTERS=( 30 )
		CHUNKSIZES=( 25 50 75 100 125 150 175 200 225 250 )
		;;
	"all")
		FILTERS=( 30 90 150 210 270 330 390 450 500 550 600 650 700 750 800 850 900 950 1000 1050 1100 1150 1200 1250 )
		CHUNKSIZES=( 25 50 75 100 125 150 175 200 225 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000 1050 1100 1150 1200 1250 1300 1350 1400 1450 1500 )
		TRIES=( 1 2 3 4 5 )
		;;
	*)
		echo "$0 [filters|chunks|all]"
esac

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
				if [ $mode == "gpu" ]; then
					modeswitch="-g"
				else
					modeswitch=""
				fi

				echo "./build/iirfilter $modeswitch -f $filter -c $chunk -x"
				./build/iirfilter $modeswitch -f $filter -c $chunk -x >> bench.xml;
			done
		done
	done
done

echo "</benchmark>" >> bench.xml

xsltproc tools/benchtocsv.xsl bench.xml > bench.csv
