#!/bin/bash

unset IFS

MODES=( gpu )
MATRIXMODES=( gpu )
TRIES=( 1 2 3 )
FILTERS=( 32 )
CHUNKSIZES=( 720 )
BLOCKSIZES=( 6 )
MATRIXBLOCKSIZES=( 128 )
MESSAGE=""


while getopts ":c:f:b:t:s:p:m:d:ho" opt; do
	case $opt in
		c)
			CHUNKSIZES=( $(echo $OPTARG | sed -e "s/:/ /g" | xargs seq -s " ") )
		;;
		f)
			FILTERS=( $(echo $OPTARG | sed -e "s/:/ /g" | xargs seq -s " ") )
		;;
		b)
			BLOCKSIZES=( $(echo $OPTARG | sed -e "s/:/ /g" | xargs seq -s " ") )
		;;
		m)
			MATRIXBLOCKSIZES=( $(echo $OPTARG | sed -e "s/:/ /g" | xargs seq -s " ") )
		;;
		t)
			TRIES=( $(seq -s " " $OPTARG) )
		;;
		s)
			MODES=( $OPTARG )
		;;
		p)
			MATRIXMODES=( $OPTARG )
		;;
		d)
			MESSAGE=$OPTARG
		;;
		o)
			FILTERS=( 30 90 150 210 270 330 390 450 500 550 600 650 700 750 800 850 900 950 )
			CHUNKSIZES=( 25 50 75 100 125 150 175 200 225 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000 )
			TRIES=( 1 2 3 4 5 )
		;;
		h)
			echo "Optionen:"
			echo " -c [start:[schrittgroesse:]]ende Chunkgroessen"
			echo " -f [start:[schrittgroesse:]]ende Filteranzahl"
			echo " -b [start:[schrittgroesse:]]ende CUDA-Blockgroessen"
			echo " -m [start:[schrittgroesse:]]ende CUDA-Blockgroessen fuer Matrixerzeugung"
			echo " -t Anzahl Versuche"
			echo " -s Signal berechnen auf [cpu|gpu|cpu gpu]"
			echo " -p Matrizen berechnen auf [cpu|gpu|cpu gpu]"
			echo " -d Testbeschreibung"
			echo " -o Original-Testbereiche"
			exit 0
		;;
		\?)
			echo "Invalid option: -$OPTARG" >&2
			exit 1
		;;
		:)
			echo "Option -$OPTARG requires an argument." >&2
			exit 1
		;;
	esac
done

if [ -n "$MESSAGE" ]; then
	MESSAGE=-$MESSAGE
fi

total=$(expr ${#MODES[@]} \* ${#MATRIXMODES[@]} \* ${#FILTERS[@]} \* ${#CHUNKSIZES[@]} \* ${#MATRIXBLOCKSIZES[@]} \* ${#BLOCKSIZES[@]} \* ${#TRIES[@]})
i=0

START=$(date +%s)

rm bench.xml

echo "<?xml version=\"1.0\" ?>" >> bench.xml
echo "<benchmark>" >> bench.xml

for mode in ${MODES[@]}
do
	for matrixmode in ${MATRIXMODES[@]}
	do
		for matrixblock in ${MATRIXBLOCKSIZES[@]}
		do
			for block in ${BLOCKSIZES[@]}
			do
				for filter in ${FILTERS[@]}
				do
					for chunk in ${CHUNKSIZES[@]}
					do
						for try in ${TRIES[@]}
						do
							i=$(expr $i + 1)
							if [ $mode == "gpu" ]; then
								modeswitch="-g"
							else
								modeswitch=""
							fi

							if [ $matrixmode == "gpu" ]; then
								matrixmodeswitch="-p"
							else
								matrixmodeswitch=""
							fi

							echo "<run>" >> bench.xml;
							echo "<settings mode=\"$mode\" matrixmode=\"$matrixmode\" blocksize=\"$block\" matrixblocksize=\"$matrixblock\" filters=\"$filter\" chunksize=\"$chunk\" samples=\"0\" />" >> bench.xml
							echo "($i/$total) ./build/iirfilter $modeswitch $matrixmodeswitch -f $filter -c $chunk -b $block -m $matrixblock -x"
							./build/iirfilter $modeswitch $matrixmodeswitch -f $filter -c $chunk -b $block -m $matrixblock -x >> bench.xml;
							echo "</run>" >> bench.xml;
						done
					done
				done
			done
		done
	done
done

END=$(date +%s)
DIFF=$(( $END - $START ))
echo "<timing seconds=\"$DIFF\" />" >> bench.xml

echo "</benchmark>" >> bench.xml

xsltproc tools/benchtocsv.xsl bench.xml > bench.csv

mv bench.xml "bench/bench-`date +%y%m%d-%H%M`$MESSAGE.xml"
mv bench.csv "bench/bench-`date +%y%m%d-%H%M`$MESSAGE.csv"
