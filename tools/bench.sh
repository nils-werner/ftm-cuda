#!/bin/bash

unset IFS

MODES=( gpu cpu )
MATRIXMODES=( gpu cpu )
TRIES=( 1 )
FILTERS=( 30 )
CHUNKSIZES=( 100 )
BLOCKSIZES=( 16 )
MATRIXBLOCKSIZES=( 64 )
MESSAGE=""


while getopts ":c:f:b:t:s:p:m:d:h" opt; do
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

							echo "($i/$total) ./build/iirfilter $modeswitch $matrixmodeswitch -f $filter -c $chunk -b $block -m $matrixblock -x"
							./build/iirfilter $modeswitch $matrixmodeswitch -f $filter -c $chunk -b $block -m $matrixblock -x >> bench.xml;
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
