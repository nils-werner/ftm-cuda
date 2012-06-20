#!/bin/bash

unset IFS

MODES=( gpu cpu )
MATRIXMODES=( gpu cpu )
TRIES=( 1 )
FILTERS=( 30 )
BLOCKSIZES=( 100 )
MESSAGE=""


while getopts ":b:f:t:c:p:m:dh" opt; do
	case $opt in
		b)
			BLOCKSIZES=( $(echo $OPTARG | sed -e "s/:/ /g" | xargs seq -s " ") )
		;;
		f)
			FILTERS=( $(echo $OPTARG | sed -e "s/:/ /g" | xargs seq -s " ") )
		;;
		t)
			TRIES=( $(seq -s " " $OPTARG) )
		;;
		c)
			MODES=( $OPTARG )
		;;
		p)
			MATRIXMODES=( $OPTARG )
		;;
		m)
			MESSAGE=$OPTARG
		;;
		d)
			FILTERS=( 30 90 150 210 270 330 390 450 500 550 600 650 700 750 800 850 900 950 )
			BLOCKSIZES=( 25 50 75 100 125 150 175 200 225 250 300 350 400 450 500 550 600 650 700 750 800 850 900 950 1000 )
			TRIES=( 1 2 3 4 5 )
		;;
		h)
			echo "Optionen:"
			echo " -b [start:[schrittgroesse:]]ende Blockgroessen"
			echo " -f [start:[schrittgroesse:]]ende Filteranzahl"
			echo " -t Anzahl Versuche"
			echo " -c Signal berechnen auf [cpu|gpu|cpu gpu]"
			echo " -p Matrizen berechnen auf [cpu|gpu|cpu gpu]"
			echo " -m Testbeschreibung"
			echo " -d Standardwerte verwenden"
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

total=$(expr ${#MODES[@]} \* ${#MATRIXMODES[@]} \* ${#FILTERS[@]} \* ${#BLOCKSIZES[@]} \* ${#TRIES[@]})
i=0

START=$(date +%s)

rm bench.xml

echo "<?xml version=\"1.0\" ?>" >> bench.xml
echo "<benchmark>" >> bench.xml

for mode in ${MODES[@]}
do
	for matrixmode in ${MATRIXMODES[@]}
	do
		for filter in ${FILTERS[@]}
		do
			for block in ${BLOCKSIZES[@]}
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
						matrixmodeswitch="-b"
					else
						matrixmodeswitch=""
					fi

					echo "($i/$total) ./build/iirfilter $modeswitch $matrixmodeswitch -f $filter -c $block -x"
					./build/iirfilter $modeswitch $matrixmodeswitch -f $filter -c $block -x >> bench.xml;
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
