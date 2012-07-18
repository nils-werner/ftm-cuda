#!/bin/bash

for file in `dir -d bench/*.xml` ; do
	basename=`basename $file .xml`
	echo $file
	xsltproc tools/benchtocsv.xsl $file > bench/$basename.csv
done
