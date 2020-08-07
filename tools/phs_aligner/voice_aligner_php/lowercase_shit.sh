!#/bin/bash


for f in data/*.lab
do
	filename=$(basename $f)
	echo $filename
	sed 's/[[:upper:]]*/\L&/g' < $f > delmee/$filename
done
