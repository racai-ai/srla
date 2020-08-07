#!/bin/sh

for f in data_new/*.wav ; do
echo $f

rm -f data/temp.wav

duration=`soxi $f | grep Duration | sed 's/.*: \([^ ]*\).*/\1/'`
length=`date +%M:%S.%2N --date "$duration UTC -2 sec"`

#echo $length
#exit 1

sox $f data/temp.wav rate 16000 channels 1 trim 0:01 $length

#exit 1

b=$(basename $f)

rm -f $f
mv data/temp.wav data/$b


done
