#!/bin/sh

for f in data_new/*.wav ; do
echo $f

rm -f data/temp.wav

sox $f data/temp.wav rate 16000 channels 1

#exit 1

b=$(basename $f)

rm -f $f
mv data/temp.wav data/$b


done
