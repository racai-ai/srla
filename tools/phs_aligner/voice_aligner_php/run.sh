#!/bin/sh

echo "Trebuie sa fie .txt + .wav in data"
echo "Fisierele tb sa fie convertite 10KHz mono"
echo "txt se poate converti in lab cu make_lab.php"
echo "make_lab.php este inclus in acest script, dar e posibil sa fie comentat"
echo ""
echo "Rulare recomandata:"
echo "php run.php 2>&1 | tee run.log"
echo ""
echo "Press enter"
read a

rm -f aligned/*
rm -f mfc/*

for i in `seq 0 19`; do
rm -f hmm$i/*
done

#exit

export PATH=/opt/htk/bin:$PATH

php make_lab.php

php run.php

