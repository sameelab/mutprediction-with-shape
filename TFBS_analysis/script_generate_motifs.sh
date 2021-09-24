#!/bin/bash

# Read file

for TF in $(cat uniq_motifnames.txt); do

  sed -n '/^>'"$TF"'/,/^>/p' logos/motifs_logo.txt |\
    sed '$d' > "logos/motif_"$TF".txt"

done

sed -n '/^>ZSCAN4_3/,/^>/p' logos/motifs_logo.txt > "logos/motif_ZSCAN4_3.txt"
