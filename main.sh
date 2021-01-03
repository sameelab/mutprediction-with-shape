#!/bin/bash
#main.sh

# A single master script for running an end-to-end pipeline

echo "Create runtime log to show how long each script takes:\n"
timelog="main_runtime.log"
touch $timelog

echo -e "Showing the configs.sh file, please proofread to ensure there are no errors: \n\n"
cat configs.sh


echo -e "Step 1: generate reference samples from sample metatable\n"
echo "Step 1" >> $timelog
/usr/bin/time -o $timelog -a \
  bash script_get_sampleids.sh

echo -e "Step 2: download VCF files in batch\n"
echo "Step 2" >> $timelog
#/usr/bin/time -o $timelog -a \
  #bash pipe_download_gnomad.sh
echo -e "This step is currently skipped.\n"

echo -e "Step 3: batch-process VCF files using multiple samples\n"
echo "Step 3" >> $timelog
#/usr/bin/time -o $timelog -a \
  #bash pipe_vcf_process.sh
echo -e "Since step 2 is skipped, this step is temporarily skipped as well.\n"

echo -e "Step 4: obtain counts for unique k-mers within certain genomic regions\n"
echo "Step 4" >> $timelog
/usr/bin/time -o $timelog -a \
  bash pipe_countkmer.sh

echo -e "Step 5: batch-process the .frq files and obtain counts for unique k-mer mutation patterns within certain genomic regions\n"
echo "Step 5" >> $timelog
/usr/bin/time -o $timelog -a \
  bash pipe_sumbykmer.sh

echo -e "Step 6: make aggregated .csv files of counts for uniqe k-mer mutation patterns within certain genomic regions\n"
echo "Step 6" >> $timelog
/usr/bin/time -o $timelog -a \
  bash pipe_getcounts.sh

echo -e "Step 7: calculate mutation rates and ratios of certain AF cutoffs using aggregated .csv files from step 5\n"
echo "Step 7" >> $timelog
/usr/bin/time -o $timelog -a \
  bash pipe_getrates.sh

echo "All parts of the pipeline are now complete. Please see error log to check if anything has gone wrong."
