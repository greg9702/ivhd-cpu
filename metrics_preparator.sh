#!/bin/bash

# path to folder from project root directory 
folder=$1

# file names should be TESTNAME_result_iteration
file_name_prefix=$2

# maximum iteration number
maximum_iteration_number=$3

# maximum iteration number
iteration_step=$4

# number of neighbours
neighbours_number=$5

# output file
output_file=$6

iteration=0

# clear output file
> $output_file

while [ $iteration -lt $maximum_iteration_number ]
  do
  tmp=`python knn_metric.py $folder/$file_name_prefix\_result\_$iteration $neighbours_number`
  echo $iteration $tmp | tee -a $output_file
  iteration=$((iteration+iteration_step))
done