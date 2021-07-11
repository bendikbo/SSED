#!/usr/bin/bash

cd "$(dirname "$0")"
proj_dir=$(dirname "$0")
conf_dir="configs"
experiment_dir="window_size_thr_2"
for config_file in "${proj_dir}/${conf_dir}/${experiment_dir}"/*
do
  #Getting conf_name for output logfile
  conf_name="$(basename $config_file)"
  conf_name=$(echo "$conf_name" | cut -f 1 -d '.')
  env/bin/python train.py $config_file > $proj_dir/outputs/$conf_name.log
done
