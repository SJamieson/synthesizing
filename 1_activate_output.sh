#/bin/bash

# Take in an unit number
#if [ "$#" -ne "1" ]; then
#  echo "Provide 1 output unit number e.g. 945 for bell pepper."
#  exit 1
#fi

# Get label for each unit
path_labels="misc/synset_words.txt"
IFS=$'\n' read -d '' -r -a labels < ${path_labels}

opt_layer=fc6
act_layer=$1
output_layer=${2:-true}
#act_layer=Addmm_1 # CORnet decoder
#act_layer=Add_8 #CORnet-S IT
#act_layer=MaxPool2d_4 #CORnet-Z IT
#units="149 396 323 128" #"${1}"
units="240 241 152 371"
xys="-1 3"

# Hyperparam settings for visualizing AlexNet
iters="600"
weights="99"
rates="0.5" # Must be x.y floats
end_lr=1e-10

# Clipping
clip=0
multiplier=3
bound_file=act_range/${multiplier}x/${opt_layer}.txt
init_file="None"

# Debug
debug=0
if [ "${debug}" -eq "1" ]; then
  rm -rf debug
  mkdir debug
fi

# Output dir
output_dir="output/$act_layer-`date +"%T"`"
#rm -rf ${output_dir}
mkdir -p ${output_dir}

list_files=""

# Sweeping across hyperparams
for unit in ${units}; do

  # Get label for each unit
  label_1=`echo ${labels[unit]} | cut -d "," -f 1 | cut -d " " -f 2`
  label_2=`echo ${labels[unit]} | cut -d "," -f 1 | cut -d " " -f 3`
  label="${label_1} ${label_2}"

  for seed in {0..0}; do
  #for seed in {0..8}; do

    for n_iters in ${iters}; do
      for w in ${weights}; do
      for xy in ${xys}; do
        for lr in ${rates}; do

          L2="0.${w}"

          # Optimize images maximizing fc8 unit
          python ./act_max.py \
              --act_layer ${act_layer} \
              --opt_layer ${opt_layer} \
              --unit ${unit} \
              --xy ${xy} \
              --n_iters ${n_iters} \
              --start_lr ${lr} \
              --end_lr ${end_lr} \
              --L2 ${L2} \
              --seed ${seed} \
              --clip ${clip} \
              --bound ${bound_file} \
              --debug ${debug} \
              --output_dir ${output_dir} \
              --init_file ${init_file}

          # Add a category label to each image
          unit_pad=`printf "%04d" ${unit}`
          f=${output_dir}/${act_layer}_${unit_pad}_${n_iters}_${L2}_${xy}_${lr}__${seed}.jpg
          convert $f -gravity south -splice 0x10 $f
          if [[ $output_layer ]]; then
          	convert $f -append -gravity Center -pointsize 30 label:"$label" -bordercolor white -border 0x0 -append $f
          fi

          list_files="${list_files} ${f}"
        done
      done
      done
    done
  
  done
done

# Make a collage
output_file=${output_dir}/example1.jpg
montage ${list_files} -tile 5x1 -geometry +1+1 ${output_file}
convert ${output_file} -trim ${output_file}
echo "=============================="
echo "Result of example 1: [ ${output_file} ]"
