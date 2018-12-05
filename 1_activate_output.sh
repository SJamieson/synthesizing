#!/bin/bash
set -e
TAG=${3:-synthesizing}
NETDIR=${1:-nets}
RESULTDIR=${2:-../NetDissect-Lite/result}

cornetz='CORnet-Z'
cornets='CORnet-S'
networks=($cornetz $cornets)

declare -A layermap
layermap[$cornetz]='MaxPool2d_4'
layermap[$cornets]='Add_8'

declare -A netmap
declare -A weightmap
netmap[$cornetz]="$NETDIR/$cornetz/Sequential.prototxt"
weightmap[$cornetz]="$NETDIR/$cornetz/Sequential.caffemodel"
netmap[$cornets]="$NETDIR/$cornets/Sequential.prototxt"
weightmap[$cornets]="$NETDIR/$cornets/Sequential.caffemodel"

declare -A tallyfile
tallyfile[$cornetz]="$RESULTDIR/pytorch_cornetz_imagenet/tally.csv"
tallyfile[$cornets]="$RESULTDIR/pytorch_cornets_imagenet/tally.csv"

# Hyperparam settings for visualizing AlexNet
iters="600"
weights="99"
rates="0.5" # Must be x.y floats
end_lr=1e-10
xys="-1"
opt_layer="fc6"

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
output_dir="output/$TAG-`date +"%T"`"
#rm -rf ${output_dir}
mkdir -p ${output_dir}

for network in ${networks}; do

    list_files=""

    testunits=$(python extract_samples.py ${tallyfile[$network]} ${count:-5})

    for test in ${testunits}; do

        act_layer=${layermap[$network]}
        unit=$(cut -d'_' -f1 <<< $test)
        label=$(cut -d'_' -f3 <<< $test)

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
                  --init_file ${init_file} \
                  --net_definition ${netmap[$network]} \
                  --net_weights ${weightmap[$network]} \
                  --tag ${test}

              # Add a category label to each image
              unit_pad=`printf "%04d" ${unit}`
              f=${output_dir}/${test}__${act_layer}_${unit_pad}_${n_iters}_${L2}_${xy}_${lr}__${seed}.jpg
              convert $f -gravity south -splice 0x10 $f
              convert $f -append -gravity Center -pointsize 30 label:"$label" -bordercolor white -border 0x0 -append $f

              list_files="${list_files} ${f}"
            done
          done
          done
        done
      done
    done

    # Make a collage
    output_file=${output_dir}/${network}.jpg
    montage ${list_files} -tile 5x3 -geometry +1+1 ${output_file}
    convert ${output_file} -trim ${output_file}
    echo "=============================="
    echo "Result of example 1: [ ${output_file} ]"

done