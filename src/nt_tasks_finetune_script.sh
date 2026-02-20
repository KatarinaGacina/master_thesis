#!/bin/bash

TASKS=(
    promoter_all
    promoter_tata
    promoter_no_tata
    enhancers
    enhancers_types
    splice_sites_all
    splice_sites_acceptors
    splice_sites_donors
    H3
    H4
    H3K9ac
    H3K14ac
    H4ac
    H3K4me1
    H3K4me2
    H3K4me3
    H3K36me3
    H3K79me3
)

for TASK in "${TASKS[@]}"
do
    echo "================================="
    echo "Running task: $TASK"
    echo "================================="

    python -m evaluation.nt_benchmark.finetune --task $TASK > logs/$TASK.log 2>&1

    echo "Finished task: $TASK"
done
