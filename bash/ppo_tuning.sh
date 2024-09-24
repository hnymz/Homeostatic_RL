#!/bin/bash

#SBATCH --time=10:00:00
#SBATCH --mem=10G
#SBATCH -n 1
#SBATCH --nodes=1
#SBATCH -J MutliplePPOs
#SBATCH -o log/R-%x.%j.out

declare -a lrs=(0.00005)
declare -a Kepochs=(10)
declare -a gammas=(0.5)
declare -a hiddens=(512)
declare -a epsclips=(0.1 0.15 0.2 0.25 0.3)
declare -a updatefreqs=(20)

# Loop over parameters and submit jobs
for lr in "${lrs[@]}"; do
    for Kepoch in "${Kepochs[@]}"; do
        for gamma in "${gammas[@]}"; do
            for hidden in "${hiddens[@]}"; do
                for epsclip in "${epsclips[@]}"; do
                    for updatefreq in "${updatefreqs[@]}"; do
                        # Submit job with specific parameters
                        sbatch \
                            --job-name="PPO_${lr}_${Kepoch}_${gamma}_${hidden}_${epsclip}_${updatefreq}" \
                            --time=1:00:00 \
                            --mem=5G \
                            --nodes=1 \
                            -o "log/R-PPO_${lr}_${Kepoch}_${gamma}_${hidden}_${epsclip}_${updatefreq}.%j.out" \
                            run_ppo_tuning.sh "$lr" "$Kepoch" "$gamma" "$hidden" "$epsclip" "$updatefreq"
                    done
                done
            done
        done
    done
done