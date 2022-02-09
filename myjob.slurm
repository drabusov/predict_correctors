#!/bin/bash
##SBATCH --job-name=correctors
##SBATCH --array=1-100
##SBATCH -e /lustre/nyx/bhs/drabusov/cluster-testing/2022-02-09/predict_correctors/output/stderror.%A.%a.log
##SBATCH -o /lustre/nyx/bhs/drabusov/cluster-testing/2022-02-09//predict_correctors/output/stdout.%A.%a.log
##SBATCH --time=01:30:00

export SLURM_ARRAY_TASK_ID=102

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
mkdir /tmp/drabusov/
mkdir /tmp/drabusov/predict_correctors_$SLURM_ARRAY_TASK_ID

rsync -avP /lustre/bhs/drabusov/cluster-testing/2022-02-09/predict_correctors/correction.py /tmp/drabusov/predict_correctors_$SLURM_ARRAY_TASK_ID
rsync -avP /lustre/bhs/drabusov/cluster-testing/2022-02-09/predict_correctors/sis100cold.seq /tmp/drabusov/predict_correctors_$SLURM_ARRAY_TASK_ID
rsync -avP /lustre/bhs/drabusov/cluster-testing/2022-02-09/predict_correctors/task.$SLURM_ARRAY_TASK_ID.sh /tmp/drabusov/predict_correctors_$SLURM_ARRAY_TASK_ID
rsync -avP /lustre/bhs/drabusov/cluster-testing/2022-02-09/predict_correctors/Optics-YEH-Jan19-from-Elegant.str /tmp/drabusov/predict_correctors_$SLURM_ARRAY_TASK_ID
ls

cd /tmp/drabusov/predict_correctors_$SLURM_ARRAY_TASK_ID

singularity exec --no-home --bind /cvmfs /cvmfs/vae.gsi.de/centos7/containers/user_container-develop.sif /tmp/drabusov/predict_correctors_$SLURM_ARRAY_TASK_ID/task.$SLURM_ARRAY_TASK_ID.sh -i

ls -l
echo -e "\n\n********I have sent the jobs to the working nodes!"
rsync -avP /tmp/drabusov/predict_correctors_$SLURM_ARRAY_TASK_ID/results/* /lustre/nyx/bhs/drabusov/cluster-testing/2022-02-09/predict_correctors/results/
