#!/bin/bash
#PBS -N haar4_qcbm_102
#PBS -q gpuq
#PBS -l select=1:ncpus=8
#PBS -l walltime=72:00:00
##PBS -e error_$(PBS_JOBID).log
##PBS -o out_$(PBS_JOBID).log
#ncores=`cat $PBS_NODEFILE|wc -l`
#source /apps/psxe2018u4/compilers_and_libraries/linux/bin/compilervars.sh intel64
#source /apps/psxe2018u4/compilers_and_libraries/linux/mpi/bin64/mpivars.sh intel64
#source /apps/psxe2018u4/compilers_and_libraries/linux/mkl/bin/mklvars.sh intel64

cd $PBS_O_WORKDIR
source activate qiskitml

datafile="nqubits_04_haar_seed102.py"
latentsize=0
annealing="linear"
output_root_directory="./log-files"
enc_lr=0.003
dec_lr=0.009
nn_type="quantum-classical"
nepochs=50
beta=1
batchsize=64
patience=7
featuremap="ZZ"

# create new directory
output_directory="$output_root_directory/data_$datafile-enc_lr_$enc_lr-dec_lr_$dec_lr-fm_$featuremap-latentsize_$latentsize-annealing_$annealing-nntype_$nn_type"
mkdir -p $output_directory

# main.py (name of datafile) (featuremap) (patience) (encoder lr) (decoder lr) (batchsize) (beta) (latentsize) (no epochs) (type of annealing) (nn type) (output directory)
python3 "../Sourcecode/main.py" "../haar-data/$datafile" $featuremap $patience $enc_lr $dec_lr $batchsize $beta $latentsize $nepochs $annealing $nn_type $output_directory

#mpirun    --mca btl self,vader -machinefile $PBS_NODEFILE -np $ncores ./a.out
#mpirun --mca mpi_leave_pinned 1 --bind-to none --report-bindings --mca btl self,vader -machinefile $PBS_NODEFILE -np $ncores ./a.out
#mpirun --mca mpi_leave_pinned 1 --bind-to none --mca btl self,vader -machinefile $PBS_NODEFILE -np $ncores ./a.out

#mpirun  -machinefile $PBS_NODEFILE -np $ncores ./a.out
#sleep 60
