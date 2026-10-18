#!/usr/bin/env bash

#SBATCH --job-name=F_t1
#SBATCH --output=output      ## need not to change
#SBATCH --ntasks-per-node=32  ## need not to change
#SBATCH --partition=besq
#SBATCH -N 1
#SBATCH --time=7200:00:00

source $HOME/OpenFOAM/OpenFOAM-6/etc/bashrc WM_LABEL_SIZE=64 WM_MPLIB=OPENMPI FOAMY_HEX_MESH=yes
module load gcc
cd ../

mpirun -np 32 rhoReactingFoam -parallel  >./run/openfoam_out
#srun ./s3d.x >s3d_out
#srun mpirun -np 32 ./s3d.x -parallel >s3d_out


