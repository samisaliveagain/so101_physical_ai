# Headless SO-101 Gazebo on RWTH Apptainer

The image contains ROS 2 Jazzy, Gazebo Harmonic, ros2_control, both camera bridges, and LeRobot
0.5.1. The project and generated datasets remain bind-mounted under `/hpcwork`, so rebuilding the
image never touches recorded data.

```bash
ssh rwth-gpu
bash /hpcwork/$USER/so101_gazebo/project/apptainer/build_gazebo_rwth.sh
sbatch /hpcwork/$USER/so101_gazebo/project/apptainer/collect_headless_gazebo_rwth.sbatch
```

The default job records 10 randomized episodes. Override without editing the script:

```bash
EPISODES=50 SEED=2026 sbatch --export=ALL \
  /hpcwork/$USER/so101_gazebo/project/apptainer/collect_headless_gazebo_rwth.sbatch
```

Gazebo uses EGL headless rendering with `--nv` by default. If Ogre cannot create an OpenGL context
on the allocated H100, resubmit with `USE_NVIDIA=0 SOFTWARE_RENDERING=1`; Mesa/llvmpipe will render
the two cameras on CPU.

For a CPU-only 100-episode dataset, first submit the one-episode smoke test,
then make the collection depend on it:

```bash
SMOKE_JOB=$(sbatch --parsable collect_headless_gazebo_cpu_smoke_rwth.sbatch)
sbatch --dependency="afterok:$SMOKE_JOB" collect_headless_gazebo_100ep_rwth.sbatch
```

This requests 8 CPUs and 16 GB on `c23ms` with no GPU. A one-episode smoke job
runs first. The collection starts only after the smoke succeeds and writes one
finalized 100-episode LeRobot dataset under `$HPC_ROOT/datasets`.
