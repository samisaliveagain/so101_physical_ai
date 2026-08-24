# Gaussian world pipeline

This directory is deliberately independent of the proven dataset-world and
RViz launch path.

Stage 1 launches:

1. an empty Gazebo Harmonic world for simulation time and future physics; and
2. a local GPU Gaussian-splat viewport showing the Scaniverse PLY.

Gazebo Harmonic does not natively interpret Gaussian PLY or SPZ files as SDF
meshes. The two renderers therefore remain separate in stage 1. Later stages
will add a physics plane, calibrated cameras, ROS camera-pose synchronization,
the SO-101, and finally image/depth compositing.

Run from the repository root:

```bash
./gazebo/scripts/run_gaussian_world_stage1.sh
```

The initial orbit camera and splat transform are isolated in
`gazebo/gaussian/config/stage1.json`. Drag to orbit, use the mouse wheel to
zoom, and press `R` to reset the view.

The Scaniverse export contains non-finite and distant sentinel records that
break GPU depth sorting. The launcher automatically creates
`gazebo/gaussian/assets/Aachen-Mitte.cleaned.ply` with the preparation tool.
The original files under `3d_assets/gaussian_splats` are never modified.

## Native Gazebo direction

`gz-rendering` supports custom render engines, and Gazebo rendering-system
plugins can subscribe to pre-render/render/post-render events. A native Ogre2
Gaussian integration therefore remains possible, but it needs a purpose-built
GPU sort/rasterization implementation; a Gaussian PLY is not accepted by the
existing mesh loader. Stage 1 keeps the renderer separate so that physics and
visual loading can be validated before that larger integration.

The viewport uses the vendored MIT-licensed PlayCanvas 2.21.0 renderer. Its
license is stored in `gazebo/gaussian/vendor/PLAYCANVAS_LICENSE.txt`.
