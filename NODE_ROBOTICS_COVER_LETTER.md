Shubham Nagar  
[Street and house number]  
[Postal code, city]  
[Email] · [Phone] · [LinkedIn/GitHub]

1 August 2026

Hiring Team  
NODE Robotics GmbH  
ARENA2036  
Pfaffenwaldring 19  
70569 Stuttgart

**Application for Robotics Simulation Engineer**

Dear NODE Robotics Hiring Team,

I am applying for the Robotics Simulation Engineer position because NODE’s goal of making mobile
robots easy to integrate, operate, and adapt closely matches how I approach robotics engineering:
build realistic systems, validate them systematically, and make the path from an experiment to
reliable hardware operation reproducible. I am particularly drawn to the opportunity to own
simulation infrastructure that supports NODE.OS, a modular platform combining localization, hybrid
navigation, and fleet management for autonomous mobile robots operating in real intralogistics
environments.

In my recent physical-AI project, I built an end-to-end training and validation workflow for an
SO-101 robot using Python, PyTorch, Hugging Face LeRobot, and HPC infrastructure. I collected and
managed a two-camera real-world dataset containing 177 episodes and 355,884 frames, then fine-tuned a
3.1-billion-parameter VLA-JEPA policy combining Qwen3-VL, V-JEPA2, a world-model predictor, and a
flow-matching DiT action head. I used PEFT/LoRA to train 334 million parameters on an H100 through
reproducible Slurm jobs, preserved the camera and joint-space interfaces needed for deployment, and
published the resulting model and processor artifacts on the Hugging Face Hub.

Just as importantly, I treated validation as an engineering system rather than relying on training
loss. I developed a repeatable regression-evaluation pipeline covering dataset versioning, camera
mapping, preprocessing, action denormalization, trajectory capture, error analysis, and failure-safe
deployment gates. The final held-out run evaluated 52,927 frames across 26 episodes and produced
per-joint MSE, RMSE, and MAE, trajectory artifacts, and gripper metrics. This experience translates
directly to simulation-based regression testing, CI/CD, observability, and closing the sim-to-real gap
between digital twins and real hardware.

NODE’s simulation roadmap is especially compelling to me: high-fidelity intralogistics environments,
robot models in URDF and SDF, LiDAR, odometry and cameras, Gazebo (Ignition/Harmonic), NVIDIA Isaac
Sim, custom simulation tooling, and automated GitLab CI/CD pipelines for software testing of
navigation and fleet management. I would bring strong Python and applied AI/robotics experience, careful
experimental design, and a quality-first approach while contributing within a ROS 2 and C++ robotics
stack and collaborating with Robotics Core engineers. I am motivated by the systems-level challenges behind mobile robot kinematics, sensor models,
localization, navigation, warehouse automation, multi-robot simulation, and scalable fleet behavior.

NODE’s combination of production impact and engineering culture is equally attractive. A platform
already operating on more than 2,000 mobile robots requires robustness, modularity, and disciplined
testing, while the greenfield scope of this role still offers room to improve tooling and coverage. I
would value working close to the physical robots at ARENA2036 and contributing through the same
collaborative improvement, knowledge sharing, customer empathy, and ownership that NODE emphasizes.

I would welcome the opportunity to discuss how my experience building reproducible robotics-learning
and validation systems can support NODE’s simulation infrastructure and help every NODE.OS release
reach real warehouses with greater confidence.

Sincerely,

Shubham Nagar
