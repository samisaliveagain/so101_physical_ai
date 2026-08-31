# VLA-JEPA HTTP shadow inference

The compute node runs `hpc_vla_jepa_server.py`; the robot laptop runs `so101_http_shadow_client.py`. The client has no actuation implementation and only performs read-only motor-state access.

## HPC allocation and server

```bash
ssh rwth-gpu-sj
salloc --partition=c23g --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=01:00:00
srun --pty bash -l
source "$HOME/lerobot/.venv/bin/activate"
export HF_HOME=/hpcwork/dl125352/hf
export HF_HUB_CACHE=/hpcwork/dl125352/hf/hub
export TRANSFORMERS_CACHE=/hpcwork/dl125352/hf/hub
cd /path/to/so101_physical_ai
# Enter the same high-entropy token on the robot laptop; do not put it in shell history.
read -rsp 'Inference token: ' SO101_INFERENCE_TOKEN && echo
export SO101_INFERENCE_TOKEN
python deployment/hpc_vla_jepa_server.py \
  --policy-path /hpcwork/dl125352/train/so101_wm_vlajepa/checkpoints/030000/pretrained_model \
  --dataset-root /path/to/so101_wm_dataset \
  --host 0.0.0.0 \
  --port 18080 \
  --auth-token-env SO101_INFERENCE_TOKEN \
  --max-observation-age-ms 10000
```

Record the compute hostname from `hostname`, then create the tunnel on the robot laptop:

```bash
ssh -N -L 18080:COMPUTE_HOSTNAME:18080 rwth-gpu-sj
```

## Shadow request

In another laptop terminal:

```bash
cd /path/to/so101_physical_ai
read -rsp 'Inference token: ' SO101_INFERENCE_TOKEN && echo
export SO101_INFERENCE_TOKEN
curl --fail -H "Authorization: Bearer ${SO101_INFERENCE_TOKEN}" http://127.0.0.1:18080/health
"$HOME/lerobot/.venv/bin/python" deployment/so101_http_shadow_client.py
```

This returns a full action chunk and reports the first action's delta from the current joint state. It does not call LeRobot's `send_action`.

## Validated pipeline run — 2026-08-02

- Slurm allocation: job `2648909`, H100 node `n23g0018`, automatically assigned to `c23g_low` because the account was over its non-project core-hour quota.
- Health check through the login node and local SSH tunnel: HTTP 200, CUDA ready, actuation disabled.
- Live input: two 640×480 JPEG camera frames, exact task instruction, and six read-only follower positions.
- Response: 7×6 physical-unit joint-space action chunk.
- Cold request: 3211.32 ms server inference, 3675.90 ms HTTP round trip.
- Warm request: 734.05 ms server inference, 885.04 ms HTTP round trip.
- Robot commands sent: **zero**.

At 30 FPS, seven action steps cover about 233 ms. The measured 734 ms warm inference cannot refill a 30 FPS action queue in time, so continuous actuation is not approved yet. Continue with shadow-mode latency/trajectory validation or optimize the inference path before adding any robot command mechanism.

## Deliberately slow, bounded rollout

`so101_http_slow_rollout.py` is separate from the shadow client and is locked by two explicit arming
confirmations. It accepts only a loopback URL, so inference traffic must traverse the SSH tunnel. Before
every movement it checks authentication, request/response identity, server-instance continuity and
response age. Non-finite or malformed output remains a hard failure. It executes only the first action:
that target is clipped to the demonstrated position envelope and dataset 99th-percentile displacement,
while violations in the unused six future actions are logged as warnings. The accepted target follows a
locally generated minimum-jerk trajectory with velocity and acceleration limits.

Keep a hand on the physical power cut/e-stop and begin with exactly one cycle:

```bash
"$HOME/lerobot/.venv/bin/python" deployment/so101_http_slow_rollout.py \
  --url http://127.0.0.1:18080 \
  --cycles 1 \
  --arm SAFE_SLOW_ROLLOUT
```

The process will ask for `SAFE_SLOW_ROLLOUT` again in its interactive terminal. Never alias or automate
the arming phrase. On HTTP timeout, stale/mismatched output, server restart, Ctrl-C, joint-limit violation,
control-loop overrun or action clipping, it commands the current measured position and stops. It leaves
torque enabled so the arm holds instead of falling; use the physical power cut if holding is unsafe.

### Sequential closed-loop behavior

The client runs each requested cycle in this order:

1. Read the current encoders and command that same position as a hold target.
2. Capture fresh left/FPV frames and the held joint state.
3. Send exactly one HTTP request. No model action is executed while the request is pending; the motor
   controller continues holding the observation pose.
4. Match the response to the request and server instance, reject stale/malformed data, and sanitize only
   the first action.
5. Interpolate toward the target locally, then repeatedly command the final target while reading encoders.
6. Require every meaningfully commanded joint to move at least 25% of its requested displacement and
   settle within the configured tolerance for three consecutive samples.
7. Hold the reached measured pose, capture a new observation, and only then begin the next request.

The tracking deadline is four seconds after interpolation. Failure to demonstrate real encoder movement
causes a hold-and-stop error; command transmission alone is never reported as movement success. Configure
feedback thresholds under `[safety]` in `so101_hardware.toml`.

For a supervised multi-step test, increase the cycle count deliberately:

```bash
"$HOME/lerobot/.venv/bin/python" deployment/so101_http_slow_rollout.py \
  --url http://127.0.0.1:18080 \
  --cycles 3 \
  --arm SAFE_SLOW_ROLLOUT
```

The process remains synchronous: there is never more than one inference request or one action target in
flight. Start with one cycle after any code, camera, calibration, checkpoint or workspace change.

Audit events (but no camera images or bearer token) are appended to `eval/out/slow_rollout.jsonl`.
The client permits at most ten cycles per invocation; keep `--cycles 1` until multiple supervised trials
are clean. This software cannot replace an independent physical e-stop.

## Failure behavior

- A Slurm allocation ending, the model process crashing, or the SSH tunnel dropping causes the pending
  HTTP request to fail; no action is issued for that observation and the local client holds position.
- The server rejects concurrent requests instead of queueing stale observations.
- A non-loopback server bind is refused unless bearer authentication is configured. Keep the compute-node
  port firewalled even with authentication.
- Each response includes the client request ID and a random server-instance ID. A response from another
  request or a restarted server is rejected.
