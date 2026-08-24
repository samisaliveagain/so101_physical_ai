# SO-101 hardware setup

The detected hardware mapping is stored in [`so101_hardware.toml`](so101_hardware.toml). It uses persistent Linux device paths rather than `/dev/videoN` numbers, because numeric camera indices can change after reconnecting USB devices.

## Labels

| Label | Physical device | Dataset input | VLA-JEPA policy input |
|---|---|---|---|
| `left` | UGREEN external workspace camera | `observation.images.left` | `observation.images.exterior_1_left` |
| `fpv` | Innomaker robot-mounted camera | `observation.images.fpv` | `observation.images.exterior_2_left` |

Both cameras are configured for 640×480 RGB input at 30 FPS using MJPEG capture. The follower uses the ID `my_awesome_follower_arm` and the copied six-motor calibration file under the Hugging Face LeRobot cache.

Re-run the non-actuating validation at any time with:

```bash
cd /home/shubhamnagar/coding/so101_physical_ai
/home/shubhamnagar/lerobot/.venv/bin/python scripts/check_so101_hardware.py --read-motors
```

This checker opens both cameras, pings motor IDs, compares cached calibration with the motor registers, and reads present positions. It never writes goal positions, torque, PID, or calibration registers.

## Current validation state

- The UGREEN `left` image clearly shows the robot workspace.
- The Innomaker `fpv` device streams successfully, but its lens is currently blocked by or pressed against a perforated surface. Uncover and aim it to reproduce the training view before inference.
- The calibration JSON exists and contains all six expected SO-101 motors and required calibration fields.
- No joint command, torque command, or calibration write was issued during discovery.

## Equivalent LeRobot robot arguments

Run LeRobot from `/home/shubhamnagar/lerobot/.venv/bin`. The hardware portion of a command is:

```bash
--robot.type=so101_follower \
--robot.port=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A68011297-if00 \
--robot.id=my_awesome_follower_arm \
--robot.cameras='{
  left: {type: opencv, index_or_path: /dev/v4l/by-id/usb-UGREEN_Camera_UGREEN_Camera_SN0001-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG},
  fpv: {type: opencv, index_or_path: /dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-720P_SN0001-video-index0, width: 640, height: 480, fps: 30, fourcc: MJPG}
}'
```

Do not run `lerobot-calibrate` merely to test this setup: calibration can write motor settings and requires hands-on positioning. The existing calibration should first be checked by a read-only observation client, with the arm supported and the emergency-stop method ready.

## Before an HTTP policy rollout

1. Uncover and aim the `fpv` camera to match its training role.
2. Verify fresh images from both named camera paths.
3. Connect to the follower and read joint positions without sending targets.
4. Start the HPC policy service and SSH tunnel.
5. Run the local client in shadow mode, logging predicted actions but sending none.
6. Only enable actuation after checking units, joint ordering, calibration, latency, and conservative per-joint action limits.

The implemented actuation path is [`so101_http_slow_rollout.py`](so101_http_slow_rollout.py). Its initial
safety envelope is in `[safety]` inside [`so101_hardware.toml`](so101_hardware.toml). The position bounds
and model/chunk displacement bounds come from the demonstrated dataset. The initial velocity and
acceleration limits are deliberately conservative operational settings and must not be increased until
supervised one-cycle trials have been reviewed.

Before every armed run:

1. Support the arm and keep an independent physical power cut/e-stop in reach.
2. Run `scripts/check_so101_hardware.py --read-motors` and a shadow prediction.
3. Confirm the FPV lens is uncovered and both live views match their labels.
4. Confirm the SSH tunnel terminates at the hostname of the active Slurm allocation.
5. Run one armed cycle and inspect `eval/out/slow_rollout.jsonl` before increasing the cycle count.

The rollout client holds the measured pose throughout each HPC request. After receiving an action, it
does not advance to the next observation until encoder feedback proves that meaningfully commanded joints
moved and settled near the target. A motor stall or insufficient tracking stops the loop and holds the
measured position.
