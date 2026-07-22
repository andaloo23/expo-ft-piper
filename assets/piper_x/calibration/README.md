# Piper calibration assets

Placeholder for per-rig calibration that the actor consumes:

- camera extrinsics (static D455 / wrist D405 vs. left_base_link), once
  measured — needed only for action-space visualization, not for control.
- master→follower base transform if the leader arms are not mounted
  identically to the followers (teleop/master_arm.py defaults to identity).

Gravity/friction/deflection calibrations for the MIT-mode ROS stack live in
the roboorchard-dev tree (deflection_calibrations.json) and are deliberately
NOT used here: this port drives firmware position mode only.
