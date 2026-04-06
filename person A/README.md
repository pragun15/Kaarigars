# Rescue Robot OpenEnv (Skeleton)

This repository now includes a deterministic, typed, OpenEnv-style environment shell for earthquake search and rescue.

## Implemented

- Project and package skeleton under `rescue_env/`
- Typed models for actions, observations, state, robot, and victims
- Environment API shell with `reset()`, `step()`, and `state()`
- Deterministic seeding by episode
- World simulation basics:
  - map generation
  - debris enrichment
  - hazard generation
  - victim generation with priority scores
  - movement and collision physics
- Robot execution path:
  - action controller
  - battery drain model
  - LiDAR, thermal, gas, acoustic, IMU hooks

## Notes

- No virtual environment was created or modified.
- This code is ready to be wired into your existing virtual environment once it is available.
