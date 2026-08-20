# Rig formulas (YAM arms)

Planar arm kinematics in the radial–vertical (r, z) plane, from the i2rt
URDF constants: upper arm L1 = 0.264 m, forearm L2 = 0.245 m,
wrist-to-fingertip L3 = 0.101 m, shoulder pitch axis L0 = 0.114 m above
the mounting plane (= table level). Shared by all YAM rigs.

- Angles: A = π − j1 (upper arm), B = j2 − j1 (forearm), C = B + j3
  (fingertip axis; gripper-down is C = −π/2).
- Forward: fingertip ≈ (0, L0) + L1·(cos A, sin A) + L2·(cos B, sin B)
  + L3·(cos C, sin C).
- Inverse (target radius r, height z, approach angle C): let
  P = (r, z) − L3·(cos C, sin C) − (0, L0) and d = |P|; then
  A = atan2(P_z, P_r) + acos((d² + L1² − L2²) / (2·L1·d)) and
  B = atan2(P_z − L1·sin A, P_r − L1·cos A); recover j1 = π − A,
  j2 = B + j1, j3 = C − B. This is the elbow-up branch — the one the
  arms use in practice. Check reachability d ≤ L1 + L2 = 0.509 first.
- Accuracy: in the gripper-down working envelope (r ≈ 0.3–0.5 m) the
  dominant error is height: the formula reads ~7–10 cm high in z
  (measured ≈8–9 cm on both arms at rig-2, similar across arms but with
  ~1.5 cm of pose dependence across the envelope). The radial bias is
  small: commanded 2-inch lateral moves measured ~1.7–2 inches against
  a calibration grid. The z error is what you would get if L3 were
  ≈0.19 m rather than 0.101 m — an offset that, gripper-down, adds
  nothing to r, consistent with both observations. The bias is nearly
  constant for a given arm, so relative moves and the local Jacobian
  are accurate — and both the forward and inverse solutions inherit it,
  so correct all absolute heights by the rig's measured z offset (or
  measure it with one closed-gripper table touch if your rig's fact
  sheet lacks it), and re-touch near the target pose when centimetre
  precision matters. Outside that envelope (arm folded) the
  approximation degrades sharply.
