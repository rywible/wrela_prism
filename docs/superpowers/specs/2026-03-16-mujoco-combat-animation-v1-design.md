# Physics-Based Combat Animation v1 Design

## Goal

Train a physically-simulated humanoid to perform intent-conditioned sword combat against a target dummy, using pre-trained combat motor skills as a starting point. All training and visualization happens in Python via MimicKit/Newton. No Wrela engine integration in v1.

## Stack

| Layer | Tool | Role |
|-------|------|------|
| Framework | MimicKit (xbpeng/MimicKit, Apache-2.0) | Implements ASE, AMP, DeepMimic on modern backends |
| Physics | Newton (NVIDIA/DeepMind/Disney, Apache-2.0) | GPU-accelerated simulation via MuJoCo-Warp solver |
| Pre-trained model | ASE LLC (trained from scratch, sword-only) | Low-level combat motor skills trained on curated sword mocap |
| RL algorithm | PPO via MimicKit's built-in training stack | Custom PPO implementation in mimickit/learning/ |
| Visualization | Newton ViewerGL (`newton.viewer.ViewerGL`) | Interactive 3D rollout inspection (pyglet-based) |
| Training monitoring | TensorBoard | Reward curves, component breakdowns |
| Hardware | NVIDIA GPU (single card sufficient) | Required for Newton/MimicKit training |

### v1 Architecture: ASE LLC → AMP Fine-Tuning with Task Reward

**Phase 1 — ASE LLC (trained from scratch):**
- Trained from scratch on curated sword-only mocap via adversarial skill embeddings
- Cannot fine-tune from pre-trained sword-shield checkpoint because removing the shield body changes body count, breaking observation space compatibility
- Input: current body state + latent skill vector (64 dimensions)
- Output: joint torques for all actuators
- Learns to move a humanoid holding a sword naturally

**Phase 2 — AMP Fine-Tuning with Task Reward:**
- Initializes from the ASE LLC checkpoint (weight transfer, not frozen LLC)
- AMP discriminator maintains motion quality (50% of reward)
- Task reward drives combat behavior (50% of reward)
- This follows MimicKit's proven `task_location` / `task_steering` pattern rather than implementing custom frozen-LLC/HLC agent code

**Why not frozen LLC/HLC for v1:** MimicKit's built-in task environments (task_location, task_steering) use AMP with task reward — not a frozen LLC/HLC split. Implementing a true ASE HLC requires custom agent code (`hlc_combat_agent.py` + `hlc_combat_model.py`). For v1, we follow the proven AMP pattern. Upgrade to true frozen-LLC/HLC architecture in v2 if motion quality demands it.

## Humanoid Model

ASE's humanoid MJCF with the shield body removed. Sword-only, right hand.

**Sword:** One capsule handle + one capsule blade, welded to right hand. Realistic mass distribution (heavier handle, lighter blade). Exact geometry inherited from ASE model with shield geoms deleted.

**Off-hand (left):** Free. The LLC learns natural off-hand behavior from mocap priors.

**Model modification:** Remove all shield-related bodies, geoms, joints, and actuators from the ASE MJCF. This is a small XML edit — the shield is a distinct sub-tree attached to the `left_lower_arm` body.

## Environment

**Scene:**
- Flat ground plane
- Humanoid spawns at randomized distance (2-5m) and angle from target dummy
- Target dummy: static humanoid-shaped body (capsule tower) with contact zones for hit registration

**Episode:**
- Duration: 10 seconds (MimicKit default, adjustable)
- Terminates early if humanoid falls (torso height below threshold)
- Domain randomization per episode: slight variations in gravity (9.5-10.1 m/s^2), ground friction (0.8-1.2), body segment masses (+-5%)

**Observation space (HLC input):**
- Proprioception: joint positions, joint velocities, body orientation (quaternion), angular velocity, center-of-mass position/velocity
- Sword state: blade tip position, blade tip velocity, blade orientation
- Target: relative position and orientation of dummy (in character's local frame)
- Intent signal: one-hot encoding of {light_attack, heavy_attack, approach, idle}

**Action space:**
- HLC: continuous latent skill vector (64 dimensions, normalized)
- LLC: continuous joint torques (all actuators), output from LLC given skill vector + body state

## Training Pipeline

### Phase 1: Train LLC from Scratch (ASE, ~1-3 hours on H200)

LLC must be trained from scratch — not fine-tuned from the pre-trained sword-shield checkpoint. Removing the shield body decreases body count by 1, which changes observation space dimensions and makes all pre-trained checkpoints incompatible. DOF count (34) is unchanged because the shield has no joints.

1. Remove shield from MJCF model (delete `<body name="shield">` sub-tree from `left_lower_arm`)
2. Curate sword-only motion clips from Reallusion dataset:
   - Keep: all sword-primary attacks, combos, kills, locomotion, turns, standoffs, idles, taunts, parries (~75 clips)
   - Remove: `ShieldCharge`, `ShieldSwipe01/02`, `Shield_Block*` (6 variants), `Taunt_ShieldKnock`, `Counter_Atk05` (~10 clips)
3. Train ASE LLC with 16K parallel envs on H200 (curated dataset, Newton backend)
4. Validate: sample random skill vectors in Newton ViewerGL — motions should look human, sword-hand natural, off-hand relaxed

**Done when:** The latent skill space contains recognizable sword-only movement — locomotion, swings, stances, transitions — without shield artifacts.

### Phase 2: AMP Fine-Tuning with Combat Task Reward (~30-60 min on H200)

Initialize from LLC checkpoint. AMP discriminator (locomotion-only clips) maintains motion quality. Task reward drives intent-conditioned combat behavior.

**Reward function:**

| Component | Weight | Description |
|-----------|--------|-------------|
| Hit reward | 0.4 | Blade-dummy contact, weighted by impact momentum (mass x velocity at contact point) |
| Intent compliance | 0.2 | Light attack: blade velocity > threshold, short duration. Heavy attack: high momentum, longer wind-up. Measured from actual blade kinematics. |
| Approach reward | 0.1 | Reduce distance to target when approach intent active. Small shaping reward. |
| Balance | 0.2 | Torso upright, center of mass over support polygon. Continuous reward, not just termination condition. |
| Energy penalty | -0.1 | Penalize horizontal root speed to discourage excessive movement |

**Intent compliance details:**
- Light attack: reward peaks when blade velocity > V_light (tuned), swing arc < 120 degrees, recovery time < T_light
- Heavy attack: reward peaks when blade momentum > M_heavy (tuned), swing arc > 150 degrees, visible wind-up phase
- Approach: reward for reducing distance, zero attack reward
- Idle: reward for maintaining stable stance, zero movement/attack reward

These thresholds are tuning parameters discovered during training iteration.

### Phase 3: Polish and Evaluate (~1-2 days including iteration)

1. Run rollouts in Newton ViewerGL with scripted intent sequences (`scripts/eval_combat.py`):
   - Approach (3s) → light attack (2s) → idle (2s) → heavy attack (2s) → approach (2s) → light attack (1s)
   - Verify each intent produces visibly distinct behavior
2. Inspect failure modes: falls, weird off-hand, jittery transitions between intents
3. Adjust reward weights in YAML → rsync to RunPod → retrain Phase 2 (~30-60 min per round) → rsync checkpoint back → evaluate locally. Expect 2-3 rounds.
4. Save best checkpoint per evaluation metric

**Total estimated GPU time: ~3-7 hours on H200 SXM (~$12-28)**

## Success Criteria

v1 is complete when all of the following are observable in Newton ViewerGL:

1. Humanoid holding a sword in the right hand walks toward a target dummy with natural human-looking gait
2. "Light attack" intent → fast, compact sword strike making blade contact with the dummy
3. "Heavy attack" intent → visibly different motion: more wind-up, wider arc, more body commitment, longer recovery
4. "Approach" intent → closes distance without attacking
5. "Idle" intent → maintains a natural ready stance
6. Balance maintained throughout — natural recovery after swings
7. Off-hand (left) looks natural, not rigid or flailing
8. Motion quality is recognizably human (AMP priors active, no optimization artifacts)

## What v1 is NOT

- No reactive opponent (dummy is static)
- No damage model, armor, or vulnerability zones
- No multiple weapon types
- No blade expressions or style conditioning
- No variable physics (gravity/friction changes are training randomization only, not gameplay)
- No cinematic/superhuman moves
- No Wrela engine integration
- No ONNX export or Rust FFI
- No retargeting to the Wrela procedural humanoid mesh

## Deliverables

1. MimicKit project configured for Newton backend with sword-only humanoid MJCF
2. Retrained LLC checkpoint (sword-only skill space)
3. Trained HLC checkpoint (intent-conditioned combat)
4. Visualization scripts for Newton ViewerGL rollouts with configurable intent sequences
5. TensorBoard training logs with per-component reward breakdown
6. This design document

## Future Versions (Not Designed Here)

- **v2:** True frozen-LLC/HLC architecture (if motion quality demands it), reactive opponent (self-play or scripted), physics-based damage model (momentum-derived), weapon variety (rapier, greatsword, hammer with different mass distributions)
- **v3:** ONNX export of LLC+HLC, MuJoCo C FFI embedded in Rust, retarget joint transforms to Wrela's 22-bone procedural humanoid, render through visbuf pipeline
- **v4:** Blade expressions as style conditioning vectors on HLC, variable runtime physics (gravity, friction, injury), cinematic actuator scaling for superhuman combat, AMP style priors per expression

## Dependencies & Setup

**Python environment:**
- Python 3.10+
- MimicKit: clone from github.com/xbpeng/MimicKit, then `pip install -r requirements.txt`. Not on PyPI — must be cloned. Pre-trained models and assets require a separate download (SharePoint link in MimicKit README) and extraction into `data/`.
- Newton: `pip install newton` (v1.0.0, released March 10, 2026)
- torch, tensorboard, pyglet (pulled by MimicKit requirements)

**Data:**
- Reallusion sword-and-shield mocap clips (bundled with MimicKit assets, curated to ~75 sword-only clips)
- Pre-trained sword-shield checkpoints are NOT used — shield removal breaks checkpoint compatibility (body count change). LLC is trained from scratch.
- AMASS motion data is not needed for v1 — Reallusion clips provide sufficient locomotion coverage

**Licensing caveat:** Reallusion motion data is licensed for noncommercial use only. Commercial deployment (v3+) would require re-training on permissively licensed motion data (AMASS is CC-BY) or purchasing a Reallusion commercial license.

**Physics backend note:** MimicKit's sword-and-shield configuration defaults to IsaacGym (`data/engines/isaac_gym_engine.yaml`). Switching to Newton requires changing to `data/engines/newton_engine.yaml`. Newton is v1.0.0 with a rapidly evolving API — verify sword-and-shield environment compatibility on Newton before committing. IsaacGym Preview 4 is a proven fallback if Newton has issues.

**Hardware:**
- NVIDIA GPU with CUDA support (Maxwell+, CUDA 12)
- Single GPU sufficient (A100, RTX 4090, or similar)
- Linux required for GPU-accelerated training (Newton and IsaacGym are Linux-only for GPU)
- Development machine is macOS (M4 Air) — training must happen on a separate Linux GPU machine or cloud instance. Workflow: develop/configure locally, push to remote, train on GPU, pull checkpoints back for visualization

**Remote training workflow:**
- SSH into Linux GPU machine or spin up cloud instance (Lambda, Vast.ai, RunPod)
- Clone MimicKit, download assets, configure environment
- Run training with TensorBoard logging
- Copy checkpoints back to local machine for visualization (Newton ViewerGL works on macOS CPU for inference/replay)

## Runtime Architecture (Preview — v3 Scope)

For context on where this is headed, the eventual in-game architecture:

```
Player input → Intent signal ──┐
                                ├──→ HLC (ONNX, ~20Hz) → skill vector
Game state → Observation ───────┘                            │
                                                             ▼
                                         LLC (ONNX, ~60Hz) → joint torques
                                                             │
                                                             ▼
                                         MuJoCo C (cross-platform) → joint transforms
                                                             │
                                                             ▼
                                         Retarget → Wrela 22-bone skeleton → GPU skinning → visbuf render
```

MuJoCo C library is cross-platform (macOS, Windows, Linux, ARM, x86). ONNX inference via `ort` crate runs on CPU. No NVIDIA dependency in the shipped game.
