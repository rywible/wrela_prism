# Player-Controlled Combat Policy Design

## Goal

Train a physics-based combat policy for the player's main character that responds to gamepad input — left stick controls movement, buttons trigger commitment-based combat actions. All motion is emergent from physics, no canned animations. The same Round 1 checkpoint branches into an NPC ("Shadow") and the player character, with Shadow serving as a sparring partner for training defensive moves.

## Relationship to v1 Spec

This spec supersedes the v1 combat animation design (2026-03-16) in several ways:

- **Humanoid model:** v1 used sword-only (shield removed). During training, we switched to sword+shield because pre-trained models (ASE LLC, AMP steering) already existed for that character, saving significant training time. The sword+shield character is now the canonical model.
- **Architecture:** v1 planned ASE LLC/HLC (64-dim latent skill vector). This spec uses a single monolithic AMP policy that directly outputs joint torques. The LLC/HLC split proved unnecessary — the AMP discriminator provides sufficient motion quality, and a monolithic policy allows fluid locomotion-during-combat that a switching architecture cannot. LLC/HLC may be revisited if motion quality plateaus.
- **AMP weighting:** v1 used 0.5/0.5 task/disc. Training experiments showed 0.4/0.6 (more disc weight) produces more stable locomotion. This spec uses 0.4/0.6.
- **Scope:** v1 targeted NPC-only combat. This spec adds player-controlled character as the primary goal, with the NPC ("Shadow") as secondary.

## Core Design Decisions

1. **Commitment-based attacks** — player chooses when and what, physics determines how the swing plays out
2. **Stick direction influences attacks** — movement velocity at the moment of attack press colors the swing (moving left + attack = sweeping cut, moving forward + attack = lunging thrust)
3. **Camera-relative controls** — standard third-person; camera heading transforms stick input to world-space velocity before it enters the policy
4. **Trained recovery over i-frames** — hits apply real physics force, the policy has a trained recovery skill that rebalances. No hyper-armor, no invincibility windows. Guard rails are in the policy, not the game rules
5. **Block button with timing-based parry** — one player input, game system determines intent. Block within ~200ms of incoming hit = parry intent, otherwise = block intent
6. **Single monolithic policy with phased training (Approach 3)** — one network handles locomotion + combat. Locomotion reward is always active so the stick always works, even mid-attack. Combat actions layer on top

## Observation Space (~185 dims)

| Input | Dims | Source |
|-------|------|--------|
| Proprioception (joints, velocities, body orientation) | ~158 | MuJoCo body state (sword+shield character) |
| Sword state (position, velocity, orientation) | 9 | Sword body in character skeleton |
| Player stick input (camera-relative velocity target) | 3 | `tar_vel_x, tar_vel_y, tar_speed` — camera heading applied before policy |
| Action intent (one-hot) | 6 | `[idle, light_attack, heavy_attack, dodge, block, parry]` |
| Enemy relative position | 3 | Enemy position in character's local heading frame |
| Enemy sword state (for parry/block timing) | 6 | Enemy sword position + velocity in local frame |

Camera heading transforms stick input to world-space before it enters the policy. The policy never sees raw stick values or camera angles.

Parry and block are separate intents at the policy level. The game system decides which to send based on timing relative to incoming attacks. The player presses one button.

Enemy state is included so the policy can orient toward threats and react to incoming swings. When training against a static dummy, enemy sword state is zero.

## Action Space

31 continuous joint torques — direct control of all actuators. Same as the current Round 1 model.

## Reward Structure

50% always-active base (guarantees stick responsiveness + balance) + 50% intent-conditioned (teaches combat behaviors).

### Always-Active Base

| Component | Weight | Logic |
|-----------|--------|-------|
| Velocity tracking | 0.25 | `exp(-scale * \|tar_vel - actual_vel\|^2)` — match player stick input |
| Balance | 0.20 | `clamp(char_up_z, 0, 1)` — stay upright |
| Facing target | 0.05 | Dot product of facing direction vs enemy direction |

### Intent-Conditioned (active only for matching intent)

| Component | Weight | Logic |
|-----------|--------|-------|
| Light attack | 0.10 | `exp(-2 * sword_to_dummy_dist) * clamp(sword_speed / threshold, 0, 1)` — proximity is exponential decay of sword-to-dummy distance, multiplied by normalized sword speed for fast compact strikes |
| Heavy attack | 0.10 | `exp(-2 * sword_to_dummy_dist) * clamp(sword_speed / (threshold * 1.5), 0, 1)` — same proximity metric, higher speed threshold rewards more momentum |
| Dodge | 0.10 | `exp(-scale * \|dodge_vel - actual_vel\|^2)` — dodge_vel is `tar_vel direction * dodge_speed` where `dodge_speed` (e.g. 3.0 m/s) is much higher than normal locomotion speed, differentiating dodge from regular velocity tracking |
| Block | 0.10 | Sword positioned between enemy and torso, low sword velocity (held steady) |
| Parry | 0.10 | Sword moves outward to deflect — high speed, specific angle relative to incoming |

All components are [0, 1] bounded. No negative rewards.

**AMP objective blending:** The total task reward (base + intent components, structured 50/50 internally) receives 0.4 weight in the AMP objective. The discriminator reward receives 0.6 weight. So the final reward is `0.4 * task_reward + 0.6 * disc_reward`. The discriminator ensures all actions look like natural human motion.

## Player Input Mapping

| Input | Policy effect |
|-------|---------------|
| Left stick direction + magnitude | `tar_vel` (transformed by camera heading to world space). Full stick = run, light tilt = walk, neutral = stop |
| Right stick | Camera only (game system, not policy input) |
| X / Square | Intent → `light_attack` |
| Y / Triangle | Intent → `heavy_attack` |
| B / Circle | Intent → `dodge` |
| LB / L1 | Game checks timing: within ~200ms of incoming hit → intent `parry`, otherwise → intent `block` |
| No button | Intent → `idle` |

Movement is always active. During attacks, the stick continues to feed `tar_vel` to the policy. The policy learned to balance locomotion with combat commitment during training.

## Training Curriculum

All training warm-starts from the Round 1 checkpoint (sword+shield character, AMP, mean episode length ~273 steps / 9.1 seconds before falling, trained on RunPod H200).

**Episode termination:** Episodes terminate early if the character falls (torso height below threshold, inherited from `enable_early_termination: True` in env config). Maximum episode length is 20 seconds (600 steps at 30Hz).

### Branch A: Shadow (NPC) — train first

Autonomous combat agent. Intents cycle automatically. Approaches dummy and attacks. This is the simpler training task and produces the sparring partner needed for the main character's defensive training.

- Env: `task_combat_env.py` with `enable_intent_reward: true`
- Autonomous intent cycling (existing implementation)
- ~400 iterations from Round 1 checkpoint
- Estimated cost: ~$3.60

### Branch B: Main Character — Phase 2 (attack)

Player-controlled locomotion + offensive combat against static dummy.

- New env: `task_player_combat_env.py`
- Velocity tracking replaces autonomous approach
- Randomized stick inputs during training: correlated random walk with momentum (direction changes every 2-5s, magnitude varies smoothly) to simulate realistic player behavior, not uniform noise
- Intents: idle, light_attack, heavy_attack only (dodge/block/parry dims zeroed in observation)
- ~400 iterations from Round 1 checkpoint
- Estimated cost: ~$3.60

### Branch B: Main Character — Phase 3 (defense)

Add defensive actions, trained against frozen Shadow as opponent.

- Same env with dodge, block, parry intents enabled
- Frozen Shadow policy drives the opponent character
- Two characters in the scene: player character (training) + Shadow (frozen, attacking)
- Training env generates parry vs block intent based on Shadow's sword proximity: when Shadow's sword is within time-to-impact threshold (~200ms at current velocity), env sends `parry` intent; otherwise `block` when defend button is simulated
- ~400 iterations from Phase 2 checkpoint
- Known limitation: Shadow's attack repertoire is limited to patterns learned during its 400 iterations. Mitigation: add action noise to Shadow during Phase 3 to diversify attacks, or train multiple Shadow checkpoints
- Estimated cost: ~$3.60

```
Round 1 checkpoint (balance + locomotion + sword proximity)
├── Shadow (NPC) ──── train ~400 iters ──── freeze ─┐
└── Main Character                                   │
    ├── Phase 2 (locomotion + attack) ~400 iters     │
    └── Phase 3 (defense) ~400 iters ◄──────────────┘
           (frozen Shadow is the sparring partner)
```

Total estimated GPU cost: ~$11 for all three training runs.

## Responsiveness Tuning

The physics sets a floor on input response time. Tuning knobs in the humanoid MJCF:

| Parameter | Effect | Trade-off |
|-----------|--------|-----------|
| Body segment masses | Lower mass = faster acceleration | Less weighty feel |
| `actuatorfrcrange` | Higher max torques = faster joint response | Can look twitchy if too high |
| Simulation timestep | Higher Hz = lower per-step latency | Longer training time |
| Velocity tracking reward scale | Higher scale = more aggressive speed matching | Can fight disc naturalness |

Target: ~80-150ms response time (athletic human, responsive for games). These are tuned by modifying the humanoid XML and retraining short runs from existing checkpoints. No code changes needed.

## Runtime Architecture (v3 scope — not built now)

```
Controller Input
├── Left stick + camera heading → tar_vel (world space)
├── X/Y/B → intent one-hot
├── LB + timing check → block or parry intent
└── nothing → idle intent

Policy inference (~60Hz, CPU via ONNX)
├── Input: proprioception + sword + tar_vel + intent + enemy state
├── Output: 31 joint torques
└── Cost: ~0.3ms per inference

MuJoCo C step (~60Hz, CPU)
├── Apply torques → simulate → new joint transforms
└── Cost: ~0.5ms per step

Retarget + Render
├── MuJoCo joint transforms → Wrela 22-bone skeleton
└── GPU skinning → visbuf pipeline
```

Total physics budget: ~0.8ms per frame. No GPU, no Python, no CUDA at runtime. ONNX via `ort` crate, MuJoCo via C bindings.

## Versioning

**v1.1 (this sprint):**
- Shadow: autonomous combat vs static dummy
- Main Character: velocity tracking + light/heavy attack vs static dummy
- Test both in Newton ViewerGL locally

**v1.2 (next sprint):**
- Shadow trained as competent attacker
- Main Character Phase 3: dodge, block, parry against frozen Shadow
- Responsiveness tuning (humanoid XML mass/force adjustments)

**v2:**
- ONNX export, MuJoCo C in Rust, Wrela engine integration
- Blade expressions (style conditioning vectors on policy)
- Variable physics (gravity, friction, injury)
- Real controller input wired through

**v3:**
- Self-play (Shadow learns from Main Character and vice versa)
- Multiple weapon types with different mass distributions
- Cinematic actuator scaling for superhuman combat

## Success Criteria (v1.1)

Observable in Newton ViewerGL:

1. Main Character tracks randomized velocity targets (simulated stick input) with responsive locomotion
2. Light attack produces a fast, compact sword strike while maintaining locomotion
3. Heavy attack produces a visibly different motion — more wind-up, wider arc, more body commitment
4. Stick direction at moment of attack visibly influences swing trajectory
5. Idle produces a stable ready stance with sword held
6. Balance maintained throughout — natural recovery after swings
7. Motion quality is recognizably human (AMP discriminator active)
8. Shadow autonomously approaches dummy and attacks with natural motion

## Files Modified / Created

| File | Change |
|------|--------|
| `mimickit/envs/task_player_combat_env.py` | New env — velocity tracking from simulated stick input, 6-intent action set |
| `data/envs/task_player_combat_ss_env.yaml` | Player character env config (sword+shield) |
| `data/envs/task_shadow_combat_ss_env.yaml` | Shadow NPC env config (sword+shield, autonomous intents) |
| `args/shadow_combat_args.txt` | Shadow training args |
| `args/player_combat_args.txt` | Main character training args |
| `mimickit/envs/task_combat_env.py` | Add `enable_intent_reward` Round 2 support (already done) |

## Dependencies

- Round 1 checkpoint: `output/combat_ss_r1/model.pt` (sword+shield, ~273 steps stability)
- Sword+shield character: `data/assets/sword_shield/humanoid_sword_shield.xml`
- Locomotion mocap: `data/datasets/dataset_humanoid_sword_shield_locomotion.yaml`
- RunPod pod `zhigzy969kczyp` with H200 SXM
- Patched `base_agent.py` for shape-tolerant model loading (already on RunPod)

## Licensing

Reallusion motion data is noncommercial only. Commercial deployment (v2+) requires re-training on permissively licensed data (AMASS, CC-BY) or purchasing a Reallusion commercial license.
