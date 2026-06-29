# Novelty Audit

Date: 2026-06-29

## Closest Prior Works Inspected

- VERA: Turning Video Models into Generalist Robot Policies, arXiv 2605.27817.
- Neural Jacobian Fields / Controlling diverse robots by inferring Jacobian fields with deep networks, arXiv 2407.08722.
- EVA: Aligning Video World Models with Executable Robot Actions via Inverse Dynamics Rewards, arXiv 2603.17808.
- World Action Verifier: Self-Improving World Models via Forward-Inverse Asymmetry, arXiv 2604.01985.
- Model Based Planning with Energy Based Models, arXiv 1909.06878.
- Planning with Sequence Models through Iterative Energy Minimization / LEAP, OpenReview cVFD6qE8gnY.
- Compositional Generative Modeling, arXiv 2402.01103.
- Potential Based Diffusion Motion Planning, arXiv 2407.06169.
- 3D Neural Scene Representations for Visuomotor Control, arXiv 2107.04004.
- RoboDreamer, arXiv 2404.12377.
- Large Video Planner Enables Generalizable Robot Control, arXiv 2512.15840.
- Visual foresight / MPC, causal confusion in imitation learning, and observational overfitting references.

## Main Novelty Risk

World Action Verifier is the closest overlap. It studies executable generated futures through a forward-inverse asymmetry signal. EVA is also close because it aligns video world models with executable robot actions through inverse-dynamics rewards. VERA and NJF are close because they use Jacobian-style local action maps for robot policy/action translation.

## Why Novelty Survives

The upgraded paper does not claim to be the first executable-video or inverse-dynamics verifier. Its narrower contribution is:

1. A tangent-space residual measuring distance from an imagined representation transition to the local action-reachable set induced by `J_phi(z)`.
2. A learned-Jacobian diagnostic-interface framing: the same local action map supports inverse action, failure diagnosis, and repair.
3. A CPU benchmark isolating non-executable generated futures across four embodiment/action-map structures plus rendered keypoint futures.
4. Hostile baselines: inverse-dynamics reconstruction, learned classifier, WAV-style proxy, wrong-map residuals, smoothness, passive likelihood, action norm, and path length.
5. Robustness audits over sample count, Jacobian noise, wrong maps, and contact-mode mismatch.
6. Embodiment swapping by changing the action map while keeping the same imagined displacement.

The inspected prior work did not reveal this exact combination of local tangent-space certificate, learned action-map diagnostic interface, low-compute benchmark, robustness audit, and repair objective.

## Claims To Avoid

- Do not claim first-ever executable video verification.
- Do not claim to improve NJF, VERA, EVA, or WAV.
- Do not claim real-robot validation.
- Do not claim the repair solver is a general nonlinear optimal-control method.
- Do not claim state-of-the-art robot policy performance.
- Do not call the toy/keypoint future generators full video models.
- Do not claim learned smooth Jacobians solve contact; the pusher result shows they do not.

## Current Novelty Verdict

Novelty survives more strongly as a Lester-aligned bridge paper. It is a controlled representation/actionability study, not a full robot foundation-model result. It fits Jason's agenda as a robot-brain mechanism paper around learned action maps, failure diagnosis, repair, and embodiment transfer. The main novelty-risk boundary remains WAV/EVA; the paper must continue to frame actionability as a tangent-space certificate and repair factor, not as first executable-video verification.
