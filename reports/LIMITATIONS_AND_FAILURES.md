# Limitations and Failures

Date: 2026-06-29

## Synthetic Scope

All environments are low-dimensional deterministic simulations or CPU-rendered keypoint representations. They are designed to isolate actionability failure modes, not to model real robot physics faithfully.

## Local Linearity

The residual depends on a local action map. It can miss failures caused by long-horizon compounding, large actions, perception drift, nonlocal dynamics, or contact-mode switches outside the local model.

## Learned Jacobian Errors

The learned-J experiment uses small local linear regressors. It does not train a neural Jacobian field. The upgraded results are strong for point mass, car, and arm with enough local data, but weak for the contact pusher. This is useful evidence but not a solved problem.

## Contact Discontinuities

The planar pusher exposes contact/no-contact mode changes, and learned smooth local maps struggle there. This is the most important failure mode. A stronger version should add active contact probing, a tangent-cone formulation, or a mode-aware Jacobian field.

## Learned Classifier Baseline

The learned classifier reaches near-perfect aggregate performance on this synthetic distribution. That makes the benchmark easy to pattern-fit. The paper therefore treats the classifier as a hostile baseline and emphasizes that actionability supplies a mechanism-level certificate rather than just a discriminative score.

## Repair Solver

Repair uses projected trajectory inference and simple embodiment-aware controllers for CPU speed. It is not a full differentiable nonlinear trajectory optimizer. The results show that adding an actionability term can repair controlled failure cases, not that the solver is globally optimal.

## No Real-Robot Validation

The package does not validate on a physical robot or on PushT. Real-robot or stronger simulator experiments would be needed before making deployment claims.

## Figure and Paper Quality

Figures were redesigned around a figure-first argument: main concept, learned-J diagnostic interface, hostile baseline comparison, robustness audit, visual/keypoint futures, repair, and embodiment swap. They are substantially better than the v0 figures, but a final human design pass in Figma/Illustrator would still help them compete with the strongest internal figures in Yilun Du papers.

## Future Experiments That Would Strengthen The Paper

- Add PushT or ManiSkill-style manipulation with a learned representation.
- Add active local probing for contact-mode diagnosis.
- Compare against a stronger learned inverse-dynamics verifier and a closer WAV-style implementation.
- Use a differentiable optimizer over states and controls for repair.
- Train a small neural or mixture-of-modes action map and evaluate robustness under representation noise.
- Add real robot or teleoperation traces for one contact-rich task.
