# Limitations and Failures

Date: 2026-06-29

## Toy-Only Scope

All environments are low-dimensional deterministic simulations. They are designed to isolate actionability failure modes, not to model real robot physics faithfully.

## Local Linearity

The residual depends on a local action map. It can miss failures caused by long-horizon compounding, large actions, perception drift, nonlocal dynamics, or contact-mode switches outside the local model.

## Learned Jacobian Errors

The learned-J experiment uses small local linear regressors. It does not train a neural Jacobian field. Sample efficiency curves should be read as a sanity check that action maps can be approximated, not as a claim about NJF-scale learning.

## Contact Discontinuities

The planar pusher exposes contact/no-contact mode changes, but the contact model is intentionally simple. Real contact dynamics include friction cones, rotation, compliance, object geometry, and stochasticity not represented here.

## Repair Solver

Repair uses projected trajectory inference and simple embodiment-aware controllers for CPU speed. It is not a full differentiable nonlinear trajectory optimizer. The results show that adding an actionability term can repair controlled failure cases, not that the solver is globally optimal.

## No Real-Robot Validation

The package does not validate on a physical robot or on PushT. Real-robot or stronger simulator experiments would be needed before making deployment claims.

## Figure and Paper Quality

Figures are vector-first matplotlib outputs with a consistent palette. They are cleaner than default plots but still below the strongest custom-designed figures in top Yilun Du papers. A human design pass in Figma/Illustrator would improve first-impression quality.

## Future Experiments That Would Strengthen The Paper

- Add PushT or ManiSkill-style manipulation with a learned representation.
- Compare against a learned inverse-dynamics verifier and a WAV-style score.
- Use a differentiable optimizer over states and controls for repair.
- Train a small neural action map and evaluate robustness under representation noise.
- Add real robot or teleoperation traces for one contact-rich task.
