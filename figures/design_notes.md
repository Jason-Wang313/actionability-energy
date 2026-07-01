# Design Notes

## Shared Visual Strategy

The figure system uses a blue-first palette inspired by the clear bar figure in "Reasoning with Sampling". Dense rankings are split into sparse panels with direct numeric labels. Qualitative panels use light cards, direct labels, and the same start/end/actionability colors throughout. Residual-field figures use a warm magma-style map so energy landscapes do not collapse into an all-blue paper.

## Failure Modes To Avoid

- Long horizontal ranked bars with too many methods.
- Heatmaps whose low-energy regions become pure black.
- Box-arrow diagrams that do not show the robot-brain mechanism.
- Legends that require repeated cross-referencing.
- Visual examples colliding with metric panels.

## Figure-Specific Takeaways

- Figure 1 should make the paper's core idea legible in five seconds: generated futures need an embodiment-aware executability test.
- Figure 4 should preserve the hostile-baseline evidence while looking like a deliberate comparison, not a dumped metric ranking.
- Figure 5 should make the robustness story clear without requiring the reader to decode a four-by-many heatmap.
- Figure 7 should keep the original warm residual-field feel Jason liked, while replacing the harsh black point-mass panel with a dark-purple low-residual color.
- Figure 8 should separate visual/keypoint examples from metric evidence and eliminate all panel overlap.
