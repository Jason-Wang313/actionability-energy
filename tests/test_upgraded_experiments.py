from experiments.exp_failure_prediction import run_failure_prediction
from experiments.exp_visual_keypoint import run_visual_keypoint


def test_failure_prediction_includes_hostile_and_learned_j_scores():
    scores, metrics, _ = run_failure_prediction(n=8, T=6, seed=7)
    for col in ["learned_j_residual", "idm_reconstruction_error", "wav_proxy", "learned_classifier"]:
        assert col in scores.columns
    assert {"learned_j_residual", "learned_classifier"}.issubset(set(metrics["score"]))


def test_visual_keypoint_experiment_runs_small():
    scores, metrics, examples = run_visual_keypoint(seed=8, quick=True)
    assert "visual_actionability" in scores.columns
    assert "learned_visual_actionability" in scores.columns
    assert len(metrics) > 0
    assert examples
