from pathlib import Path

from mmcv import Config


ROOT = Path(__file__).resolve().parents[1]
COUNTERFACTUAL_CONFIG = ROOT / (
    'projects/configs/RepDETR3D/'
    'repdetr3d_eva02_800_bs2_seq_24e_tm_mlp_policy_4level_reset4_'
    'counterfactual_teacher_forcing_no_distill.py'
)


def test_counterfactual_reset4_config():
    cfg = Config.fromfile(str(COUNTERFACTUAL_CONFIG))
    tm = cfg.model.temporal_memory

    assert tm.max_frames_between_full == 4
    assert tm.policy.load_weights is True
    assert tm.policy_supervision.target == 'counterfactual_action'
    assert tm.policy_supervision.start_train_step == 4219
    assert tm.policy_supervision.max_train_steps == 42192
    assert tm.policy_offline_features == 'teacher_forcing'
    assert cfg.optimizer.paramwise_cfg.custom_keys.temporal_memory_policy.lr_mult == 0.1
