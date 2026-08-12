import ast
from pathlib import Path

import pytest
from mmcv import Config


ROOT = Path(__file__).resolve().parents[1]


def _module(path):
    return ast.parse((ROOT / path).read_text())


def _class_init_args(path, class_name):
    tree = _module(path)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    return [arg.arg for arg in item.args.args], item
    raise AssertionError(f"{class_name}.__init__ not found in {path}")


def load_config_namespace(path):
    namespace = {}
    exec(compile(path.read_text(), str(path), "exec"), namespace)
    return namespace


def test_repdetr3d_and_streampetr_accept_disabled_temporal_memory_config():
    for path, class_name in [
        ("projects/mmdet3d_plugin/models/detectors/repdetr3d.py", "RepDetr3D"),
        ("projects/mmdet3d_plugin/models/detectors/petr3d.py", "Petr3D"),
    ]:
        args, init_node = _class_init_args(path, class_name)
        assert "temporal_memory" in args
        assert "tm_debug_shapes" in args
        assert "tm_debug_latency" in args

        defaults = {
            arg.arg: default
            for arg, default in zip(
                init_node.args.args[-len(init_node.args.defaults):],
                init_node.args.defaults,
            )
        }
        assert isinstance(defaults["temporal_memory"], ast.Constant)
        assert defaults["temporal_memory"].value is None
        assert isinstance(defaults["tm_debug_shapes"], ast.Constant)
        assert defaults["tm_debug_shapes"].value is False
        assert isinstance(defaults["tm_debug_latency"], ast.Constant)
        assert defaults["tm_debug_latency"].value is False


def test_repdetr3d_eva02_temporal_memory_config_is_disabled_by_default():
    config = ROOT / "projects/configs/RepDETR3D/repdetr3d_eva02_800_bs2_seq_24e_temporal_memory.py"
    assert config.exists()

    namespace = {}
    exec(config.read_text(), namespace)

    assert namespace["_base_"] == ["./repdetr3d_eva02_800_bs2_seq_24e.py"]
    temporal_memory = namespace["model"]["temporal_memory"]
    assert temporal_memory["enabled"] is False
    assert temporal_memory["detector_adapter"] == "repdetr3d"
    assert temporal_memory["memory_is_pre_pe"] is True
    assert temporal_memory["pe_mode"] == "separate_recompute"
    assert namespace["model"]["tm_debug_shapes"] is False
    assert namespace["model"]["tm_debug_latency"] is False


def test_repdetr3d_eva02_temporal_memory_experiment_configs():
    cases = {
        "repdetr3d_eva02_800_bs2_seq_24e_tm_baseline.py": dict(enabled=False),
        "repdetr3d_eva02_800_bs2_seq_24e_tm_reuse_k3.py": dict(
            enabled=True, updater="reuse", keyframe_interval=3),
        "repdetr3d_eva02_800_bs2_seq_24e_tm_reuse_k5.py": dict(
            enabled=True, updater="reuse", keyframe_interval=5),
        "repdetr3d_eva02_800_bs2_seq_24e_tm_gated_k3_distill.py": dict(
            enabled=True, updater="gated_residual", keyframe_interval=3,
            use_distillation=True, train_with_teacher=True,
            freeze_pretrained_detector=True),
        "repdetr3d_eva02_800_bs2_seq_24e_tm_gated_k5_distill.py": dict(
            enabled=True, updater="gated_residual", keyframe_interval=5,
            use_distillation=True, train_with_teacher=True,
            freeze_pretrained_detector=True),
        "repdetr3d_eva02_800_bs2_seq_24e_tm_gated_k3_distill_ego.py": dict(
            enabled=True, updater="gated_residual", keyframe_interval=3,
            use_distillation=True, train_with_teacher=True,
            use_ego_conditioning=True, freeze_pretrained_detector=True),
        "repdetr3d_eva02_800_bs2_seq_24e_tm_partial_only_l8.py": dict(
            enabled=True, updater="partial_only", keyframe_interval=3,
            partial_layer="block8", train_queue_length=3,
            freeze_pretrained_detector=True),
        "repdetr3d_eva02_800_bs2_seq_24e_tm_partial_only_l12.py": dict(
            enabled=True, updater="partial_only", keyframe_interval=3,
            partial_layer="block12", train_queue_length=3,
            freeze_pretrained_detector=True),
        "repdetr3d_eva02_800_bs2_seq_24e_tm_partial_only_l16.py": dict(
            enabled=True, updater="partial_only", keyframe_interval=3,
            partial_layer="block16", train_queue_length=3,
            freeze_pretrained_detector=True),
        "repdetr3d_eva02_800_bs2_seq_24e_tm_gated_k2_l8_distill.py": dict(
            enabled=True, updater="gated_residual", keyframe_interval=2,
            partial_layer="block8", use_distillation=True,
            train_with_teacher=True, train_queue_length=2,
            freeze_pretrained_detector=True),
        "repdetr3d_eva02_800_bs2_seq_24e_tm_gated_cycle_l8_l12_l16_distill.py": dict(
            enabled=True, updater="gated_residual", keyframe_interval=4,
            policy=dict(type="cycle", actions=("L24", "L8", "L12", "L16")),
            partial_layers=(8, 12, 16), shared_memory_gate=True,
            use_distillation=True, train_with_teacher=True,
            train_queue_length=4, freeze_pretrained_detector=True),
        "repdetr3d_eva02_800_bs2_seq_24e_tm_attention_k3_l8_no_distill.py": dict(
            enabled=True, updater="attention_residual", keyframe_interval=3,
            partial_layer="block8", train_queue_length=3,
            freeze_pretrained_detector=True),
        "repdetr3d_eva02_800_bs2_seq_24e_tm_gated_attention_k3_l8_no_distill.py": dict(
            enabled=True, updater="gated_attention_residual", keyframe_interval=3,
            partial_layer="block8", train_queue_length=3,
            freeze_pretrained_detector=True),
        "repdetr3d_eva02_800_bs2_seq_24e_tm_per_camera_attention_k3_l8_no_distill.py": dict(
            enabled=True, updater="per_camera_attention_residual", keyframe_interval=3,
            partial_layer="block8", train_queue_length=3,
            freeze_pretrained_detector=True),
        "repdetr3d_eva02_800_bs2_seq_24e_tm_inverted_attention_k3_l8_no_distill.py": dict(
            enabled=True, updater="inverted_attention_residual", keyframe_interval=3,
            partial_layer="block8", train_queue_length=3,
            freeze_pretrained_detector=True),
        "repdetr3d_eva02_800_bs2_seq_24e_tm_mlp_policy_trainable_4level_similarity_no_distill.py": dict(
            enabled=True, updater="gated_residual", keyframe_interval=5,
            partial_layers=(8, 12, 16), train_queue_length=5,
            freeze_pretrained_detector=True, use_distillation=False,
            train_with_teacher=False, policy_target="similarity"),
        "repdetr3d_eva02_800_bs2_seq_24e_tm_mlp_policy_trainable_4level_reset5_similarity_no_distill.py": dict(
            enabled=True, updater="gated_residual", keyframe_interval=5,
            max_frames_between_full=5, partial_layers=(8, 12, 16),
            train_queue_length=5, freeze_pretrained_detector=True,
            use_distillation=False, train_with_teacher=False,
            policy_target="similarity"),
        "repdetr3d_eva02_800_bs2_seq_24e_tm_mlp_policy_trainable_4level_cka_no_distill.py": dict(
            enabled=True, updater="gated_residual", keyframe_interval=5,
            partial_layers=(8, 12, 16), train_queue_length=5,
            freeze_pretrained_detector=True, use_distillation=False,
            train_with_teacher=False, policy_target="cka"),
    }

    for filename, expected in cases.items():
        config = ROOT / "projects/configs/RepDETR3D" / filename
        assert config.exists(), filename
        namespace = {}
        exec(config.read_text(), namespace)
        assert namespace["_base_"]
        assert "work_dir" in namespace
        assert namespace["load_from"] == "./ckpts/repdetr3d_eva02_800_bs4_seq_24e.pth"
        temporal_memory = namespace["model"]["temporal_memory"]
        for key, value in expected.items():
            if key == "train_queue_length":
                assert namespace["data"]["train"]["queue_length"] == value
                continue
            if key == "policy_target":
                assert temporal_memory["policy_supervision"]["target"] == value, filename
                continue
            assert temporal_memory[key] == value, f"{filename}:{key}"
        assert namespace["model"]["tm_debug_latency"] is False


def test_repdetr3d_temporal_memory_trainable_updater_configs_keep_checkpoint_safe_ddp_mode():
    configs = [
        "repdetr3d_eva02_800_bs2_seq_24e_tm_gated_k3_distill.py",
        "repdetr3d_eva02_800_bs2_seq_24e_tm_gated_k5_distill.py",
        "repdetr3d_eva02_800_bs2_seq_24e_tm_gated_k3_distill_ego.py",
        "repdetr3d_eva02_800_bs2_seq_24e_tm_partial_only_l8.py",
        "repdetr3d_eva02_800_bs2_seq_24e_tm_gated_k2_l8_distill.py",
        "repdetr3d_eva02_800_bs2_seq_24e_tm_gated_cycle_l8_l12_l16_distill.py",
        "repdetr3d_eva02_800_bs2_seq_24e_tm_attention_k3_l8_no_distill.py",
        "repdetr3d_eva02_800_bs2_seq_24e_tm_gated_attention_k3_l8_no_distill.py",
        "repdetr3d_eva02_800_bs2_seq_24e_tm_per_camera_attention_k3_l8_no_distill.py",
        "repdetr3d_eva02_800_bs2_seq_24e_tm_inverted_attention_k3_l8_no_distill.py",
        "repdetr3d_eva02_800_bs2_seq_24e_tm_mlp_policy_trainable_4level_similarity_no_distill.py",
        "repdetr3d_eva02_800_bs2_seq_24e_tm_mlp_policy_trainable_4level_reset5_similarity_no_distill.py",
        "repdetr3d_eva02_800_bs2_seq_24e_tm_mlp_policy_trainable_4level_cka_no_distill.py",
    ]

    for filename in configs:
        namespace = {}
        exec((ROOT / "projects/configs/RepDETR3D" / filename).read_text(), namespace)
        assert namespace.get("find_unused_parameters", False) is False, filename


def test_depth_router_config_is_online_only_reset5_and_single_sample_eval():
    filename = "repdetr3d_eva02_800_bs2_seq_24e_tm_depth_router_budget50_reset5.py"
    config = ROOT / "projects/configs/RepDETR3D" / filename
    assert config.exists(), filename

    namespace = {}
    exec(config.read_text(), namespace)
    temporal_memory = namespace["model"]["temporal_memory"]
    router = temporal_memory["router"]

    assert temporal_memory["max_frames_between_full"] == 5
    assert temporal_memory["keyframe_interval"] == 5
    assert temporal_memory["partial_layers"] == (8, 12, 16)
    assert router["type"] == "depth_exit"
    assert router["actions"] == ("L8", "L12", "L16", "L24")
    assert router["target_cost"] == 0.5
    assert all("gt" not in name.lower() for name in router["feature_names"])
    assert all("drift" not in name.lower() for name in router["feature_names"])
    assert namespace["data"]["samples_per_gpu"] == 4
    assert namespace["router_eval_cfg"]["data"]["samples_per_gpu"] == 1


def test_mod_configs_use_global_half_capacity_and_differ_by_training_regime():
    filenames = {
        "frozen": "repdetr3d_eva02_800_bs2_seq_24e_tm_mod_global50_frozen.py",
        "warmup": "repdetr3d_eva02_800_bs2_seq_24e_tm_mod_global50_frozen_adapter_warmup.py",
        "joint": "repdetr3d_eva02_800_bs2_seq_24e_tm_mod_global50_joint.py",
    }
    configs = {}
    for name, filename in filenames.items():
        path = ROOT / "projects/configs/RepDETR3D" / filename
        assert path.exists(), filename
        namespace = {}
        exec(path.read_text(), namespace)
        configs[name] = namespace

        mod_router = namespace["model"]["temporal_memory"]["mod_router"]
        assert mod_router["routed_blocks"] == (9, 12, 15, 18, 21)
        assert mod_router["capacity_ratio"] == 0.5
        if name == "frozen":
            assert namespace["data"]["samples_per_gpu"] == 4
        assert namespace["router_eval_cfg"]["data"]["samples_per_gpu"] == 1

    assert configs["frozen"]["model"]["temporal_memory"]["mod_router"]["train_backbone"] is False
    assert configs["warmup"]["model"]["temporal_memory"]["mod_router"]["train_backbone"] is False
    assert configs["warmup"]["model"]["temporal_memory"]["mod_router"]["adapter_warmup_only"] is True
    assert configs["joint"]["model"]["temporal_memory"]["mod_router"]["train_backbone"] is True


def test_streampetr_eva02_highres_teacher_config_is_camera_only_and_checkpoint_safe():
    config = ROOT / "projects/configs/StreamPETR/stream_petr_eva02_flash_1600_bs1_seq_24e.py"
    namespace = {}
    exec(config.read_text(), namespace)

    assert namespace["num_gpus"] == 4
    assert namespace["batch_size"] == 1
    assert namespace["num_epochs"] == 24
    assert namespace["num_iters_per_epoch"] == 28130 // 4
    assert namespace["ida_aug_conf"]["final_dim"] == (640, 1600)
    assert namespace["input_modality"]["use_camera"] is True
    assert namespace["input_modality"]["use_lidar"] is False
    assert namespace["input_modality"]["use_radar"] is False
    assert namespace["load_from"] == "./ckpts/eva02_L_coco_det_sys_o365_remapped.pth"
    assert namespace["checkpoint_config"]["interval"] == namespace["num_iters_per_epoch"]
    assert namespace["checkpoint_config"]["max_keep_ckpts"] == 2
    assert namespace["runner"]["max_iters"] == (
        namespace["num_epochs"] * namespace["num_iters_per_epoch"])
    assert [hook["type"] for hook in namespace["log_config"]["hooks"]] == [
        "TextLoggerHook", "TensorboardLoggerHook"]

    loaded = Config.fromfile(str(config))
    assert loaded.train_pipeline[4]["data_aug_conf"]["final_dim"] == (640, 1600)
    assert loaded.test_pipeline[1]["data_aug_conf"]["final_dim"] == (640, 1600)
    assert loaded.evaluation["pipeline"][1]["data_aug_conf"]["final_dim"] == (640, 1600)


@pytest.mark.parametrize('filename, expected', [
    (
        'stream_petr_eva02_flash_1600_bs1_seq_24e_tm_baseline.py',
        dict(enabled=False, detector_adapter='streampetr'),
    ),
    (
        'stream_petr_eva02_flash_1600_bs1_seq_24e_tm_reuse_k3.py',
        dict(
            enabled=True,
            detector_adapter='streampetr',
            updater='reuse',
            keyframe_interval=3,
            train_queue_length=3,
        ),
    ),
    (
        'stream_petr_eva02_flash_1600_bs1_seq_24e_tm_gated_l12_k4_distill.py',
        dict(
            enabled=True,
            detector_adapter='streampetr',
            updater='gated_residual',
            partial_layer='block12',
            keyframe_interval=4,
            max_frames_between_full=4,
            memory_is_pre_pe=True,
            pe_mode='separate_recompute',
            train_with_teacher=True,
            use_distillation=True,
            train_queue_length=4,
        ),
    ),
    (
        'stream_petr_eva02_flash_1600_bs1_seq_24e_tm_mlp_policy_reset4_similarity_online_offline_teacher_forcing_motion_reset_feature_noise_no_distill.py',
        dict(
            enabled=True,
            detector_adapter='streampetr',
            updater='gated_residual',
            partial_layers=(8, 12, 16),
            max_frames_between_full=4,
            use_distillation=False,
            train_queue_length=4,
        ),
    ),
])
def test_streampetr_highres_temporal_configs(filename, expected):
    path = ROOT / 'projects/configs/StreamPETR' / filename
    namespace = load_config_namespace(path)
    tm = namespace['model']['temporal_memory']
    for key, value in expected.items():
        if key == 'train_queue_length':
            assert namespace['data']['train']['queue_length'] == value
        else:
            assert tm[key] == value

    loaded = Config.fromfile(str(path))
    assert loaded.input_modality.use_camera is True
    assert loaded.input_modality.use_lidar is False
    assert loaded.input_modality.use_radar is False
    for split in ('train', 'val', 'test'):
        assert loaded.data[split].modality.use_camera is True
        assert loaded.data[split].modality.use_lidar is False
        assert loaded.data[split].modality.use_radar is False


def test_streampetr_highres_temporal_policy_marks_oracle_training_features():
    filename = (
        'stream_petr_eva02_flash_1600_bs1_seq_24e_'
        'tm_mlp_policy_reset4_similarity_online_offline_teacher_forcing_'
        'motion_reset_feature_noise_no_distill.py'
    )
    namespace = load_config_namespace(
        ROOT / 'projects/configs/StreamPETR' / filename)
    tm = namespace['model']['temporal_memory']
    policy = tm['policy']
    semantics = tm['policy_feature_semantics']

    assert policy['actions'] == dict(
        easy='L8', medium='L12', hard='L16', very_hard='L24')
    assert policy['feature_names'] == (
        semantics['online_deployable_features']
        + semantics['offline_training_only_oracle_features'])
    assert 'ego_speed_mps' in semantics['online_deployable_features']
    assert 'max_object_speed_mps' in semantics[
        'offline_training_only_oracle_features']
    assert semantics['training_offline_source'] == 'previous_ground_truth_teacher_forcing'
    assert semantics['validation_inference_offline_source'] == 'previous_predictions_only'
    assert tm['policy_offline_features'] == 'teacher_forcing'
    assert tm['adaptive_reset'] == dict(
        enabled=True,
        max_frames_between_full=2,
        ego_speed_mps=2.0,
        max_object_speed_mps=5.0,
    )
    assert tm['policy_supervision']['target'] == 'similarity'
    assert tm['policy_supervision']['feature_noise_std'] == 0.10


def test_streampetr_highres_temporal_configs_have_unique_handoff_metadata():
    config_dir = ROOT / 'projects/configs/StreamPETR'
    filenames = [
        'stream_petr_eva02_flash_1600_bs1_seq_24e_tm_baseline.py',
        'stream_petr_eva02_flash_1600_bs1_seq_24e_tm_reuse_k3.py',
        'stream_petr_eva02_flash_1600_bs1_seq_24e_tm_gated_l12_k4_distill.py',
        (
            'stream_petr_eva02_flash_1600_bs1_seq_24e_'
            'tm_mlp_policy_reset4_similarity_online_offline_teacher_forcing_'
            'motion_reset_feature_noise_no_distill.py'
        ),
    ]
    namespaces = {
        filename: load_config_namespace(config_dir / filename)
        for filename in filenames
    }
    work_dirs = [namespace['work_dir'] for namespace in namespaces.values()]
    assert len(work_dirs) == len(set(work_dirs))
    for filename, namespace in namespaces.items():
        assert Path(namespace['work_dir']).name == Path(filename).stem

    teacher_checkpoint = (
        './work_dirs/stream_petr_eva02_flash_1600_bs1_seq_24e/latest.pth')
    assert 'load_from' not in namespaces[filenames[0]]
    assert 'load_from' not in namespaces[filenames[1]]
    assert namespaces[filenames[2]]['load_from'] == teacher_checkpoint
    assert namespaces[filenames[3]]['load_from'] == teacher_checkpoint
    assert all(
        namespace.get('load_from')
        != './ckpts/repdetr3d_eva02_800_bs4_seq_24e.pth'
        for namespace in namespaces.values())

    results = (ROOT / 'RESULTS.md').read_text()
    for filename, namespace in namespaces.items():
        assert results.count(filename) == 1
        work_dir = namespace['work_dir']
        if work_dir.startswith('./'):
            work_dir = work_dir[2:]
        assert work_dir in results
