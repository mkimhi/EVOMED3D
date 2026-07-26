import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _quat_from_yaw(degrees):
    yaw = math.radians(degrees)
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def test_collect_policy_signal_records_computes_motion_and_annotations():
    from tools.policy_signal_stats import collect_policy_signal_records

    infos = [
        dict(
            token="a0",
            scene_token="scene-a",
            timestamp=0,
            ego2global_translation=[0.0, 0.0, 0.0],
            ego2global_rotation=_quat_from_yaw(0),
            lidar2ego_translation=[0.0, 0.0, 0.0],
            lidar2ego_rotation=_quat_from_yaw(0),
            gt_boxes=np.array([[3.0, 4.0, 0.0, 2, 3, 4, 0], [6.0, 8.0, 0.0, 1, 1, 1, 0]]),
            gt_names=np.array(["car", "truck"]),
            gt_velocity=np.array([[3.0, 4.0], [0.0, 0.0]]),
            valid_flag=np.array([True, False]),
            bboxes2d=[np.array([[0.0, 0.0, 10.0, 20.0]])],
        ),
        dict(
            token="a1",
            scene_token="scene-a",
            timestamp=500000,
            ego2global_translation=[3.0, 4.0, 0.0],
            ego2global_rotation=_quat_from_yaw(45),
            lidar2ego_translation=[0.0, 0.0, 0.0],
            lidar2ego_rotation=_quat_from_yaw(0),
            gt_boxes=np.array([[0.0, 5.0, 0.0, 2, 3, 4, 0], [12.0, 5.0, 0.0, 1, 2, 3, 0]]),
            gt_names=np.array(["car", "truck"]),
            gt_velocity=np.array([[0.0, 0.0], [6.0, 8.0]]),
            valid_flag=np.array([True, True]),
            bboxes2d=[np.array([[0.0, 0.0, 20.0, 20.0], [0.0, 0.0, 10.0, 10.0]])],
        ),
        dict(
            token="b0",
            scene_token="scene-b",
            timestamp=1000000,
            ego2global_translation=[100.0, 100.0, 0.0],
            ego2global_rotation=_quat_from_yaw(90),
            lidar2ego_translation=[0.0, 0.0, 0.0],
            lidar2ego_rotation=_quat_from_yaw(0),
            gt_boxes=np.zeros((0, 7)),
            gt_names=np.array([]),
            gt_velocity=np.zeros((0, 2)),
            valid_flag=np.zeros((0,), dtype=bool),
            bboxes2d=[],
        ),
    ]

    records = collect_policy_signal_records(infos, new_object_match_distance_m=6.0)

    assert records[0]["ego_speed_mps"] == 0.0
    assert records[0]["abs_yaw_rate_deg_s"] == 0.0
    assert records[0]["frame_number"] == 0
    assert records[0]["num_annotations"] == 1
    assert records[0]["avg_object_distance_m"] == 5.0
    assert records[0]["avg_box_volume_m3"] == 24.0
    assert records[0]["avg_object_speed_mps"] == 5.0
    assert records[0]["new_object_proxy_count"] == 1
    assert records[0]["camera_annotation_count"] == 1
    assert records[0]["avg_2d_box_area_px"] == 200.0

    assert records[1]["frame_number"] == 1
    assert records[1]["ego_speed_mps"] == 10.0
    assert records[1]["abs_yaw_rate_deg_s"] == 90.0
    assert records[1]["ego_accel_mps2"] == 20.0
    assert records[1]["yaw_accel_deg_s2"] == 180.0
    assert records[1]["num_annotations"] == 2
    assert records[1]["avg_object_distance_m"] == 9.0
    assert records[1]["avg_box_volume_m3"] == 15.0
    assert records[1]["new_object_proxy_count"] == 1
    assert records[1]["matched_object_proxy_count"] == 1
    assert records[1]["near_object_count_10m"] == 1
    assert records[1]["camera_annotation_count"] == 2
    assert records[1]["avg_2d_box_area_px"] == 250.0

    assert records[2]["frame_number"] == 0
    assert records[2]["ego_speed_mps"] == 0.0
    assert records[2]["abs_yaw_rate_deg_s"] == 0.0
    assert records[2]["num_annotations"] == 0
    assert math.isnan(records[2]["avg_object_distance_m"])


def test_yaw_delta_uses_shortest_angle_across_wraparound():
    from tools.policy_signal_stats import yaw_delta_degrees

    assert abs(yaw_delta_degrees(math.radians(179), math.radians(-179)) - 2.0) < 1e-6
    assert abs(yaw_delta_degrees(math.radians(-179), math.radians(179)) + 2.0) < 1e-6


def test_collect_policy_signal_records_can_disable_new_object_proxy():
    from tools.policy_signal_stats import collect_policy_signal_records

    infos = [
        dict(
            token="a0", scene_token="scene-a", timestamp=0,
            ego2global_translation=[0.0, 0.0, 0.0], ego2global_rotation=_quat_from_yaw(0),
            lidar2ego_translation=[0.0, 0.0, 0.0], lidar2ego_rotation=_quat_from_yaw(0),
            gt_boxes=np.array([[0.0, 0.0, 0.0, 1, 1, 1, 0]]),
            gt_names=np.array(["car"]), valid_flag=np.array([True]),
        ),
    ]

    records = collect_policy_signal_records(infos, compute_new_object_proxy=False)

    assert "new_object_proxy_count" not in records[0]
