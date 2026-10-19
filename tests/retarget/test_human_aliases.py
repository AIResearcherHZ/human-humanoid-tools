"""Source skeleton aliases that must resolve to one anatomical target."""

from hhtools.retarget.newton_basic.human_aliases import (
    auto_source_to_canonical,
    list_detected_rig_type,
)


def test_biomechanical_glb_uses_arm_and_upper_spine_targets() -> None:
    names = (
        "b_root",
        "b_spine0", "b_spine1", "b_spine2", "b_spine3",
        "b_l_shoulder", "b_l_arm", "b_l_forearm", "b_l_wrist",
        "b_r_shoulder", "b_r_arm", "b_r_forearm", "b_r_wrist",
        "b_l_upleg", "b_l_leg", "b_l_talocrural", "b_l_ball", "b_l_foot",
        "b_r_upleg", "b_r_leg", "b_r_talocrural", "b_r_ball", "b_r_foot",
    )

    mapping = auto_source_to_canonical(names)

    assert list_detected_rig_type(names) == "Biomechanical b_* GLB"
    assert mapping["b_spine1"] == "spine"
    assert mapping["b_spine3"] == "chest"
    assert mapping["b_l_shoulder"] == "left_collar"
    assert mapping["b_l_arm"] == "left_shoulder"
    assert mapping["b_r_shoulder"] == "right_collar"
    assert mapping["b_r_arm"] == "right_shoulder"
    assert mapping["b_l_talocrural"] == "left_ankle"
    assert mapping["b_l_ball"] == "left_foot"

    consumed = {"spine", "chest", "left_shoulder", "right_shoulder"}
    candidates = {
        canonical: [source for source, target in mapping.items() if target == canonical]
        for canonical in consumed
    }
    assert all(len(sources) == 1 for sources in candidates.values())
