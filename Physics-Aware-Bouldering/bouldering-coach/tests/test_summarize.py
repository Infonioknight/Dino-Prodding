from coach.summarize import Move, MoveAction, MovesFile, VideoMeta, Climber


def test_moves_file_schema_round_trips():
    move = Move(
        id=3,
        t=[8.2, 10.1],
        action=MoveAction(limb="left_hand", from_hold=4, to_hold=7),
        contacts_during=["right_hand:5", "left_foot:2", "right_foot:3"],
        com_offset_max_norm=0.42,
        peak_com_speed_norm=1.8,
        dynamic=True,
        torso_wall_angle_deg=31,
        keyframe="moves/003.jpg",
    )
    meta = VideoMeta(fps=30, duration_s=22.4, wall_angle_deg=12.0, wall_angle_confidence=0.71)
    moves_file = MovesFile(video_meta=meta, climber=Climber(), moves=[move])

    dumped = moves_file.model_dump_json()
    reloaded = MovesFile.model_validate_json(dumped)
    assert reloaded.moves[0].action.to_hold == 7
    assert reloaded.video_meta.wall_angle_confidence == 0.71
