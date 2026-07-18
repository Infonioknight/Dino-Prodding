from coach.coach_llm import validate_citations


def test_validate_citations_passes_when_every_paragraph_cites_a_move():
    report = (
        "# Coaching notes\n\n"
        "In move 3, your CoM offset hit 0.42 shoulder-widths outside the base "
        "of support — that's the balance loss on the left-hand reach.\n\n"
        "Move 5 was static (peak speed 0.3), good control there.\n"
    )
    assert validate_citations(report, valid_move_ids={3, 5}) == []


def test_validate_citations_flags_uncited_paragraph():
    report = (
        "# Coaching notes\n\n"
        "You should generally keep your hips closer to the wall.\n\n"
        "Move 3 showed a CoM offset of 0.42.\n"
    )
    violations = validate_citations(report, valid_move_ids={3})
    assert len(violations) == 1
    assert "hips closer to the wall" in violations[0]


def test_validate_citations_rejects_reference_to_nonexistent_move():
    report = "Move 99 had a huge offset.\n"
    violations = validate_citations(report, valid_move_ids={1, 2, 3})
    assert len(violations) == 1
