from concept_platform.approvals import EvaluationVersionBundle, plan_recalculation, thread_id_for


def base_bundle() -> EvaluationVersionBundle:
    return EvaluationVersionBundle(
        run_id="run-1",
        concept_version="concept-v1",
        evaluation_version="evaluation-v1",
        evidence_manifest_hash="evidence-v1",
        model_rule_versions={"solver": "m1", "rule-set": "r1"},
    )


def test_concept_version_change_creates_new_thread_and_invalidates_approval():
    old = base_bundle()
    new = old.model_copy(update={"concept_version": "concept-v2"})
    assert thread_id_for(old) != thread_id_for(new)
    plan = plan_recalculation(old, new)
    assert plan.approval_invalidated
    assert plan.affected_nodes == (
        "package", "drive_energy", "engineering_core", "reality_check",
        "expert_review", "concept_paper", "report_issuance",
    )


def test_evaluation_version_change_creates_new_thread_and_recomputes_downstream_only():
    old = base_bundle()
    new = old.model_copy(update={"evaluation_version": "evaluation-v2"})
    assert thread_id_for(old) != thread_id_for(new)
    assert plan_recalculation(old, new).affected_nodes == (
        "engineering_core", "reality_check", "expert_review", "concept_paper", "report_issuance",
    )


def test_evidence_manifest_change_creates_new_thread_and_invalidates_approval():
    old = base_bundle()
    new = old.model_copy(update={"evidence_manifest_hash": "evidence-v2"})
    assert thread_id_for(old) != thread_id_for(new)
    plan = plan_recalculation(old, new)
    assert plan.approval_invalidated
    assert "package" in plan.affected_nodes and "reality_check" in plan.affected_nodes


def test_model_or_rule_version_change_creates_new_thread_and_invalidates_approval():
    old = base_bundle()
    new = old.model_copy(update={"model_rule_versions": {"solver": "m2", "rule-set": "r1"}})
    assert thread_id_for(old) != thread_id_for(new)
    plan = plan_recalculation(old, new)
    assert plan.approval_invalidated
    assert "engineering_core" in plan.affected_nodes and "reality_check" in plan.affected_nodes
