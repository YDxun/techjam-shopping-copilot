from __future__ import annotations

import json
import math
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from agent.dialogue.diagnostics import DecisionTraceRecorder, DialogueDecisionTrace
from agent.dialogue.models import (
    CandidateAttributeSignal,
    CandidateQuestionSignals,
    Constraint,
    ConstraintOperation,
    ConstraintStrength,
    DialogueAct,
    DialogueState,
    GuardAction,
    GuardDecision,
    OperationKind,
    Polarity,
    QuestionDecision,
    RecognitionResult,
    RecognitionSource,
)
from config.env_config import EnvConfig
from config.loader import ConfigError, load_config
from config.models import (
    AppConfig,
    DecisionConfig,
    DecisionTraceConfig,
    DialogueUnderstandingConfig,
    LLMConfig,
)


def trace(
    session_id: str = "session-17",
    *,
    decision_reason: str = "highest_dynamic_utility",
    guard_action: str = "apply",
) -> DialogueDecisionTrace:
    """A hand-written, safe diagnostic record; no raw conversational input is accepted."""
    return DialogueDecisionTrace(
        session_id=session_id,
        turn=2,
        recognition_source="llm",
        dialogue_act="add_constraint",
        recognition_confidence=0.8754321,
        ambiguities=("color", "size"),
        fallback_reason="rule_confidence_below_threshold",
        guard_action=guard_action,
        guard_reason="guard_disabled",
        intent_version=3,
        added_constraints=(("Color", "  RED  ", "hard"),),
        removed_constraints=(("brand", "old-brand", "soft"),),
        candidate_count=27,
        score_summary={"selected_utility": float("nan"), "unknown": 1.0},
        missing_rates={"color": float("inf")},
        attribute_scores={
            "color": {
                "expected_shrink": 0.123456789,
                "resolve_at_10": 0.9,
                "raw_llm_response": 99.0,
            }
        },
        selected_attribute="color",
        decision_reason=decision_reason,
        finish_pressure=0.333333333,
        lookahead_depth=2,
        recommendation_count=10,
        prompt_tokens=12,
        completion_tokens=7,
    )


class DecisionTraceConfigTest(unittest.TestCase):
    def test_decision_trace_defaults_and_environment_overrides_all_fields(self) -> None:
        # Break caught: enabling diagnostics changes the safe default or ignores a switch.
        default = EnvConfig.from_env(environ={}).diagnostics.decision_trace
        configured = EnvConfig.from_env(
            environ={
                "SHOPPING_DIAGNOSTICS__DECISION_TRACE__ENABLED": "1",
                "SHOPPING_DIAGNOSTICS__DECISION_TRACE__INCLUDE_ATTRIBUTE_SCORES": "0",
                "SHOPPING_DIAGNOSTICS__DECISION_TRACE__INCLUDE_STATE_DIFF": "false",
                "SHOPPING_DIAGNOSTICS__DECISION_TRACE__MAX_TRACES": "9",
                "SHOPPING_DIAGNOSTICS__DECISION_TRACE__OUTPUT_PATH": "tmp/traces.jsonl",
            }
        ).diagnostics.decision_trace

        self.assertFalse(default.enabled)
        self.assertEqual(default.max_traces, 5000)
        self.assertTrue(configured.enabled)
        self.assertFalse(configured.include_attribute_scores)
        self.assertFalse(configured.include_state_diff)
        self.assertEqual(configured.max_traces, 9)
        self.assertEqual(configured.output_path, "tmp/traces.jsonl")

    def test_decision_trace_rejects_invalid_cap_and_blank_path(self) -> None:
        # Break caught: unusable diagnostics config is allowed through the canonical loader.
        with self.assertRaisesRegex(ConfigError, "diagnostics.decision_trace.max_traces"):
            load_config(
                overrides={"diagnostics": {"decision_trace": {"max_traces": -1}}}, environ={}
            )
        with self.assertRaisesRegex(ConfigError, "diagnostics.decision_trace.output_path"):
            load_config(
                overrides={"diagnostics": {"decision_trace": {"output_path": "  "}}}, environ={}
            )

    def test_app_config_old_positional_llm_argument_remains_aligned(self) -> None:
        # Break caught: inserting diagnostics before llm shifts legacy positional callers.
        legacy_llm = LLMConfig(provider="none")
        config = AppConfig(
            "dev",
            "bm25",
            10,
            "embedding",
            "reranker",
            "offline.npy",
            "encoder",
            "other",
            False,
            False,
            None,
            "results.json",
            3,
            False,
            False,
            False,
            DialogueUnderstandingConfig(),
            DecisionConfig(),
            legacy_llm,
        )

        self.assertIs(config.llm, legacy_llm)
        self.assertFalse(config.diagnostics.decision_trace.enabled)


class DecisionTraceRecorderTest(unittest.TestCase):
    def test_trace_categories_are_closed_allowlists_in_repr_export_and_summary(self) -> None:
        # Break caught: underscore-only secret strings survive category normalization anywhere.
        secret = "customer_secret_token"
        record = DialogueDecisionTrace(
            session_id="s",
            turn=1,
            recognition_source=f"rule_{secret}",
            dialogue_act=f"add_constraint_{secret}",
            fallback_reason=f"request_failed:timeout:{secret}",
            guard_action=f"apply_{secret}",
            guard_reason=f"guard_disabled_{secret}",
            decision_reason=f"highest_dynamic_utility_{secret}",
        )
        recorder = DecisionTraceRecorder(DecisionTraceConfig(enabled=True))
        recorder.record(record)
        payload = record.to_dict()

        self.assertEqual(payload["recognition_source"], "unknown")
        self.assertEqual(payload["dialogue_act"], "unknown")
        self.assertEqual(payload["fallback_reason"], "request_failed:timeout")
        self.assertEqual(payload["guard_action"], "unknown")
        self.assertEqual(payload["guard_reason"], "unknown")
        self.assertEqual(payload["decision_reason"], "unknown")
        self.assertEqual(recorder.summary()["decision_reasons"], {"unknown": 1})
        self.assertEqual(recorder.summary()["guard_actions"], {"unknown": 1})
        self.assertNotIn(secret, repr(record))
        self.assertNotIn(secret, json.dumps(payload, sort_keys=True))
        self.assertNotIn(secret, json.dumps(recorder.summary(), sort_keys=True))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trace.jsonl"
            recorder.export_jsonl(output)
            self.assertNotIn(secret, output.read_text(encoding="utf-8"))

    def test_trace_hashes_session_omits_unapproved_fields_and_rounds_only_on_export(self) -> None:
        # Break caught: an identifier or unapproved field leaks into an exported trace.
        record = trace("secret-session")
        payload = record.to_dict()
        encoded = json.dumps(payload, allow_nan=False, sort_keys=True)

        self.assertEqual(record.session_hash, trace("secret-session").session_hash)
        self.assertNotEqual(record.session_hash, "secret-session")
        self.assertNotIn("secret-session", encoded)
        self.assertNotIn("raw_llm_response", encoded)
        self.assertEqual(record.attribute_scores["color"]["expected_shrink"], 0.123456789)
        self.assertEqual(payload["attribute_scores"]["color"]["expected_shrink"], 0.123457)
        self.assertEqual(payload["score_summary"]["selected_utility"], 0.0)
        self.assertEqual(payload["missing_rates"]["color"], 0.0)

    def test_trace_defensively_freezes_nested_maps_and_normalizes_state_diffs(self) -> None:
        # Break caught: later code mutates a score map or emits unstable constraint spelling.
        scores = {"color": {"expected_shrink": 0.2}}
        record = DialogueDecisionTrace(
            session_id="s",
            turn=1,
            attribute_scores=scores,
            added_constraints=((" Color ", " Navy Blue ", "soft"),),
        )
        scores["color"]["expected_shrink"] = 0.9

        self.assertEqual(record.attribute_scores["color"]["expected_shrink"], 0.2)
        self.assertEqual(record.added_constraints, (("color", "navy blue", "soft"),))
        with self.assertRaises(TypeError):
            record.attribute_scores["color"]["expected_shrink"] = 0.5
        with self.assertRaises(FrozenInstanceError):
            record.turn = 4

    def test_trace_canonicalizes_constraint_values_and_redacts_product_identifiers(self) -> None:
        # Break caught: trace diffs retain punctuation variants or product IDs.
        record = DialogueDecisionTrace(
            session_id="s",
            turn=1,
            added_constraints=(
                (" Color ", "  Color:   Navy Blue,  ", "HARD"),
                ("asin", "B0123ABC45", "soft"),
                ("brand", "B0123ABC45", "soft"),
                ("customer_secret_token", "private product title", "soft"),
            ),
        )
        encoded = json.dumps(record.to_dict(), sort_keys=True)

        self.assertIn(("color", "navy blue", "hard"), record.added_constraints)
        self.assertIn(("<redacted>", "<redacted>", "soft"), record.added_constraints)
        self.assertNotIn("B0123ABC45", repr(record))
        self.assertNotIn("B0123ABC45", encoded)
        self.assertNotIn("customer_secret_token", repr(record))
        self.assertNotIn("private product title", encoded)

    def test_trace_sanitizes_nonfinite_scalar_numbers_before_json_serialization(self) -> None:
        # Break caught: a numerical failure outside a score map makes JSONL export invalid.
        record = DialogueDecisionTrace(
            session_id="s",
            turn=float("nan"),
            candidate_count=float("inf"),
            intent_version=float("-inf"),
            lookahead_depth=float("nan"),
            prompt_tokens=float("inf"),
            completion_tokens=float("nan"),
        )

        self.assertEqual(record.to_dict()["turn"], 0)
        self.assertEqual(record.to_dict()["candidate_count"], 0)
        self.assertEqual(record.sanitizations, 6)
        json.dumps(record.to_dict(), allow_nan=False)

    def test_trace_drops_freeform_reason_text_that_could_be_an_error_or_response(self) -> None:
        # Break caught: freeform provider errors leak through the reason-code diagnostic fields.
        record = DialogueDecisionTrace(
            session_id="s",
            turn=1,
            fallback_reason="provider timeout: secret response text",
            guard_reason="api error contains private text",
            decision_reason="highest_dynamic_utility",
        )
        encoded = json.dumps(record.to_dict())

        self.assertEqual(record.to_dict()["fallback_reason"], "unknown")
        self.assertEqual(record.to_dict()["guard_reason"], "unknown")
        self.assertIn("highest_dynamic_utility", encoded)
        self.assertNotIn("secret response text", encoded)

    def test_trace_keeps_all_real_policy_reason_codes(self) -> None:
        # Break caught: a real policy outcome is erased as unknown by the closed allowlist.
        for reason in (
            "no_preference_other",
            "maximum_questions_reached",
            "turn_limit_guardrail",
        ):
            with self.subTest(reason=reason):
                record = DialogueDecisionTrace(
                    session_id="s",
                    turn=1,
                    decision_reason=reason,
                )

                self.assertEqual(record.to_dict()["decision_reason"], reason)

    def test_from_turn_computes_asin_differences_before_redacting_deltas(self) -> None:
        # Break caught: redacting before comparison hides an ASIN replacement as no state change.
        before = DialogueState(
            session_id="s",
            user_profile={},
            active_constraints=(
                Constraint(
                    attribute="asin",
                    value="B0123ABC45",
                    polarity=Polarity.INCLUDE,
                    strength=ConstraintStrength.HARD,
                    evidence="private",
                    source_turn=1,
                    tokens=(),
                ),
            ),
        )
        after = DialogueState(
            session_id="s",
            user_profile={},
            active_constraints=(
                Constraint(
                    attribute="asin",
                    value="B0987ZYX65",
                    polarity=Polarity.INCLUDE,
                    strength=ConstraintStrength.HARD,
                    evidence="private",
                    source_turn=2,
                    tokens=(),
                ),
            ),
        )

        record = DialogueDecisionTrace.from_turn(
            before_state=before,
            after_state=after,
            recognition=object(),
            guard_decision=object(),
            candidate_signals=None,
            question_decision=object(),
            attribute_components={},
            recommendation_count=0,
            prompt_tokens=0,
            completion_tokens=0,
            lookahead_depth=1,
        )
        encoded = json.dumps(record.to_dict(), sort_keys=True)

        self.assertEqual(record.added_constraints, (("<redacted>", "<redacted>", "hard"),))
        self.assertEqual(record.removed_constraints, (("<redacted>", "<redacted>", "hard"),))
        self.assertNotIn("B0123ABC45", repr(record))
        self.assertNotIn("B0987ZYX65", encoded)

    def test_trace_from_turn_uses_only_normalized_decision_facts(self) -> None:
        # Break caught: the integration boundary serializes an input message or constraint evidence.
        before = DialogueState(session_id="secret-session", user_profile={})
        after = DialogueState(
            session_id="secret-session",
            user_profile={},
            intent_version=2,
            turn=4,
            active_constraints=(
                Constraint(
                    attribute="color",
                    value="Navy Blue",
                    polarity=Polarity.INCLUDE,
                    strength=ConstraintStrength.HARD,
                    evidence="private user wording",
                    source_turn=4,
                    tokens=("navy", "blue"),
                ),
            ),
        )
        recognition = RecognitionResult(
            dialogue_act=DialogueAct.ADD_CONSTRAINT,
            category=None,
            constraint_operations=(
                ConstraintOperation(
                    operation=OperationKind.ADD,
                    attribute="color",
                    value="Navy Blue",
                    polarity=Polarity.INCLUDE,
                    strength=ConstraintStrength.HARD,
                    evidence="private user wording",
                    confidence=0.9,
                ),
            ),
            explicit_rejected_asins=(),
            confidence=0.9,
            source=RecognitionSource.RULE,
            ambiguities=("size",),
        )
        guard = GuardDecision(GuardAction.APPLY, recognition, "guard_disabled")
        signal = CandidateAttributeSignal(
            attribute="color",
            coverage=0.8,
            expected_remaining=4.0,
            expected_shrink=0.7,
            resolve_at_10=0.9,
            resolve_at_3=0.5,
            resolve_at_1=0.2,
            p90_remaining=8.0,
            worst_case_remaining=10,
            missing_rate=0.2,
            extraction_confidence=0.8,
        )
        candidate_signals = CandidateQuestionSignals(
            candidate_count=12,
            by_attribute={"color": signal},
            target_probabilities={"product-id": 1.0},
        )
        decision = QuestionDecision(True, "color", "highest_dynamic_utility", 0.8, {"color": 0.8})

        record = DialogueDecisionTrace.from_turn(
            before_state=before,
            after_state=after,
            recognition=recognition,
            guard_decision=guard,
            candidate_signals=candidate_signals,
            question_decision=decision,
            attribute_components={
                "color": {"exploration_gain": 0.6, "finish_gain": 0.2, "utility": 0.8}
            },
            recommendation_count=10,
            prompt_tokens=11,
            completion_tokens=5,
            lookahead_depth=1,
        )
        encoded = json.dumps(record.to_dict(), sort_keys=True)

        self.assertEqual(record.added_constraints, (("color", "navy blue", "hard"),))
        self.assertEqual(record.attribute_scores["color"]["resolve_at_10"], 0.9)
        self.assertEqual(record.attribute_scores["color"]["utility"], 0.8)
        self.assertNotIn("secret-session", encoded)
        self.assertNotIn("private user wording", encoded)
        self.assertNotIn("product-id", encoded)

    def test_disabled_recorder_is_a_noop(self) -> None:
        # Break caught: disabled local diagnostics still retain session-level information.
        recorder = DecisionTraceRecorder(DecisionTraceConfig(enabled=False))
        recorder.record(trace())

        self.assertEqual(recorder.records(), ())
        self.assertEqual(
            recorder.summary(),
            {
                "enabled": False,
                "recorded": 0,
                "total_seen": 0,
                "decision_reasons": {},
                "guard_actions": {},
                "sanitizations": 0,
            },
        )

    def test_disabled_export_preserves_existing_file_and_zero_cap_keeps_aggregates(self) -> None:
        # Break caught: disabled export truncates files or a zero cap drops aggregate accounting.
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trace.jsonl"
            output.write_text("keep me", encoding="utf-8")
            DecisionTraceRecorder(DecisionTraceConfig(enabled=False)).export_jsonl(output)
            self.assertEqual(output.read_text(encoding="utf-8"), "keep me")

        recorder = DecisionTraceRecorder(DecisionTraceConfig(enabled=True, max_traces=0))
        recorder.record(trace(decision_reason="ask_other_first", guard_action="apply"))
        self.assertEqual(recorder.records(), ())
        self.assertEqual(recorder.summary()["total_seen"], 1)
        self.assertEqual(recorder.summary()["decision_reasons"], {"ask_other_first": 1})

    def test_export_refuses_missing_parent_without_creating_it(self) -> None:
        # Break caught: a malformed path silently creates an unexpected diagnostics directory.
        with tempfile.TemporaryDirectory() as directory:
            missing_parent = Path(directory) / "missing"
            output = missing_parent / "trace.jsonl"
            recorder = DecisionTraceRecorder(DecisionTraceConfig(enabled=True))
            recorder.record(trace(decision_reason="ask_other_first"))

            with self.assertRaises(FileNotFoundError):
                recorder.export_jsonl(output)

            self.assertFalse(missing_parent.exists())

    def test_cap_preserves_aggregate_counters_and_export_is_deterministic_jsonl(self) -> None:
        # Break caught: capped traces lose counts or output non-deterministic JSON.
        recorder = DecisionTraceRecorder(DecisionTraceConfig(enabled=True, max_traces=2))
        recorder.record(trace("a", decision_reason="ask_other_first", guard_action="apply"))
        recorder.record(
            trace("b", decision_reason="highest_dynamic_utility", guard_action="clarify")
        )
        recorder.record(trace("c", decision_reason="ask_other_first", guard_action="apply"))

        self.assertEqual(len(recorder.records()), 2)
        self.assertEqual(
            recorder.summary(),
            {
                "enabled": True,
                "recorded": 2,
                "total_seen": 3,
                "decision_reasons": {"ask_other_first": 2, "highest_dynamic_utility": 1},
                "guard_actions": {"apply": 2, "clarify": 1},
                "sanitizations": 6,
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trace.jsonl"
            recorder.export_jsonl(output)
            contents = output.read_text(encoding="utf-8")
            lines = contents.splitlines()

        self.assertEqual(len(lines), 2)
        self.assertTrue(
            all(
                math.isfinite(value)
                for line in lines
                for value in json.loads(line).values()
                if isinstance(value, float)
            )
        )
        self.assertEqual(lines, sorted(lines, key=lambda line: json.loads(line)["turn"]))

    def test_export_orders_multiple_turns_deterministically(self) -> None:
        # Break caught: equivalent multi-turn traces have order dependent JSONL output.
        recorder = DecisionTraceRecorder(DecisionTraceConfig(enabled=True))
        recorder.record(
            DialogueDecisionTrace(
                session_id="session-a",
                turn=2,
                decision_reason="ask_other_first",
                guard_action="apply",
            )
        )
        recorder.record(
            DialogueDecisionTrace(
                session_id="session-a",
                turn=1,
                decision_reason="ask_other_first",
                guard_action="apply",
            )
        )
        recorder.record(
            DialogueDecisionTrace(
                session_id="session-a",
                turn=1,
                decision_reason="highest_dynamic_utility",
                guard_action="apply",
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trace.jsonl"
            recorder.export_jsonl(output)
            turns = [json.loads(line)["turn"] for line in output.read_text().splitlines()]

        self.assertEqual(turns, [1, 1, 2])

    def test_export_is_identical_for_differently_ordered_recorders_including_ties(self) -> None:
        # Break caught: record insertion order affects a deterministic diagnostics artifact.
        traces = (
            DialogueDecisionTrace(
                session_id="session-a",
                turn=1,
                decision_reason="ask_other_first",
                guard_action="apply",
            ),
            DialogueDecisionTrace(
                session_id="session-a",
                turn=1,
                decision_reason="highest_dynamic_utility",
                guard_action="apply",
            ),
            DialogueDecisionTrace(
                session_id="session-b",
                turn=1,
                decision_reason="no_preference_other",
                guard_action="clarify",
            ),
        )
        first = DecisionTraceRecorder(DecisionTraceConfig(enabled=True))
        second = DecisionTraceRecorder(DecisionTraceConfig(enabled=True))
        for record in traces:
            first.record(record)
        for record in reversed(traces):
            second.record(record)
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.jsonl"
            second_path = Path(directory) / "second.jsonl"
            first.export_jsonl(first_path)
            second.export_jsonl(second_path)

            self.assertEqual(
                first_path.read_text(encoding="utf-8"), second_path.read_text(encoding="utf-8")
            )


if __name__ == "__main__":
    unittest.main()
