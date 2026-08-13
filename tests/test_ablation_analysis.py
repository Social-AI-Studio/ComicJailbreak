from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from analysis.ablation import (
    STRUCTURE_CONDITIONS,
    TEXT_CONDITIONS,
    VISUAL_CONDITIONS,
    contrast_weights,
    ensemble_metrics,
    fixed_denominator_metrics,
    paired_bootstrap_contrast,
    visual_factorial_effects,
    visual_structure_effects,
)


class AblationAnalysisTests(unittest.TestCase):
    def test_main_effect_weights(self) -> None:
        weights = contrast_weights(list(TEXT_CONDITIONS), "N")
        self.assertAlmostEqual(weights.sum(), 0.0)
        self.assertAlmostEqual(weights[4:].sum(), 1.0)
        self.assertAlmostEqual(weights[:4].sum(), -1.0)

    def test_known_narrative_effect_and_bootstrap(self) -> None:
        rows = []
        for seed in range(20):
            row = {}
            for condition_id in TEXT_CONDITIONS:
                row[condition_id] = 1.0 if "_n1_" in condition_id else 0.0
            rows.append(row)
        matrix = pd.DataFrame(rows)
        result = paired_bootstrap_contrast(matrix, "N", samples=200, seed=7)
        self.assertAlmostEqual(result["Estimate"], 1.0)
        self.assertAlmostEqual(result["CI_Lower"], 1.0)
        self.assertAlmostEqual(result["CI_Upper"], 1.0)

    def test_fixed_denominators_include_incompatible_slots(self) -> None:
        scored = pd.DataFrame(
            [
                {
                    "Model": "m",
                    "Experiment": "text_factorial",
                    "Condition": "text_n0_r0_c0",
                    "Template": "code",
                    "Is_Harmful": True,
                    "Input_Available": True,
                    "Attack_Success": 1,
                    "Refusal": 0,
                    "Seed_Index": 0,
                },
                {
                    "Model": "m",
                    "Experiment": "text_factorial",
                    "Condition": "text_n0_r0_c0",
                    "Template": "code",
                    "Is_Harmful": True,
                    "Input_Available": False,
                    "Attack_Success": 0,
                    "Refusal": 0,
                    "Seed_Index": 1,
                },
            ]
        )
        metrics = fixed_denominator_metrics(scored)
        harmful = metrics[metrics["Metric"] == "ASR"].iloc[0]
        self.assertEqual(harmful["Denominator"], 200)
        self.assertEqual(harmful["Positive"], 1)
        self.assertAlmostEqual(harmful["Rate"], 0.005)

    def test_ensemble_is_any_success_across_templates(self) -> None:
        scored = pd.DataFrame(
            [
                {
                    "Model": "m",
                    "Experiment": "text_factorial",
                    "Condition": "text_n0_r0_c0",
                    "Template": template,
                    "Is_Harmful": True,
                    "Attack_Success": int(template == "article"),
                    "Refusal": 0,
                    "Seed_Index": 0,
                }
                for template in ("article", "code")
            ]
        )
        metric = ensemble_metrics(scored).iloc[0]
        self.assertEqual(metric["Metric"], "EASR")
        self.assertEqual(metric["Positive"], 1)
        self.assertEqual(metric["Denominator"], 200)

    def test_visual_effect_exports_paired_contrasts(self) -> None:
        rows = []
        for seed in range(4):
            for condition in VISUAL_CONDITIONS:
                rows.append(
                    {
                        "Model": "m",
                        "Experiment": "visual_factorial",
                        "Condition": condition,
                        "Template": "article",
                        "Seed_Index": seed,
                        "Is_Harmful": True,
                        "Outcome": int("_r1_" in condition),
                    }
                )
            for condition in STRUCTURE_CONDITIONS:
                rows.append(
                    {
                        "Model": "m",
                        "Experiment": "visual_structure",
                        "Condition": condition,
                        "Template": "article",
                        "Seed_Index": seed,
                        "Is_Harmful": True,
                        "Outcome": int(condition == "structure_reversed"),
                    }
                )
        scored = pd.DataFrame(rows)
        visual = visual_factorial_effects(scored, bootstrap_samples=50)
        role = visual[
            (visual["Model"] == "m")
            & (visual["Endpoint"] == "ensemble")
            & (visual["Contrast"] == "R")
        ].iloc[0]
        self.assertEqual(role["Estimate"], 1.0)
        structure = visual_structure_effects(scored, bootstrap_samples=50)
        reversed_effect = structure[
            (structure["Model"] == "m")
            & (structure["Endpoint"] == "ensemble")
            & (structure["Contrast"] == "structure_reversed-original")
        ].iloc[0]
        self.assertEqual(reversed_effect["Estimate"], 1.0)


if __name__ == "__main__":
    unittest.main()
