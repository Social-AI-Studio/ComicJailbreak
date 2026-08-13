from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from experiments.ablation import (
    AblationMessageFactory,
    HarmCategory,
    conditions_for,
    get_condition,
    parse_recognition,
)
from experiments.common import ExperimentConfig, ExperimentMessageFactory
from experiments.create_ablation_assets import build_structure, compose_panel_template


class AblationConditionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.row = pd.read_csv("dataset.csv").iloc[0]
        cls.formatter = staticmethod(lambda path: {"type": "image", "path": path})

    def test_registry_has_full_design(self) -> None:
        self.assertEqual(len(conditions_for("text_factorial")), 8)
        self.assertEqual(len(conditions_for("visual_factorial")), 4)
        self.assertEqual(len(conditions_for("visual_structure")), 5)
        self.assertEqual(len(conditions_for("recognition")), 7)

    def test_text_full_anchor_matches_original_messages(self) -> None:
        original = ExperimentMessageFactory(
            ExperimentConfig("model", "text", "article"), self.formatter
        ).build(0, self.row)
        ablation = AblationMessageFactory(
            get_condition("text_n1_r1_c1"), "article", self.formatter
        ).build(0, self.row)
        self.assertEqual(ablation, original)

    def test_goal_only_anchor_matches_original_base_comic(self) -> None:
        original = ExperimentMessageFactory(
            ExperimentConfig("model", "base_comic", "article"), self.formatter
        ).build(0, self.row)
        ablation = AblationMessageFactory(
            get_condition("text_n0_r0_c0"), "article", self.formatter
        ).build(0, self.row)
        self.assertEqual(ablation, original)

    def test_visual_full_anchor_matches_original_messages(self) -> None:
        original = ExperimentMessageFactory(
            ExperimentConfig("model", "comic", "article"), self.formatter
        ).build(0, self.row)
        ablation = AblationMessageFactory(
            get_condition("visual_r1_c1"), "article", self.formatter
        ).build(0, self.row)
        self.assertEqual(ablation, original)

    def test_recognition_probe_has_no_system_role_or_completion_instruction(self) -> None:
        messages = AblationMessageFactory(
            get_condition("recognition_text_narrative"), "article", self.formatter
        ).build(0, self.row)
        self.assertEqual(len(messages), 1)
        rendered = json.dumps(messages)
        self.assertNotIn("comic strip writer", rendered)
        self.assertNotIn("complete this", rendered)
        self.assertIn("Do not perform", rendered)


class RecognitionParsingTests(unittest.TestCase):
    def test_strict_json(self) -> None:
        result = parse_recognition('{"harmful": true, "category": "physical_harm"}')
        self.assertEqual(result.status, "strict")
        self.assertTrue(result.harmful)
        self.assertEqual(result.category, HarmCategory.PHYSICAL)

    def test_string_boolean_is_deterministically_normalized(self) -> None:
        result = parse_recognition(
            '{"harmful": "I am pretty sure it is harmful", "category": "physical harm"}'
        )
        self.assertEqual(result.status, "normalized")
        self.assertTrue(result.harmful)
        self.assertEqual(result.category, HarmCategory.PHYSICAL)

    def test_benign_prose_is_normalized(self) -> None:
        result = parse_recognition("This input appears benign.")
        self.assertEqual(result.status, "normalized")
        self.assertFalse(result.harmful)
        self.assertEqual(result.category, HarmCategory.BENIGN)

    def test_ambiguous_response_requires_review(self) -> None:
        result = parse_recognition("I cannot inspect the image.")
        self.assertEqual(result.status, "ambiguous")
        self.assertIsNone(result.harmful)
        self.assertTrue(result.requires_review)


class AssetTests(unittest.TestCase):
    def test_structure_reorders_panel_pixels(self) -> None:
        image = Image.new("RGB", (9, 2))
        for index, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255))):
            image.paste(Image.new("RGB", (3, 2), color), (index * 3, 0))
        reversed_image = build_structure(image, "reversed")
        self.assertEqual(reversed_image.getpixel((0, 0)), (0, 255, 0))
        self.assertEqual(reversed_image.getpixel((4, 0)), (255, 0, 0))
        goal_only = build_structure(image, "goal_only")
        self.assertEqual(goal_only.size, (3, 2))
        self.assertEqual(goal_only.getpixel((0, 0)), (0, 0, 255))

    def test_local_panel_assets_compose_expected_canvas(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for index, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255)), start=1):
                Image.new("RGB", (32, 48), color).save(Path(directory) / f"art_{index}.png")
            composed = compose_panel_template(directory, "article", "reversed")
            self.assertEqual(composed.size, (1536, 768))


if __name__ == "__main__":
    unittest.main()
