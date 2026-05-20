import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.qualitative_clustering import load_feedback, run_analysis


def write_sample_csv(path: Path) -> None:
    rows = [
        {"Date": "1/1/2026", "Rating": 4, "Comments": "The guide was friendly and answered every question.", "Phone": "555", "Email": "a@example.com", "What type of tour did you take?": "Weekday campus Tour"},
        {"Date": "1/2/2026", "Rating": 4, "Comments": "Our tour guide made everyone feel welcome and informed.", "Phone": "555", "Email": "b@example.com", "What type of tour did you take?": "Weekday campus Tour"},
        {"Date": "1/3/2026", "Rating": 2, "Comments": "Parking directions were confusing and hard to follow.", "Phone": "555", "Email": "c@example.com", "What type of tour did you take?": "Large group Tour"},
        {"Date": "1/4/2026", "Rating": 2, "Comments": "We had trouble finding parking and the meeting location.", "Phone": "555", "Email": "d@example.com", "What type of tour did you take?": "Large group Tour"},
        {"Date": "1/5/2026", "Rating": 3, "Comments": "N/A", "Phone": "555", "Email": "e@example.com", "What type of tour did you take?": "Weekday campus Tour"},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


class QualitativeClusteringTests(unittest.TestCase):
    def test_load_feedback_filters_placeholder_comments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "feedback.csv"
            write_sample_csv(input_path)

            df = load_feedback(input_path)

        self.assertEqual(len(df), 4)
        self.assertIn("clean_comment", df.columns)

    def test_run_analysis_writes_expected_artifacts_without_contact_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "feedback.csv"
            output_dir = tmp_path / "outputs"
            write_sample_csv(input_path)

            selection = run_analysis(input_path, output_dir, min_clusters=2, max_clusters=2, random_state=7)

            self.assertEqual(selection.cluster_count, 2)
            self.assertTrue((output_dir / "cluster_summary.csv").exists())
            self.assertTrue((output_dir / "cluster_terms.csv").exists())
            self.assertTrue((output_dir / "clustered_feedback.csv").exists())
            self.assertTrue((output_dir / "metrics.json").exists())

            clustered = pd.read_csv(output_dir / "clustered_feedback.csv")
            self.assertNotIn("Phone", clustered.columns)
            self.assertNotIn("Email", clustered.columns)
            self.assertIn("cluster", clustered.columns)


if __name__ == "__main__":
    unittest.main()
