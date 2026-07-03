#!/usr/bin/env python3
"""Generate a deterministic synthetic dataset for Theme D."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Tuple


def normalize_name(name: str) -> str:
    """Normalize the chef name and remove accents and non-letter characters.

    This function produces a reproducible string that is used to generate the seed.
    """
    normalized = unicodedata.normalize("NFKD", name)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    # Keep only uppercase letters to avoid spaces and punctuation.
    return re.sub(r"[^A-Z]", "", ascii_text.upper())


def build_seed(chef_name: str) -> int:
    """Convert the normalized chef name into a deterministic numeric seed."""
    normalized = normalize_name(chef_name)
    if not normalized:
        raise ValueError("Le nom du chef de groupe ne peut pas être vide.")

    seed = 0
    for index, char in enumerate(normalized):
        # Use a simple rolling hash based on character codes and position.
        seed = (seed * 31 + ord(char) + index) % 2_147_483_647
    # Add a fixed offset to avoid trivial small seeds.
    return seed + 10_000_003


def clip(value: float, minimum: float, maximum: float) -> float:
    """Clip a numeric value between a minimum and maximum bound."""
    return max(minimum, min(maximum, value))


def generate_dataset(chef_name: str, count: int = 250) -> Tuple[List[Dict[str, object]], int]:
    """Generate a deterministic dataset of student records.

    The dataset uses a seeded random generator to ensure reproducibility.
    """
    seed = build_seed(chef_name)
    rng = random.Random(seed)
    rows: List[Dict[str, object]] = []

    for student_id in range(1, count + 1):
        # Create a performance profile with three bands of student ability.
        profile = rng.random()
        if profile < 0.25:
            base_score = rng.uniform(42, 58)
        elif profile < 0.70:
            base_score = rng.uniform(58, 78)
        else:
            base_score = rng.uniform(78, 95)

        # Generate score around the base score with realistic variability.
        score_eval = round(clip(base_score + rng.uniform(-8, 8), 0, 100), 1)
        # Derive time spent studying from the score with noise.
        temps_etude = round(clip(4.5 + 0.16 * score_eval + rng.uniform(-2.2, 2.2), 2.0, 25.0), 1)
        # Derive attendance rate from the score with noise.
        assiduite = round(clip(36 + 0.45 * score_eval + rng.uniform(-7.0, 7.0), 0.0, 100.0), 1)

        # Create a synthetic orientation decision using a weighted signal.
        signal = score_eval * 0.65 + assiduite * 0.25 + temps_etude * 0.3
        if score_eval >= 82 and assiduite >= 80 and temps_etude >= 13:
            orientation = "scientifique"
        elif score_eval <= 46 or assiduite <= 34:
            orientation = "littéraire"
        else:
            orientation = "scientifique" if signal > 103 else "littéraire"

        rows.append(
            {
                "id": student_id,
                "score_eval": score_eval,
                "temps_etude_heures": temps_etude,
                "assiduite_pct": assiduite,
                "orientation_recommandee": orientation,
            }
        )

    return rows, seed


def write_dataset(rows: List[Dict[str, object]], output_path: Path, chef_name: str, seed: int) -> None:
    """Write the dataset CSV and a metadata JSON side by side."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "score_eval", "temps_etude_heures", "assiduite_pct", "orientation_recommandee"],
        )
        writer.writeheader()
        writer.writerows(rows)

    metadata_path = output_path.with_suffix(".meta.json")
    metadata = {
        "chef_groupe": chef_name,
        "nom_normalise": normalize_name(chef_name),
        "graine": seed,
        "nombre_eleves": len(rows),
        "colonnes": ["id", "score_eval", "temps_etude_heures", "assiduite_pct", "orientation_recommandee"],
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def main() -> None:
    """Parse command-line arguments and generate the dataset."""
    parser = argparse.ArgumentParser(description="Génère un dataset synthétique pour le thème D")
    parser.add_argument("--chef", required=True, help="Nom complet du chef de groupe")
    parser.add_argument("--output", default="data/dataset.csv", help="Chemin du fichier de sortie")
    parser.add_argument("--count", type=int, default=250, help="Nombre d'élèves à générer")
    args = parser.parse_args()

    rows, seed = generate_dataset(args.chef, args.count)
    output_path = Path(args.output)
    write_dataset(rows, output_path, args.chef, seed)

    print(f"Dataset généré : {output_path}")
    print(f"Graine : {seed}")
    print("Extrait :")
    for row in rows[:3]:
        print(row)


if __name__ == "__main__":
    main()
