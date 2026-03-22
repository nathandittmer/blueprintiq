from __future__ import annotations

import json
from pathlib import Path

import typer

from blueprintiq.service.model import ModelService

app = typer.Typer(help="BlueprintIQ batch inference CLI")


@app.command()
def run(
    input_dir: str = typer.Option(..., "--input-dir", help="Directory containing input images"),
    output_path: str = typer.Option(
        "run/batch_predictions/predictions.json",
        "--output-path",
        help="Path to save JSON results",
    ),
    score_threshold: float | None = typer.Option(
        None,
        "--score-threshold",
        help="Optional override for score threshold",
    ),
) -> None:
    input_path = Path(input_dir)

    if not input_path.exists():
        raise typer.BadParameter(f"Input directory not found: {input_dir}")

    if not input_path.is_dir():
        raise typer.BadParameter(f"Input path is not a directory: {input_dir}")

    service = ModelService()

    image_paths = sorted(input_path.glob("*.png"))
    if not image_paths:
        raise typer.BadParameter(f"No PNG files found in: {input_dir}")

    results = []
    for image_path in image_paths:
        result = service.predict(
            image_path=str(image_path),
            score_threshold=score_threshold,
        )
        results.append(result)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    typer.echo(f"Wrote {len(results)} predictions to {out_path}")


if __name__ == "__main__":
    app()