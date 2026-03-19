from __future__ import annotations

import json
from pathlib import Path

def main(log_path: str = "runs/monitoring/predictions.jsonl", low_conf_threshold: float = 0.3):
    path = Path(log_path)

    if not path.exists():
        print(f"No log file found at: {path}")
        return

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        print("No prediction records found.")
        return

    total_predictions = len(records)
    scores = [float(r.get("score", 0.0)) for r in records]
    avg_confidence = sum(scores) / total_predictions
    low_conf_count = sum(1 for s in scores if s < low_conf_threshold)
    low_conf_rate = low_conf_count / total_predictions

    print("===Monitoring Summary ===")
    print(f"total_predictions={total_predictions}")
    print(f"average_confidence={avg_confidence:.4f}")
    print(f"low_conf_threshold={low_conf_threshold}")
    print(f"low_conf_predictions={low_conf_count}")
    print(f"low_conf_rate={low_conf_rate:.2%}")


if __name__ == "__main__":
    main()