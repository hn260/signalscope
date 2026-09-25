"""
Report generator module.
Exports analysis summaries in JSON and CSV formats with full metadata, method transparency,
and quality warnings.
"""

import json
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Dict, Any

from .metadata import SignalMetadata, ParameterResult, QualityFlag
from . import __version__


def generate_json_report(
    metadata: SignalMetadata,
    parameters: List[ParameterResult],
    quality_flags: List[QualityFlag],
    quality_score: float,
    classification_result: Tuple[str, float, Dict[str, float]]
) -> str:
    """
    Generates a structured JSON analysis report string.
    """
    best_class, class_confidence, class_probs = classification_result

    report_dict: Dict[str, Any] = {
        "signalscope_version": __version__,
        "analysis_timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "honesty_disclaimer": (
            "SignalScope operates under strict scope limits. RF center frequency is relative "
            "to baseband unless provided by metadata. Modulation classifications are statistical "
            "model estimates, not ground truth guarantees."
        ),
        "metadata": {
            "filename": metadata.filename,
            "format": metadata.file_format,
            "file_size_bytes": metadata.file_size_bytes,
            "num_samples": metadata.num_samples,
            "duration_sec": metadata.duration_sec,
            "sample_rate_hz": metadata.sample_rate_hz,
            "center_freq_hz": metadata.center_freq_hz,
            "frequency_is_absolute": metadata.frequency_is_absolute,
            "origin": metadata.metadata_origin.value,
        },
        "quality_assessment": {
            "overall_quality_score_percent": round(quality_score, 1),
            "flags": [
                {
                    "code": f.code,
                    "severity": f.severity.value,
                    "message": f.message,
                    "recommendation": f.recommendation
                }
                for f in quality_flags
            ]
        },
        "extracted_parameters": [
            {
                "name": p.name,
                "display_name": p.display_name,
                "value": p.value,
                "unit": p.unit,
                "method": p.method,
                "status": p.status.value,
                "confidence": round(p.confidence, 2),
                "assumptions": p.assumptions
            }
            for p in parameters
        ],
        "modulation_classification": {
            "predicted_class": best_class,
            "confidence_heuristic": round(class_confidence, 2),
            "class_probabilities": {k: round(v, 4) for k, v in class_probs.items()}
        }
    }

    return json.dumps(report_dict, indent=2)


def generate_csv_table(parameters: List[ParameterResult]) -> str:
    """
    Generates a CSV string of extracted parameters.
    """
    records = [
        {
            "Parameter": p.display_name,
            "Value": p.value,
            "Unit": p.unit,
            "Measurement Method": p.method,
            "Status": p.status.value,
            "Confidence Score": p.confidence,
            "Assumptions & Notes": p.assumptions,
        }
        for p in parameters
    ]

    df = pd.DataFrame(records)
    return df.to_csv(index=False)
