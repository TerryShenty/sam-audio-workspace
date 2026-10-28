"""Fail closed on unverified bases and deltas from the broken legacy loader."""


def verified_base(model):
    report = getattr(model, "_pretrained_load_report", None)
    if not report or report.get("version") != 2 or not report.get("verified") or not report.get("strict"):
        raise RuntimeError("A verified strict pretrained load is required; legacy caches/deltas are invalid")
    return {
        **({"conversion": model._target_only_conversion} if hasattr(model, "_target_only_conversion") else {}),
        **{key: report[key] for key in ("version", "checkpoint_sha256", "config_sha256")},
        "text_encoder_identity": report.get("text_encoder_identity", {}),
    }


def require_matching_base(checkpoint, model):
    actual = verified_base(model)
    if checkpoint.get("verified_base") != actual:
        raise ValueError("Delta/resume base mismatch or legacy unverified checkpoint; start from a verified pretrained base")
