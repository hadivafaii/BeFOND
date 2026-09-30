"""Data streams yield ordinary [batch, input_dimension] tensors."""

def make_source(config, device="cpu"):
    """Construct the selected benchmark lazily; core imports need no LLM packages."""
    data = config["data"]
    if data["kind"] in ("synthetic", "superposition"):
        from .synthetic import SyntheticSource
        return SyntheticSource(data, device=device, input_dim=config["model"]["input_dim"])
    if data["kind"] == "gemma":
        from .gemma import CachedActivationStream
        return CachedActivationStream(data, device=device)
    if data["kind"] == "toy":
        from .toy import ToySource
        return ToySource(config, device=device)
    raise ValueError(f"Unknown data kind: {data['kind']}")
