"""Dictionary alignment and streamed feature detection on synthetic data."""

import torch

from .feature_metrics import GroundTruthFeatureMetrics, feature_frequency_reference
from .matching import final_matching_method, match_dictionary_atoms


@torch.no_grad()
def evaluate(model, source, *, samples=500000, steps=20, batch_size=2048,
             beta=1.0, seed=49100, frequency_samples=500000, frequency_seed=315031):
    """Report one-to-one dictionary MCC, all-feature F1, rare-tail F1, and sparsity."""
    truth = source.generator.feature_dict.feature_vectors
    scale = source.config.get("input_scale", 1.0)
    decoder = source.normalizer.decoder_to_raw(model.decoder.detach() / scale).T
    method = final_matching_method(len(decoder))
    _, _, cosines = match_dictionary_atoms(decoder, truth, method, absolute=True)
    frequencies = feature_frequency_reference(source.generator, samples=frequency_samples,
                                               seed=frequency_seed)
    metrics = GroundTruthFeatureMetrics(decoder, truth, frequencies, method)
    active = torch.zeros(model.config.num_latents, device=model.decoder.device, dtype=torch.bool)
    count, total_active, squared_error = 0, 0, 0.0
    for step, start in enumerate(range(0, samples, batch_size)):
        x, ground_truth = source.sample_with_codes(min(batch_size, samples - start), step,
                                                  stream="test", seed=seed)
        means = model.infer(x, steps=steps, beta=beta).mean
        codes = means * (means > 0.5)
        metrics.add_batch(codes, ground_truth)
        active |= (codes != 0).any(0)
        total_active += int((codes != 0).sum())
        reconstruction = means @ model.decoder.T
        if model.bias is not None:
            reconstruction += model.bias
        squared_error += float((x - reconstruction).square().sum())
        count += len(x)
    result = metrics.compute()
    result.update(mcc=float(cosines.mean()),
                  macro_f1=result["gt_f1"], rare_f1=result["rare_f1"],
                  mean_l0=total_active / count, dead_latents=int((~active).sum()),
                  mean_squared_error=squared_error / (count * model.config.input_dim),
                  samples=count, steps=steps, seed=seed, matching=method)
    return result
