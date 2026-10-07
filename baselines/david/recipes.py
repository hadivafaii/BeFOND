"""The full-width baseline recipe registry; all overrides are relative to BASE."""

D_IN, D_SAE = 768, 16384

BASE = dict(
    arch='btk',                 # btk | mat | jr
    autocast=True,              # upstream: bf16 autocast for the SAE and the data generator
    lr_decay=0.0,               # fraction of training over which the LR decays linearly to 0 (upstream 0)
    # BatchTopK / Matryoshka
    aux_coef=1.0,               # AuxK coefficient (SAELens default 1)
    dead_window=1000,           # steps without firing before a latent counts as dead (SAELens default 1000)
    mat_widths=(128, 512, 2048),  # Matryoshka prefixes; the full width is appended
    mat_aux=True,               # upstream: per-level Matryoshka aux loss
    gate='plain',               # plain | sinkhorn | minfire
    gate_until=0.0,             # fraction of training during which the gate is active
    floor_frac=1.0,             # Sinkhorn: floor as a multiple of the fair share batch*k/n_latents
    min_fires=1, minfire_every='auto',  # MinFire: forced firings per latent, and the round-robin period
                                #   ('auto': smallest period keeping forced firings <= 70% of the k budget)
    # dead-latent resampling (any arch): every N steps re-initialise latents dead for > dead_window steps
    resample_every=0, resample_until=0.8,
    # JumpReLU
    bw=1.0, bw_end=None, bw_until=0.8,   # bandwidth (absolute, normalised-input units); optional anneal
    ste=False,                  # straight-through gradient to the pre-activations (Anthropic) or not (GDM)
    penalty='step',             # step (GDM L0) | tanh (Anthropic)
    preact=None,                # Anthropic pre-activation loss coefficient on dead latents
    init_threshold=1.0, calibrate_threshold=False,  # or set the threshold so the initial L0 is k
    tuner_gain=3e-4,            # autotuner integral gain (upstream 3e-4): max relative coefficient change per step
    l0_coef=1.0,                # base L0 coefficient the autotuner multiplies (upstream 1.0)
    revive=False,               # JumpReLU: lower a dead latent's threshold to 0.9x its best batch pre-activation
)

RECIPES = {
    # ---------------- BatchTopK
    'btk': dict(arch='btk'),
    'btk_fp32': dict(arch='btk', autocast=False),
    'btk_win100': dict(arch='btk', dead_window=100),
    'btk_aux4': dict(arch='btk', aux_coef=4.0),
    'btk_aux16': dict(arch='btk', aux_coef=16.0),
    'btk_win100_aux4': dict(arch='btk', dead_window=100, aux_coef=4.0),
    'btk_sink25': dict(arch='btk', gate='sinkhorn', gate_until=0.25, floor_frac=1.0),
    'btk_sink25_f5': dict(arch='btk', gate='sinkhorn', gate_until=0.25, floor_frac=5.0),
    'btk_sink90': dict(arch='btk', gate='sinkhorn', gate_until=0.9, floor_frac=1.0),
    'btk_minfire': dict(arch='btk', gate='minfire', gate_until=0.9, minfire_every=1),
    'btk_minfire8': dict(arch='btk', gate='minfire', gate_until=0.9, minfire_every=8),
    'btk_resample': dict(arch='btk', resample_every=2500),
    'btk_decay': dict(arch='btk', lr_decay=0.2),
    # ---------------- Matryoshka (BatchTopK)
    'mat': dict(arch='mat'),
    'mat_fp32': dict(arch='mat', autocast=False),
    'mat_noaux': dict(arch='mat', mat_aux=False),
    'mat_w8k': dict(arch='mat', mat_widths=(128, 512, 2048, 8192)),
    'mat_hier': dict(arch='mat', mat_widths=(128, 640, 2688)),
    'mat_win100': dict(arch='mat', dead_window=100),
    'mat_aux4': dict(arch='mat', aux_coef=4.0),
    'mat_sink25': dict(arch='mat', gate='sinkhorn', gate_until=0.25, floor_frac=1.0),
    'mat_minfire8': dict(arch='mat', gate='minfire', gate_until=0.9, minfire_every=8),
    'mat_resample': dict(arch='mat', resample_every=2500),
    'mat_decay': dict(arch='mat', lr_decay=0.2),
    # ---------------- JumpReLU
    'jr': dict(arch='jr'),
    'jr_fp32': dict(arch='jr', autocast=False),
    'jr_bw3': dict(arch='jr', bw=3.0),
    'jr_bw10': dict(arch='jr', bw=10.0),
    'jr_anneal': dict(arch='jr', bw=10.0, bw_end=1.0),
    'jr_ste': dict(arch='jr', ste=True),
    'jr_ste_bw3': dict(arch='jr', ste=True, bw=3.0),
    'jr_tanh': dict(arch='jr', ste=True, penalty='tanh'),
    'jr_tanh_bw3': dict(arch='jr', ste=True, penalty='tanh', bw=3.0),
    'jr_anth': dict(arch='jr', ste=True, penalty='tanh', preact=3e-6),
    'jr_anth_bw3': dict(arch='jr', ste=True, penalty='tanh', preact=3e-6, bw=3.0),
    'jr_preact': dict(arch='jr', preact=3e-4),
    'jr_calib': dict(arch='jr', calibrate_threshold=True),
    'jr_resample': dict(arch='jr', resample_every=2500),
    'jr_decay': dict(arch='jr', lr_decay=0.2),
    'jr_fast': dict(arch='jr', tuner_gain=3e-3),
    'jr_tanh_bw3_fast': dict(arch='jr', ste=True, penalty='tanh', bw=3.0, tuner_gain=3e-3),
    # ---------------- round 2: combinations of the round-1 winners
    'btk_minfire2': dict(arch='btk', gate='minfire', gate_until=0.9, min_fires=2),
    'btk_minfire_u75': dict(arch='btk', gate='minfire', gate_until=0.75),
    'btk_minfire_u97': dict(arch='btk', gate='minfire', gate_until=0.97),
    'btk_minfire_aux16': dict(arch='btk', gate='minfire', gate_until=0.9, aux_coef=16.0),
    'btk_minfire_resample': dict(arch='btk', gate='minfire', gate_until=0.9, resample_every=2500),
    'btk_minfire_decay': dict(arch='btk', gate='minfire', gate_until=0.9, lr_decay=0.2),
    'mat_minfire': dict(arch='mat', gate='minfire', gate_until=0.9),
    'mat_minfire2': dict(arch='mat', gate='minfire', gate_until=0.9, min_fires=2),
    'mat_minfire_u97': dict(arch='mat', gate='minfire', gate_until=0.97),
    'mat_minfire_aux16': dict(arch='mat', gate='minfire', gate_until=0.9, aux_coef=16.0),
    'mat_minfire_resample': dict(arch='mat', gate='minfire', gate_until=0.9, resample_every=2500),
    'mat_minfire_decay': dict(arch='mat', gate='minfire', gate_until=0.9, lr_decay=0.2),
    'mat_resample_r1000': dict(arch='mat', resample_every=1000),
    'mat_resample_u95': dict(arch='mat', resample_every=2500, resample_until=0.95),
    'mat_resample_win100': dict(arch='mat', resample_every=2500, dead_window=100),
    'jr_ste_resample': dict(arch='jr', ste=True, resample_every=2500),
    'jr_ste_resample_u95': dict(arch='jr', ste=True, resample_every=2500, resample_until=0.95),
    'jr_ste_resample_decay': dict(arch='jr', ste=True, resample_every=2500, lr_decay=0.2),
    'jr_ste_resample_r1000': dict(arch='jr', ste=True, resample_every=1000),
    'jr_ste_resample_win100': dict(arch='jr', ste=True, resample_every=2500, dead_window=100),
    'jr_ste_c01': dict(arch='jr', ste=True, l0_coef=0.1),
    'jr_ste_c01_resample': dict(arch='jr', ste=True, l0_coef=0.1, resample_every=2500),
    'jr_ste_bw03': dict(arch='jr', ste=True, bw=0.3),
    'jr_ste_bw01': dict(arch='jr', ste=True, bw=0.1),
    'jr_ste_bw03_resample': dict(arch='jr', ste=True, bw=0.3, resample_every=2500),
    'jr_ste_revive': dict(arch='jr', ste=True, revive=True),
    # ---------------- round 3: refinements of the round-2 winners
    'jr_ste_c003_resample': dict(arch='jr', ste=True, l0_coef=0.03, resample_every=2500),
    'jr_ste_c001_resample': dict(arch='jr', ste=True, l0_coef=0.01, resample_every=2500),
    'jr_ste_c01_resample_decay': dict(arch='jr', ste=True, l0_coef=0.1, resample_every=2500, lr_decay=0.2),
    'jr_ste_c01_resample_u95': dict(arch='jr', ste=True, l0_coef=0.1, resample_every=2500, resample_until=0.95),
    'jr_ste_c01_resample_win100': dict(arch='jr', ste=True, l0_coef=0.1, resample_every=2500, dead_window=100),
    'mat_resample_win100_decay': dict(arch='mat', resample_every=2500, dead_window=100, lr_decay=0.2),
    'mat_resample_win100_u95': dict(arch='mat', resample_every=2500, dead_window=100, resample_until=0.95),
    'mat_resample_win100_r1000': dict(arch='mat', resample_every=1000, dead_window=100),
    'mat_resample_win30': dict(arch='mat', resample_every=2500, dead_window=30),
    'mat_minfire_resample_win100': dict(arch='mat', gate='minfire', gate_until=0.9, resample_every=2500, dead_window=100),
    'btk_resample_win100': dict(arch='btk', resample_every=2500, dead_window=100),
    'btk_minfire_resample_win100': dict(arch='btk', gate='minfire', gate_until=0.9, resample_every=2500, dead_window=100),
    'btk_minfire_decay_u97': dict(arch='btk', gate='minfire', gate_until=0.97, lr_decay=0.2),
    'btk_minfire_decay_aux16': dict(arch='btk', gate='minfire', gate_until=0.9, lr_decay=0.2, aux_coef=16.0),
}
def recipe(name):
    if name not in RECIPES:
        raise KeyError(f'Unknown recipe {name!r}; known: {sorted(RECIPES)}')
    r = dict(BASE, **RECIPES[name])
    r['name'] = name
    return r
