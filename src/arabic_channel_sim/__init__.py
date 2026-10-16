# Copyright 2026 Bayanat Labs
# SPDX-License-Identifier: Apache-2.0
"""arabic-channel-sim: render clean speech under fixed, documented channel conditions.

Conditions (telephone band-limit, resampling, codecs, synthetic room reverberation) are defined in a
versioned YAML file; every render is a deterministic function of the parent audio, the condition id and
the parent id.
"""
from ._version import __version__
from .metrics import clipped_fraction, estoi, si_sdr
from .pipeline import (DEFAULT_CONDITIONS, conditions_sha256, decode_audio, encode_flac, load_conditions, render,
                     variant_seed)

__all__ = ["__version__", "render", "load_conditions", "conditions_sha256", "variant_seed", "encode_flac",
           "decode_audio", "si_sdr", "estoi", "clipped_fraction", "DEFAULT_CONDITIONS"]
