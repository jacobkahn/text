#!/usr/bin/env python3
"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT-style license found in the
LICENSE file in the root directory of this source tree.
"""

import logging

from .flashlight_lib_text_decoder import (
    CriterionType,
    DecodeResult,
    EmittingModelState,
    LexiconDecoder,
    LexiconDecoderOptions,
    LexiconFreeDecoder,
    LexiconFreeDecoderOptions,
    LexiconSeq2SeqDecoder,
    LexiconSeq2SeqDecoderOptions,
    LexiconFreeSeq2SeqDecoder,
    LexiconFreeSeq2SeqDecoderOptions,
    LM,
    LMState,
    SmearingMode,
    Trie,
    TrieNode,
    ZeroLM,
    create_emitting_model_state,
    get_obj_from_emitting_model_state,
)

try:
    from .flashlight_lib_text_decoder import KenLM
except ImportError:
    logging.info("Flashlight Text Python bindings installed without KenLM.")
