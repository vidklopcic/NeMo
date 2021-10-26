# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import torch
from packaging import version

from nemo.utils import logging
from typing import (
    Tuple, Optional, Union, Any, Sequence, TYPE_CHECKING
)

import torch
import torch.nn.functional as F
from torch.types import _size
from torch._lowrank import svd_lowrank, pca_lowrank
from torch.overrides import (
    has_torch_function, has_torch_function_unary, has_torch_function_variadic,
    handle_torch_function)
from torch._jit_internal import boolean_dispatch, List
from torch._jit_internal import _overload as overload
from torch._autograd_functions import _LU

Tensor = torch.Tensor
from torch import _VF

# Library version globals
TORCH_VERSION = None
TORCH_VERSION_MIN = version.Version('1.7')

def stft_patch(
        input: torch.Tensor,
        n_fft: int,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: Optional[torch.Tensor] = None,
        center: bool = True,
        pad_mode: str = 'reflect',
        normalized: bool = False,
        onesided: Optional[bool] = None,
        return_complex: Optional[bool] = None,
):
    """
    Patch over torch.stft for PyTorch <= 1.6.
    Arguments are same as torch.stft().

    # TODO: Remove once PyTorch 1.7+ is a requirement.
    """
    global TORCH_VERSION
    if TORCH_VERSION is None:
        TORCH_VERSION = version.parse(torch.__version__)

        logging.warning(
            "torch.stft() signature has been updated for PyTorch 1.7+\n"
            "Please update PyTorch to remain compatible with later versions of NeMo."
        )

    if TORCH_VERSION < TORCH_VERSION_MIN:
        return stft(
            input=input,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            normalized=normalized,
            onesided=True,
        )
    else:
        return stft(
            input=input,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
            normalized=normalized,
            onesided=onesided,
            return_complex=return_complex,
        )


def istft_patch(
        input: torch.Tensor,
        n_fft: int,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: Optional[torch.Tensor] = None,
        center: bool = True,
        normalized: bool = False,
        onesided: Optional[bool] = None,
        length: int = None,
        return_complex: Optional[bool] = False,
):
    """
    Patch over torch.stft for PyTorch <= 1.6.
    Arguments are same as torch.stft().

    # TODO: Remove once PyTorch 1.7+ is a requirement.
    """
    global TORCH_VERSION
    if TORCH_VERSION is None:
        TORCH_VERSION = version.parse(torch.__version__)

        logging.warning(
            "torch.stft() signature has been updated for PyTorch 1.7+\n"
            "Please update PyTorch to remain compatible with later versions of NeMo."
        )

    if TORCH_VERSION < TORCH_VERSION_MIN:
        return istft(
            input=input,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            length=length,
            normalized=normalized,
            onesided=True,
        )
    else:
        return istft(
            input=input,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            length=length,
            normalized=normalized,
            onesided=onesided,
            return_complex=return_complex,
        )


def stft(input: Tensor, n_fft: int, hop_length: Optional[int] = None,
         win_length: Optional[int] = None, window: Optional[Tensor] = None,
         center: bool = True, pad_mode: str = 'reflect', normalized: bool = False,
         onesided: Optional[bool] = None,
         return_complex: Optional[bool] = None) -> Tensor:
    # -----
    # print('---------')
    # print('size:', input.size())
    # print('stride:', input.stride())
    # print('n_fft:', n_fft)
    # print('hop_length:', hop_length)
    # print('win_length:', win_length)
    # ---->
    # size: torch.Size([16, 206720])
    # n_fft=512
    # hop_length=160
    # win_length=400
    # n_fft: 512
    # hop_length: 160
    # win_length: 400
    if has_torch_function_unary(input):
        return handle_torch_function(
            stft, (input,), input, n_fft, hop_length=hop_length, win_length=win_length,
            window=window, center=center, pad_mode=pad_mode, normalized=normalized,
            onesided=onesided, return_complex=return_complex)
    # TODO: after having proper ways to map Python strings to ATen Enum, move
    #       this and F.pad to ATen.
    if center:
        signal_dim = input.dim()
        extended_shape = [1] * (3 - signal_dim) + list(input.size())
        pad = int(n_fft // 2)
        input = F.pad(input.view(extended_shape), [pad, pad], pad_mode)
        input = input.view(input.shape[-signal_dim:])
    input = input.unfold(-1, n_fft, hop_length).flatten(-2)
    hop_length = n_fft
    return _VF.stft(input, n_fft, hop_length, win_length, window,  # type: ignore[attr-defined]
                    normalized, onesided, return_complex)


def istft(input: Tensor, n_fft: int, hop_length: Optional[int] = None,
          win_length: Optional[int] = None, window: Optional[Tensor] = None,
          center: bool = True, normalized: bool = False,
          onesided: Optional[bool] = None, length: Optional[int] = None,
          return_complex: bool = False) -> Tensor:
    if has_torch_function_unary(input):
        return handle_torch_function(
            istft, (input,), input, n_fft, hop_length=hop_length, win_length=win_length,
            window=window, center=center, normalized=normalized, onesided=onesided,
            length=length, return_complex=return_complex)

    input = input.unfold(-1, n_fft, hop_length).flatten(-2)
    hop_length = n_fft
    return _VF.istft(input, n_fft, hop_length, win_length, window, center,  # type: ignore[attr-defined]
                     normalized, onesided, length, return_complex)
