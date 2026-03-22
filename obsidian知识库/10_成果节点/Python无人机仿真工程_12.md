---
title: "Python无人机仿真工程_12"
task_type: "成果"
source_path: "D:\UAV_Communication_GA\12"
copied_path: "D:\论文无人机\成果本身\代码工程\12"
---
# Python无人机仿真工程_12

## 节点总结

该目录是无人机论文线的主工程，包含 GA、GAN、KPI/BLER 评估和实测对照输出。

## 写作价值

是论文方法与结果章节的直接来源。

## 原始位置

- `D:\UAV_Communication_GA\12`

## 复制后位置

- `D:\论文无人机\成果本身\代码工程\12`

## 文件规模

- 文件数：`206`

## 关键文件

- `成果本身\代码工程\12\bler_mc.py`
- `成果本身\代码工程\12\bler_sionna.py`
- `成果本身\代码工程\12\compare_measured_cornercases.py`
- `成果本身\代码工程\12\compare_random_ga_gan.py`
- `成果本身\代码工程\12\evaluate.py`
- `成果本身\代码工程\12\gan_uav_pipeline.py`
- `成果本身\代码工程\12\kpi.py`
- `成果本身\代码工程\12\plot_results.py`
- `成果本身\代码工程\12\quick_demo_pipeline.py`
- `成果本身\代码工程\12\UAV_GA.py`
- `成果本身\代码工程\12\uav_gan_cli.py`
- `成果本身\代码工程\12\__backup_before_upstream_20260302_143119\compare_random_ga_gan.py`

## 代表性内容预览

### `成果本身\代码工程\12\bler_mc.py`

```text
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np


def _qpsk_map(bits: np.ndarray) -> np.ndarray:
    # bits shape: (N, 2) -> symbols shape: (N,)
    b0 = bits[:, 0].astype(np.float32)
    b1 = bits[:, 1].astype(np.float32)
    # Gray mapping: 00->(1+1j), 01->(1-1j), 11->(-1-1j), 10->(-1+1j)
    i = 1.0 - 2.0 * b0
    q = 1.0 - 2.0 * b1
    s = (i + 1j * q) / np.sqrt(2.0)
    return s.astype(np.complex64)


def _qpsk_demod_hard(symbols: np.ndarray) -> np.ndarray:
    i = (np.real(symbols) < 0.0).astype(np.int8)
    q = (np.imag(symbols) < 0.0).astype(np.int8)
    return np.stack([i, q], axis=1)


def _qam16_map(bits: np.ndarray) -> np.ndarray:
    # bits shape: (N, 4) Gray mapping with levels {-3,-1,1,3}, normalized
    b0, b1, b2, b3 = [bits[:, k].astype(np.int8) for k in range(4)]

    def gray_2bit_to_level(msb: np.ndarray, lsb: np.ndarray) -> np.ndarray:
        # 00->-3, 01->-1, 11->+1, 10->+3
        lvl = np.empty_like(msb, dtype=np.float32)
        m0 = (msb == 0) & (lsb == 0)
        m1 = (msb == 0) & (lsb == 1)
        m2 = (msb == 1) & (lsb == 1)
        m3 = (msb == 1) & (lsb == 0)
        lvl[m0] = -3.0
        lvl[m1] = -1.0
        lvl[m2] = 1.0
        lvl[m3] = 3.0
        return lvl

    i = gray_2bit_to_level(b0, b1)
    q = gray_2bit_to_level(b2, b3)

```

### `成果本身\代码工程\12\bler_sionna.py`

```text
from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class SionnaConfig:
    modulation: str = "qpsk"  # qpsk|16qam
    n_bits_per_block: int = 1024
    n_blocks: int = 400
    seed: int = 0
    channel: str = "awgn"  # awgn|rician
    rician_k: float = 5.0


def _require_sionna():
    try:
        import tensorflow as tf  # noqa: F401
        import sionna  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Sionna/TensorFlow not available. Install tensorflow + sionna first, "
            "or provide an existing BLER_B.csv LUT."
        ) from e


def simulate_bler_curve_sionna(
    sinr_db_grid: Iterable[float],
    cfg: SionnaConfig = SionnaConfig(),
) -> Tuple[np.ndarray, np.ndarray]:
    _require_sionna()

    import tensorflow as tf
    # Sionna v1.x namespace
    from sionna.phy.mapping import BinarySource, Constellation, Mapper, Demapper
    from sionna.phy.channel import AWGN

    mod = str(cfg.modulation).strip().lower()
    if mod not in ("qpsk", "16qam"):
        raise ValueError("modulation must be 'qpsk' or '16qam'")

    bits_per_sym = 2 if mod == "qpsk" else 4
    n_bits = int(max(cfg.n_bits_per_block, bits_per_sym))
    n_bits = int((n_bits // bits_per_sym) * bits_per_sym)
    n_sy
```

### `成果本身\代码工程\12\compare_measured_cornercases.py`

```text
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import zipfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MethodType
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from UAV_GA import DroneCommProblem, InterferenceSource
import gan_uav_pipeline as gan_pipeline
from kpi import outage_probability, summarize_distribution, tail_mean


DEFAULT_AERPAW_ZIP = r"d:\Downloads\aerpaw-dataset-24.zip"
DEFAULT_DRYAD_ZIP = r"d:\Downloads\doi_10_5061_dryad_wh70rxx06__v20250521.zip"
DEFAULT_REF_PDF = r"d:\UAV_Communication_GA\key reference\Aerial_RF_and_Throughput_Measurements_on_a_Non-Standalone_5G_Wireless_Network.pdf"

AERPAW_NR_FILES = (
    "aerpaw-dataset-24/Logs/pawprints_5G_NR_altitude30m_flight1.csv",
    "aerpaw-dataset-24/Logs/pawprints_5G_NR_altitude30m_flight2.csv",
)
AERPAW_THR_FILES = ("aerpaw-dataset-24/Logs/pawprints_iperf_throughput_altitude30m_flight1.csv",)
DRYAD_SINR_FILES = (
    "Dryad/yaw45/inputf1_sinr_with_header.csv",
    "Dryad/yaw315/inputf1_sinr_with_header.csv",
)
DRYAD_THR_FILES = (
    "Dryad/yaw45/input_throughput_with_header.csv",
    "Dryad/yaw315/input_throughput_with_header.csv",
)


def _base_user_params() -> Dict[
```
