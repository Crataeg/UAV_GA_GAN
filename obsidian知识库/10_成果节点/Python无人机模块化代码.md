---
title: "Python无人机模块化代码"
task_type: "成果"
source_path: "D:\UAV_Communication_GA\UAV_Communication_GA"
copied_path: "D:\论文无人机\成果本身\代码工程\UAV_Communication_GA"
---
# Python无人机模块化代码

## 节点总结

该目录保留无人机通信工程的模块化实现，可作为论文中的模型封装和子模块说明依据。

## 写作价值

适合支撑模块设计、实现结构和可复现实验描述。

## 原始位置

- `D:\UAV_Communication_GA\UAV_Communication_GA`

## 复制后位置

- `D:\论文无人机\成果本身\代码工程\UAV_Communication_GA`

## 文件规模

- 文件数：`37`

## 关键文件

- `成果本身\代码工程\UAV_Communication_GA\main.py`
- `成果本身\代码工程\UAV_Communication_GA\requirements.py`
- `成果本身\代码工程\UAV_Communication_GA\combined\_init_.py`
- `成果本身\代码工程\UAV_Communication_GA\distance\_init_.py`
- `成果本身\代码工程\UAV_Communication_GA\interference\_init_.py`
- `成果本身\代码工程\UAV_Communication_GA\obstruction\_init_.py`
- `成果本身\代码工程\UAV_Communication_GA\utils\_init_.py`
- `成果本身\代码工程\UAV_Communication_GA\velocity\_init_.py`
- `成果本身\代码工程\UAV_Communication_GA\combined\combined.py`
- `成果本身\代码工程\UAV_Communication_GA\utils\common_utils.py`
- `成果本身\代码工程\UAV_Communication_GA\distance\distance.py`
- `成果本身\代码工程\UAV_Communication_GA\interference\interference.py`

## 代表性内容预览

### `成果本身\代码工程\UAV_Communication_GA\main.py`

```text
#!/usr/bin/env python3
"""
无人机通信劣化程度优化系统 - 主程序入口（三维可视化版本）
总通信劣化程度 = W₁ × D₁(d) + W₂ × D₂(o) + W₃ × D₃(i) + W₄ × D₄(v)
"""

import numpy as np
import sys
import os
import argparse
from datetime import datetime
import traceback

# Keep the legacy console prints from crashing on GBK terminals.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

print("=== 系统启动调试信息 ===")
print(f"当前目录: {current_dir}")
print(f"Python版本: {sys.version}")
print(f"工作目录: {os.getcwd()}")

# 调试信息：检查模块是否存在
print("\n=== 检查模块是否存在 ===")
modules_to_check = [
    ('distance', 'DistanceOptimizer'),
    ('obstruction', 'ObstructionOptimizer3D'),
    ('interference', 'InterferenceOptimizer'),
    ('velocity', 'VelocityOptimizer'),
    ('combined', 'CombinedOptimizer')
]

for package_name, class_name in modules_to_check:
    package_path = os.path.join(current_dir, package_name)
    module_path = os.path.join(package_path, f"{package_name}.py")
    if os.path.exists(package_path) and os.path.exists(module_path):
        print(f"✓ 找到 {package_name} 包和模块")
    else:
        print(f"✗ 未找到 {package_name} 包或模块")

# 导入各个模块 - 使用包结构导入
print("\n=== 尝试导入模块 ===")
try:
    from distan
```

### `成果本身\代码工程\UAV_Communication_GA\requirements.py`

```text
"""
依赖检查和管理工具
自动检查并安装项目所需的依赖包
"""

import sys
import subprocess
import importlib
from typing import Dict, List, Tuple, Optional

# 必需依赖包及其最低版本（使用导入名）
REQUIRED_PACKAGES = {
    'numpy': '1.19.0',
    'scipy': '1.5.0',
    'geatpy': '2.7.0',
    'matplotlib': '3.3.0',
    'pandas': '1.1.0',
    'tqdm': '4.62.0',
    'yaml': '5.4.0',  # 导入名是 yaml，安装名是 pyyaml
    'colorlog': '6.0.0'
}

# 包名映射（导入名 -> 安装名）
PACKAGE_INSTALL_MAP = {
    'yaml': 'pyyaml'
}

# 特殊包版本处理
SPECIAL_PACKAGE_VERSIONS = {
    'colorlog': '6.10.1'  # 手动指定已知版本
}


def check_python_version() -> Tuple[bool, str]:
    """检查Python版本"""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major == 3 and version.minor >= 6:
        return True, version_str
    else:
        return False, version_str


def get_installed_version(package_name: str) -> Optional[str]:
    """获取已安装包的版本"""
    try:
        # 特殊处理 PyYAML
        if package_name == 'yaml':
            try:
                import yaml
                return getattr(yaml, '__version__', '6.0.1')  # PyYAML 6.0.1
            except ImportError:
                return None

        # 特殊处理 colorlog（没有标准版本属性）
        if package_name == 'colorlog':
            try:
                import colorlog
                # colorlog 没有 __version__ 属性，返回已知版本
                return SPECIAL_PACKAGE_VERSIONS.get('colorlog',
```

### `成果本身\代码工程\UAV_Communication_GA\combined\_init_.py`

```text
"""
综合联调系统
组合距离、遮挡、干扰、速度四个中间件进行全局优化（三维版本）
总通信劣化程度 = W₁ × D₁(d) + W₂ × D₂(o) + W₃ × D₃(i) + W₄ × D₄(v)
"""
import numpy as np

from .combined import (
    CombinedDegradationModel,
    CombinedOptimizer
)

__all__ = [
    'CombinedDegradationModel',
    'CombinedOptimizer'
]

__version__ = '2.0.0'
__author__ = '无人机通信优化系统'

# 包级别的默认配置
DEFAULT_PARAMETERS = {
    'default_weights': {
        'distance': 0.3,
        'obstruction': 0.2,
        'interference': 0.3,
        'velocity': 0.2
    },
    'variable_ranges': {
        'gs_position': [0, 1000],
        'obstruction_position': [0, 1000],
        'obstruction_height': [0, 100],
        'interference_position': [0, 1000],
        'interference_height': [0, 100],
        'interference_frequency': [100e6, 6000e6],
        'interference_bandwidth': [1e6, 100e6],
        'interference_power': [0.1, 10],
        'uav_velocities': [0, 50]
    }
}


def get_module_info():
    """返回模块信息"""
    return {
        'name': '三维综合联调系统',
        'version': __version__,
        'description': '组合四个因素进行无人机通信劣化程度全局优化（支持三维空间）',
        'author': __author__,
        'main_classes': __all__,
        'formula': '总通信劣化程度 = W₁ × D₁(d) + W₂ × D₂(o) + W₃ × D₃(i) + W₄ × D₄(v)',
        'features': ['三维距离计算', '三维遮挡模型', '三维干扰模型', '高度感知优化']
    }


def quick_test():
    """快速测试函数"""
    print("三维综合联调系统快速测试...")

    # 创建测试无人机位置（三维）
    uav_positions = np.ar
```
