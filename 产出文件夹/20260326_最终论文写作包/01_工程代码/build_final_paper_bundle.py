import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(str(dst))
    shutil.copytree(str(src), str(dst))


def _optional_copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    if src.is_dir():
        _copy_tree(src, dst)
    else:
        _copy_file(src, dst)
    return True


def build_bundle(run_id: str, out_dir: Path) -> Dict:
    compare_dir = ROOT / "output" / run_id / "compare"
    gan_dir = ROOT / "output" / run_id / "gan"

    code_dir = out_dir / "01_工程代码"
    analysis_dir = out_dir / "02_行级技术解析"
    source_dir = out_dir / "03_来源与开源数据"
    output_dir = out_dir / "04_最终运行输出"
    writing_dir = out_dir / "05_论文写作入口"
    vscode_dir = out_dir / ".vscode"

    out_dir.mkdir(parents=True, exist_ok=True)
    code_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    writing_dir.mkdir(parents=True, exist_ok=True)
    vscode_dir.mkdir(parents=True, exist_ok=True)

    code_files = [
        "UAV_GA.py",
        "compare_random_ga_gan.py",
        "compare_measured_map_search.py",
        "gan_uav_pipeline.py",
        "evaluate.py",
        "kpi.py",
        "plot_results.py",
        "project_defaults.py",
        "experiment_profiles.py",
        "current_compare_rules.py",
        "iterative_algorithm_benchmark.py",
        "measured_dataset_loaders.py",
        "publication_figures.py",
        "sinr_focus_report.py",
        "build_final_paper_bundle.py",
        "quick_demo_pipeline.py",
        "bler_mc.py",
        "bler_sionna.py",
        "model_source_registry.json",
        "COMM_DEGRADATION_REPORT.md",
        "INTERFERENCE_SOURCE_TABLE.md",
    ]
    copied_code: List[str] = []
    for name in code_files:
        src = ROOT / name
        if src.exists():
            _copy_file(src, code_dir / name)
            copied_code.append(name)

    for name in [
        "SINR_FIRST_MODEL_AND_LINE_MAP.md",
        "RESEARCH_THOUGHT_TO_CODE_MAP.md",
        "FIVE_GOAL_CLOSURE_CHECK.md",
        "CURRENT_COMPARE_RULES.md",
    ]:
        _copy_file(ROOT / "final_bundle_assets" / name, analysis_dir / name)

    for name in [
        "PAPER_WRITING_GUIDE.md",
        "01_基于目前工程的行级写作大纲.md",
        "02_最终文件夹指南.md",
        "03_工程技术细节通俗解释_含公式推导与研究依据.md",
    ]:
        _copy_file(ROOT / "final_bundle_assets" / name, writing_dir / name)

    _optional_copy(ROOT / ".vscode", vscode_dir)
    _optional_copy(ROOT / "参考文献" / "论文文献" / "key reference" / "20260325_全网补证_方法来源", source_dir / "20260325_全网补证_方法来源")

    dataset_dir = source_dir / "open_data_zips"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_candidates = [
        Path(r"C:\Users\52834\Downloads\aerpaw-dataset-24.zip"),
        Path(r"C:\Users\52834\Downloads\doi_10_5061_dryad_wh70rxx06__v20250521.zip"),
    ]
    copied_datasets = []
    for path in dataset_candidates:
        if _optional_copy(path, dataset_dir / path.name):
            copied_datasets.append(str(path))

    _optional_copy(compare_dir, output_dir / "compare")
    _optional_copy(gan_dir, output_dir / "gan")
    _optional_copy(ROOT / "产出文件夹" / "20260325_可信来源与非臆造主链", output_dir / "20260325_可信来源与非臆造主链")
    _optional_copy(ROOT / "产出文件夹" / "20260326_算法优越性迭代基准", output_dir / "20260326_算法优越性迭代基准")

    readme = writing_dir / "00_README.md"
    readme.write_text(
        "# Final Paper Bundle\n\n"
        f"Run ID: `{run_id}`\n\n"
        "This folder is the direct writing bundle for the UAV paper project.\n\n"
        "## Read Order\n\n"
        "1. `05_论文写作入口/00_README.md`\n"
        "2. `05_论文写作入口/PAPER_WRITING_GUIDE.md`\n"
        "3. `02_行级技术解析/SINR_FIRST_MODEL_AND_LINE_MAP.md`\n"
        "4. `02_行级技术解析/RESEARCH_THOUGHT_TO_CODE_MAP.md`\n"
        "5. `04_最终运行输出/compare/final_integrated_report.json`\n"
        "6. `04_最终运行输出/compare/sinr_focus_summary.json`\n",
        encoding="utf-8",
    )
    readme.write_text(
        "# Final Paper Bundle\n\n"
        f"Run ID: `{run_id}`\n\n"
        "This folder is the direct writing bundle for the UAV paper project.\n\n"
        "## Read Order\n\n"
        "1. `05_论文写作入口/00_README.md`\n"
        "2. `05_论文写作入口/02_最终文件夹指南.md`\n"
        "3. `05_论文写作入口/01_基于目前工程的行级写作大纲.md`\n"
        "4. `05_论文写作入口/03_工程技术细节通俗解释_含公式推导与研究依据.md`\n"
        "5. `05_论文写作入口/PAPER_WRITING_GUIDE.md`\n"
        "6. `02_行级技术解析/SINR_FIRST_MODEL_AND_LINE_MAP.md`\n"
        "7. `02_行级技术解析/RESEARCH_THOUGHT_TO_CODE_MAP.md`\n"
        "8. `04_最终运行输出/compare/final_integrated_report.json`\n"
        "9. `04_最终运行输出/compare/sinr_focus_summary.json`\n",
        encoding="utf-8",
    )

    manifest = {
        "run_id": run_id,
        "copied_code_files": copied_code,
        "copied_dataset_sources": copied_datasets,
        "compare_dir": str(compare_dir),
        "gan_dir": str(gan_dir),
        "bundle_root": str(out_dir),
    }
    with open(out_dir / "FINAL_BUNDLE_MANIFEST.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final paper-writing bundle.")
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--out_dir", default=str(ROOT / "产出文件夹" / "20260326_最终论文写作包"))
    args = parser.parse_args()

    manifest = build_bundle(args.run_id, Path(args.out_dir))
    print("Final bundle:", manifest["bundle_root"])


if __name__ == "__main__":
    main()
