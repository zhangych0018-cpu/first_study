"""本测试模块用轻量假对象模拟 CST 结果树，验证真实 CST 导出适配层能生成 BO 后处理器可直接消费的标准文件。"""

from __future__ import annotations

from pathlib import Path

from sws_bo.utils.cst_result_export import export_standard_dsg_results
from sws_bo.utils.postprocessing import parse_dsg_cst_results


class FakeResultItem:
    """模拟 CST 单个 1D 结果节点，只暴露导出层实际使用的 `get_data` 接口。"""

    def __init__(self, data):
        self._data = data

    def get_data(self):
        return self._data


class FakeResults3D:
    """模拟 CST 3D 结果访问对象，用于在无 CST 环境下验证节点映射与文件导出。"""

    def __init__(self, items: dict[str, list[tuple]]):
        self._items = items

    def get_result_item(self, tree_item: str) -> FakeResultItem:
        return FakeResultItem(self._items[tree_item])


def test_export_standard_dsg_results_feeds_postprocessing(tmp_path: Path):
    items = {
        r"1D Results\S-Parameters\S1,1": [(96.0, 0.08 + 0.01j), (100.0, 0.05 + 0.01j), (104.0, 0.07 + 0.01j)],
        r"1D Results\S-Parameters\S2,1": [(96.0, 0.75 + 0.02j), (100.0, 0.72 + 0.02j), (104.0, 0.70 + 0.02j)],
        r"1D Results\DSG\Dispersion TM21": [(96.0, 80.0, 0.149), (100.0, 90.0, 0.145), (104.0, 100.0, 0.143)],
        r"1D Results\DSG\Dispersion Fundamental": [(96.0, 40.0, 0.260), (100.0, 45.0, 0.255), (104.0, 50.0, 0.250)],
        r"1D Results\DSG\Kc TM21": [(96.0, 7.6), (100.0, 8.3), (104.0, 7.8)],
        r"1D Results\DSG\Kc Fundamental": [(96.0, 4.0), (100.0, 5.0), (104.0, 4.8)],
        r"1D Results\Mode Frequencies\Mode Summary": [(1.0, 95.5), (2.0, 100.0)],
    }
    tree_items = sorted(items)

    summary = export_standard_dsg_results(
        results_3d=FakeResults3D(items),
        tree_items=tree_items,
        run_dir=tmp_path,
    )

    assert summary["required_complete"] is True
    assert (tmp_path / "mode_frequencies.csv").exists()

    metrics = parse_dsg_cst_results(
        dispersion_tm21_path=tmp_path / "dispersion_tm21.txt",
        dispersion_fundamental_path=tmp_path / "dispersion_fundamental.txt",
        kc_tm21_path=tmp_path / "kc_tm21.txt",
        kc_fundamental_path=tmp_path / "kc_fundamental.txt",
        sparameters_path=tmp_path / "sparameters.txt",
        mode_frequencies_path=tmp_path / "mode_frequencies.csv",
    )
    assert metrics["Kc_TM21_mean"] > 7.0
    assert metrics["mode_ratio"] > 1.2
    assert metrics["mode_frequency_count"] == 2.0
