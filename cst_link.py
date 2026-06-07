"""本模块提供一个最小可运行的 CST Python 示例，用于演示如何新建微波工程、配置本征模求解、运行求解器并从结果树中读取本征频率。它主要作为真实 DSG 自动化开发前的参考样例，帮助快速验证本机的 CST-Python 连接链路。"""

from __future__ import annotations

from pathlib import Path

from cst.interface import DesignEnvironment


# CST工程文件保存路径。
PROJECT_PATH = Path(r"C:\Users\87007\Desktop\SWS\DSG_SWS_1.cst")

# txt结果文件保存路径，和CST工程文件在同一个目录下。
RESULT_TXT_PATH = PROJECT_PATH.with_name(f"{PROJECT_PATH.stem}_eigenmode_results.txt")


def run_pillbox_eigenmode_simulation() -> None:
    """创建并运行一个简单的pillbox本征模仿真。"""

    # 如果目标文件夹不存在，先自动创建文件夹。
    PROJECT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 启动CST Design Environment，并创建一个新的Microwave Studio工程。
    design_environment = DesignEnvironment()
    project = None
    try:
        project = design_environment.new_mws()

        # 先保存一次工程，确保CST工程有明确的文件路径。
        project.save(PROJECT_PATH, allow_overwrite=True)

        # 依次创建几何、设置边界条件、设置本征模求解器。
        create_geometry(project)
        set_boundary_conditions(project)
        setup_solver(project)

        # 保存工程，运行CST求解器，然后再次保存求解后的结果。
        project.save(PROJECT_PATH, allow_overwrite=True)
        project.model3d.run_solver()
        project.save(PROJECT_PATH, allow_overwrite=True)

        # 读取CST结果树中的本征频率，并导出到txt文件。
        extract_results(PROJECT_PATH)
    finally:
        # 无论仿真是否成功，都尽量关闭CST工程和Design Environment。
        if project is not None:
            try:
                project.close()
            except RuntimeError:
                pass
        design_environment.close()


def add_history(project, name: str, vba_code: str) -> None:
    """向CST History列表添加一段VBA命令。"""

    # CST Python接口主要通过add_to_history执行建模和求解器设置命令。
    project.model3d.add_to_history(name, vba_code)


def create_geometry(project) -> None:
    """创建半径50 mm、高度100 mm的PEC pillbox圆柱。"""

    # 设置工程单位：几何单位为mm，频率单位为GHz，时间单位为ns。
    add_history(
        project,
        "set units",
        """
With Units
    .Geometry "mm"
    .Frequency "GHz"
    .Time "ns"
End With
""",
    )

    # 创建PEC实心圆柱，圆柱轴向为z方向，z范围为-50 mm到50 mm。
    add_history(
        project,
        "create pillbox",
        """
With Cylinder
    .Reset
    .Name "Pillbox"
    .Component "component1"
    .Material "PEC"
    .OuterRadius "50"
    .InnerRadius "0"
    .Axis "z"
    .Zrange "-50", "50"
    .Xcenter "0"
    .Ycenter "0"
    .Segments "0"
    .Create
End With
""",
    )


def set_boundary_conditions(project) -> None:
    """设置仿真区域六个方向的边界条件。"""

    # 将六个边界全部设置为electric，相当于理想电壁边界。
    add_history(
        project,
        "electric boundaries",
        """
With Boundary
    .Xmin "electric"
    .Xmax "electric"
    .Ymin "electric"
    .Ymax "electric"
    .Zmin "electric"
    .Zmax "electric"
    .Xsymmetry "none"
    .Ysymmetry "none"
    .Zsymmetry "none"
    .ApplyInAllDirections "False"
End With
""",
    )

    # 背景空间全部设为0，使仿真区域紧贴当前模型范围。
    add_history(
        project,
        "zero background spacing",
        """
With Background
    .ResetBackground
    .XminSpace "0"
    .XmaxSpace "0"
    .YminSpace "0"
    .YmaxSpace "0"
    .ZminSpace "0"
    .ZmaxSpace "0"
    .ApplyInAllDirections "False"
End With
""",
    )


def setup_solver(project) -> None:
    """设置CST本征模求解器，搜索1到5 GHz范围内的第一个模。"""

    # 这个CST版本识别的本征模求解器名称是"HF Eigenmode"。
    add_history(project, "solver type", 'ChangeSolverType "HF Eigenmode"')

    # 设置本征模搜索频率范围，单位已经在前面设置为GHz。
    add_history(project, "freq range", 'Solver.FrequencyRange "1", "5"')

    # 设置只求解1个本征模。
    add_history(
        project,
        "eigenmode solver settings",
        """
With EigenmodeSolver
    .Reset
    .SetNumberOfModes "1"
End With
""",
    )


def extract_results(project_path: Path) -> None:
    """读取CST本征频率结果，并导出到同目录txt文件。"""

    from cst.results import ProjectFile

    # 打开已经保存的CST工程结果文件。
    result_project = ProjectFile(str(project_path), allow_interactive=True)
    results_3d = result_project.get_3d()
    tree_items = results_3d.get_tree_items()

    # CST本征模求解后，最终频率通常保存在这个结果树节点中。
    mode_frequency_item = r"1D Results\Mode Frequencies\Mode 1"

    print(f"CST project saved to: {project_path}")
    if mode_frequency_item in tree_items:
        # 读取Mode 1本征频率，单位为GHz。
        data = results_3d.get_result_item(mode_frequency_item).get_data()
        print(f"Mode 1 eigenfrequency (GHz): {data}")

        # 将结果写入txt文件，文件与.cst工程位于同一目录。
        RESULT_TXT_PATH.write_text(
            "\n".join(
                [
                    "CST pillbox eigenmode simulation result",
                    f"Project file: {project_path}",
                    f"Result item: {mode_frequency_item}",
                    f"Mode 1 eigenfrequency (GHz): {data}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print(f"Result txt saved to: {RESULT_TXT_PATH}")
    else:
        # 如果固定节点没找到，就打印所有疑似本征模相关的结果节点，方便排查。
        likely_items = [
            item
            for item in tree_items
            if "mode" in item.lower() or "eigen" in item.lower() or "frequency" in item.lower()
        ]
        print("Solver finished, but the final Mode 1 frequency item was not found.")
        if likely_items:
            print("Available related result items:")
            for item in likely_items:
                print(f"  {item}")


if __name__ == "__main__":
    # 直接运行本文件时，执行完整CST建模、求解和txt导出流程。
    run_pillbox_eigenmode_simulation()
