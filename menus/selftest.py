"""自检 / 运行测试菜单"""
import subprocess
import sys

from InquirerPy import inquirer
from InquirerPy.base.control import Choice


def menu_selftest():
    """运行项目自检测试"""
    mode = inquirer.select(
        message="选择自检模式:",
        choices=[
            Choice("quick", "⚡ 快速自检 (仅核心函数)"),
            Choice("verbose", "📋 详细自检 (显示每条测试)"),
        ],
        default="quick",
    ).execute()

    args = [sys.executable, "-m", "pytest", "tests/", "--tb=short", "-q"]
    if mode == "verbose":
        args = [sys.executable, "-m", "pytest", "tests/", "-v"]

    print("\n🔍 正在运行自检...\n")
    result = subprocess.run(args)

    if result.returncode == 0:
        print("\n✅ 所有自检通过，环境正常！")
    else:
        print("\n⚠️  部分自检未通过，请检查上方输出。")
