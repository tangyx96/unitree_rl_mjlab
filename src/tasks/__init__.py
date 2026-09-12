"""任务自动发现与导入。

``import src.tasks`` 会执行本文件，从而调用 import_packages。
后者基于 pkgutil.iter_modules 递归遍历本包下所有带子包的目录
（仅识别含 __init__.py 的目录），并对每个子包执行 __import__。
Python 导入包时会运行其 __init__.py；各机器人配置包在顶层调用
register_mjlab_task()，以副作用写入 mjlab 全局任务注册表。

黑名单按子串匹配包路径：".mdp"、"utils" 为工具模块，不含注册代码。
中间任一层缺少 __init__.py，walker 无法下行，其下任务不会被注册。
"""
from mjlab.utils.lab_api.tasks.importer import import_packages

_BLACKLIST_PKGS = ["utils", ".mdp"]

import_packages(__name__, _BLACKLIST_PKGS)