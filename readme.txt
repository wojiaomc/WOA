“基于鲸鱼优化算法的城市旅游路线规划系统”软件使用说明书
一、运行环境
1．硬件环境：
		为了正常运行“基于鲸鱼优化算法的城市旅游路线规划系统”软件，用户需要确保计算机系统符合以下最低要求：
	处理器（CPU）：
	最低要求：Intel Core i5 或同等性能的处理器
	推荐配置：Intel Core i7 或更高性能的处理器
	内存（RAM）：
	最低要求：8 GB
	推荐配置：16 GB 或更高
	存储（硬盘）：
	最低要求：256 GB SSD 或 500 GB HDD
	推荐配置：512 GB SSD 或更高
	显卡（GPU）：
	最低要求：集成显卡（如 Intel UHD Graphics 620）
	推荐配置：独立显卡（如 NVIDIA GeForce GTX 1050 或更高）
	显示器：
	分辨率：最低 1920x1080（全高清）
	推荐配置：2560x1440 或更高
	操作系统：
	支持的操作系统：Windows 10 或更高版本，macOS 10.15 或更高版本
	网络：
	互联网连接：建议使用稳定的宽带连接，以确保软件更新和在线功能的正常使用
	其他外设：
	鼠标和键盘：标准 USB 或无线鼠标和键盘
	打印机：如需打印功能，建议使用兼容的 USB 或网络打印机
	通过明确的硬件环境要求，用户可以更好地了解他们的设备是否能够支持运行该软件，并有助于避免因硬件不兼容而导致的问题。

2．软件环境
软件环境要求部分应该涵盖软件工具的版本和配置，以确保用户的系统可以正常运行和开发软件。
（1） 数据工程部分（Jupyter Notebook）：
   - Anaconda：推荐安装Anaconda最新版本，以确保Jupyter Notebook及其依赖库的稳定性。
   - Jupyter Notebook：确保Jupyter Notebook已在Anaconda环境中正确安装，并可以通过浏览	器访问。
   - Python版本：建议使用Python 3.x版本，支持Jupyter Notebook中常用的数据处理和可视化	库。
   - 数据处理库：安装常用的数据处理库，如NumPy、Pandas、Matplotlib等，以满足数据工程需	求。
（2）系统开发部分（PyCharm）：
   - PyCharm：安装PyCharm最新专业版或社区版，以支持系统开发和调试功能。
   - Python解释器：PyCharm需要配置正确的Python解释器，建议使用与Jupyter Notebook相同的	Python环境。
   - 第三方库：安装项目所需的第三方库和依赖，在PyCharm中设置Python解释器路径以确保正确	导入。
通过明确软件环境要求，用户可以更轻松地配置其系统和软件环境，以确保能够顺利进行数据工程和系统开发工作。同时，建议根据实际项目需要安装和配置其他必要的软件工具和库，以满足具体的开发需求。

二、软件安装方法
以下是安装和配置所需软件环境的详细步骤：
安装步骤：

1.安装 Anaconda：
访问Anaconda官方网站（https://www.anaconda.com/products/distribution）下载适合您
操作系统的Anaconda安装程序。选择最新版本，并双击安装程序。
在安装过程中，按照向导的提示一步步操作，可以选择是否将Anaconda添加至系统环境变量。
创建虚拟环境：
打开 Anaconda Prompt 或终端，运行以下命令创建并激活虚拟环境：
conda create -n xxxxx_env python=3.8
conda activate xxxxx_env

2.安装必要的 Python 库：
在Anaconda环境中安装所需的Python版本和第三方库，可以通过Anaconda Navigator的环境管理器进行操作。
在PyCharm中设置Python解释器路径，可在File -> Settings -> Project Interpreter中选	择Anaconda环境中的Python解释器，确保项目与Jupyter Notebook使用相同的Python环境。
在激活的虚拟环境中，运行以下命令安装所需的库：
conda install numpy pandas matplotlib seaborn scikit-learn
pip install tensorflow  # 或者 pip install torch
pip install flask  # 或者 pip install django

安装 Jupyter Notebook：
在激活的虚拟环境中，运行以下命令安装 Jupyter Notebook：
conda install jupyter
打开Anaconda Navigator后，在导航器界面找到Jupyter Notebook，点击启动，并创建一个新的Notebook来验证安装是否成功。

3.安装 PyCharm：
访问PyCharm官方网站（https://www.jetbrains.com/pycharm/download）下载适合您操作系
统的PyCharm安装程序。选择专业版或社区版，根据个人需求。
运行安装程序，按照安装向导的步骤安装PyCharm。可以选择安装默认设置，也可以根据个人偏好进行自定义设置。
安装完成后，启动PyCharm，创建一个新的项目，并根据需要配置Python解释器路径，以确保项目能够正确导入所需的库和依赖。
配置 PyCharm：
打开 PyCharm，设置项目解释器为您创建的 Anaconda 虚拟环境（xxxxx_env）。

4.在PyCharm中使用Flask：
安装Flask：通过PyCharm的项目解释器安装Flask库，或使用Anaconda环境进行安装。
创建Flask应用：在PyCharm中创建一个Flask应用程序，设置应用结构、路由和视图函数。
运行Flask应用：通过PyCharm直接运行Flask应用，调试和查看应用程序的输出。
调试Flask应用：利用PyCharm的调试功能，可以方便地调试Flask应用程序，查看变量状态和代码执行过程。

5.读取数据集
使用 Pandas 库读取 CSV 文件：
import pandas as pd
# 读取 CSV 文件
data = pd.read_csv('path/to/your/dataset.csv')
# 显示前五行数据
print(data.head())

请确保软件环境符合上述要求，以获得最佳的用户体验和软件性能。

三、软件功能说明
首先，系统的初始界面需要输入旅游的地区，用户可以根据自身的意向随意选择旅游地区进行旅游规划。

系统会读取用户所输入的字符并在后端中进行搜索操作，对数据集的“城市”字段进行查找，如果查找失败，则会系统会转跳到失败界面，此时用户点击返回，则会转跳到系统的初始界面，可以重新进行旅游地区的输入。确保系统稳定运行。

当用户输入了正确的旅游地区，即后端可以在数据集中查询到对应“城市”特征的数据后。系统会重新创建一个DataFrame对象，仅将符合条件的数据保存在这个新建的数据集对象中，然后通过可视化的方式，将选定区域的数据分布展现出来。给用户提供旅游规划的详细参考，用户根据给出的数据信息来进行旅游的进一步规划，输入旅游所需要花费的时间与预算。然后提交给后端进行算法的搜索。

当系统读取到了所有的数据条件后，便会调用鲸鱼优化算法，进行最佳旅游路线的搜索。并最终将最佳路线的所有景点名称，以及整条路线的评分和花费都显示出来。同时，用户还可以点击再来一次，回到初始界面重新进行整个旅游规划。