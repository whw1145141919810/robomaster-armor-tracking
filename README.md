# RoboMaster 装甲板定位与运动预测

用同一台固定相机完成棋盘格标定、课堂 ONNX 模型四角点检测、OpenCV PnP 解算和三维卡尔曼滤波。支持海康 MVS 工业相机、普通 USB 相机和同一台相机拍摄的原始视频。

## 安装

在项目目录运行下列命令。海康相机需要已安装 MVS SDK，本机默认从 `/opt/MVS` 加载。如果 SDK 安装路径不同，可设置 `MVS_PYTHON_PATH` 为 SDK 的 `MvImport` 目录，`MVCAM_COMMON_RUNENV` 为 SDK 的库目录。

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

请将课程提供的四关键点 ONNX 模型放到 `models/armor.onnx`，也可通过 `--model` 指定其实际位置。实验使用课堂的 `armor.onnx`，其元数据未记录原始权重文件名。运行前保持工业相机通电，在 MVS 客户端断开设备连接，释放相机给程序。

## 1. 标定

照片所示标定板有 6 列 × 9 行内角点，共 54 个。单格实测边长为 27 mm。

```bash
python calibrate.py --mvs 0 --cols 6 --rows 9 --square-mm 27 \
  --save-images runs/calibration_views --output calibration.json
```

将完整棋盘格置于画面中，每次改变位置、倾角或距离，按空格保存一组。建议至少拍 10 组，覆盖画面中部、边缘与四角。按 Enter 计算并保存相机内参、畸变系数、分辨率、整体及逐图重投影误差。若某一张误差明显偏大，检查模糊、反光和角点后重新采集。Esc 或 Q 退出。每次重新采集请为 `--save-images` 指定一个新目录。

也可以先通过 MVS 保存多角度原始图像，然后执行离线标定。

```bash
python calibrate.py --images runs/calibration_views --cols 6 --rows 9 \
  --square-mm 27 --output calibration.json
```

普通 USB 相机可将 `--mvs 0` 替换为 `--camera 0`，并在标定时用 `--width`、`--height` 请求相机支持的分辨率。标定与跟踪使用同一台相机、镜头、焦距和 ROI 设置。程序会核对输入帧分辨率。

## 2. 实时跟踪与演示素材

```bash
python run_tracking.py --mvs 0 --model models/armor.onnx \
  --calibration calibration.json --armor-width-mm 225 --armor-height-mm 50 \
  --csv runs/tracking.csv --record runs/demo.mp4 --snapshot runs/demo.png
```

窗口实时显示编号角点、位姿坐标轴、原始位置、滤波位置、距离、速度与 0.15 秒后的预测位置。按 Q 或 Esc 结束，并保存最后一张成功解算的截图。CSV 记录实际采集时间。MP4 按固定帧率输出，可用 `--record-fps` 调整播放速度。再次录制请换输出文件名，避免覆盖已有实验数据。

如果已有同一台相机录下的原始视频，可把 `--mvs 0` 替换为 `--video 原始视频.mp4`。无图形界面时加 `--headless`，仍可保存标注视频、截图和 CSV。需要定时退出时加 `--duration-s 30`。

- 按左上、右上、右下、左下将二维关键点和三维矩形角点一一对应。课堂模型的关键点在灯条中心线上，宽度应量两根灯条中心线的间距，高度应量灯条上下端点的间距。示例的 225 mm × 50 mm 是近似值，正式测试请用实测关键点间距。若画面中有不同规格的装甲板，每种尺寸单独测试。
- 装甲板中心为物体坐标原点。PnP 输出该中心在相机坐标系中的位置，X 向右、Y 向下、Z 向前，距离为三轴位置的欧氏范数。姿态以 Rodrigues 旋转向量记录。程序拒绝负深度或重投影误差超阈值的解。平面目标在接近正视时的姿态会更容易受到角点误差影响。
- 卡尔曼状态是三维位置和三维速度。每帧用实际时间间隔更新运动模型，有有效 PnP 结果时修正位置，短暂漏检时只预测。画面和 CSV 用 `measured`、`coasting`、`lost` 区分三种状态。丢失超过 0.35 秒后重新建立轨迹。跟踪一次处理一块装甲板，初始选有效解中模型置信度最高者，再按类别和预测像素位置续接。

## 3. 对比实验

固定相机，先让装甲板静止数秒，再左右移动并改变距离，最后在另一个距离停稳。记录各阶段在 CSV 中的时间范围。下面的起止时间是用法示例。

```bash
python analyze_log.py runs/tracking.csv --mode static --start-s 2 --end-s 6
python analyze_log.py runs/tracking.csv --mode moving --start-s 7 --end-s 14
```

静止段用 X、Y、Z 和距离的样本标准差衡量抖动，避开刚开始或重捕获的过渡帧。移动段比较坐标范围、位移、轨迹长度和同帧滤波差异，结合录像查看急停和变向时的跟随延迟。有实测距离时，可另外把测距误差与抖动一起记录。调参时可用 `--measurement-std-m` 分别设置三轴测量噪声，或用 `--acceleration-std-mps2` 调整对速度变化的响应。

## 实拍结果

- [亮灯装甲板实拍视频](demo-lit.mp4)与[该段逐帧数据](tracking-lit.csv)。新录制共处理 554 帧，其中 288 帧有有效 PnP 解。
- [亮灯装甲板的角点与 PnP 标注](armor-lit-pnp.png)
- [本次相机标定参数](mvs-calibration-1440x1080.json)、[逐帧跟踪数据](tracking.csv)与[实验分析](experiment.md)

仓库中的标定参数对应实拍时的 MVS 相机与 1440 × 1080 分辨率。按上述流程重新标定会在项目根目录生成 `calibration.json`，之后可直接使用跟踪示例命令。

## 参考

- [OpenCV 相机标定](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [OpenCV PnP 位姿解算](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)
- [OpenCV KalmanFilter](https://docs.opencv.org/4.x/dd/d6a/classcv_1_1KalmanFilter.html)
