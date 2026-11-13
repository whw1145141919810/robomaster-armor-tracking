"""BGR frame capture for OpenCV cameras and the optional Hikrobot MVS SDK."""

import ctypes
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np


def open_camera(index, *, width=None, height=None, mvs=False):
    if index < 0:
        raise ValueError("相机编号必须大于等于 0。")
    if mvs:
        if width is not None or height is not None:
            raise ValueError("请通过 MVS 客户端设置工业相机分辨率和 ROI。")
        return MvsCapture(index)

    camera = cv2.VideoCapture(index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"无法打开 OpenCV 相机 {index}。海康工业相机请使用 --mvs。")
    if width is not None:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return camera


def check_mvs(result, action):
    if result:
        raise RuntimeError(f"{action}失败，MVS 错误码 0x{result & 0xffffffff:08X}。")


class MvsCapture:
    def __init__(self, index):
        sdk_dir = Path(os.getenv("MVS_PYTHON_PATH", "/opt/MVS/Samples/64/Python/MvImport"))
        os.environ.setdefault("MVCAM_COMMON_RUNENV", "/opt/MVS/lib")
        sys.path.insert(0, str(sdk_dir))
        try:
            import MvCameraControl_class as sdk
        except (ImportError, OSError) as exc:
            raise RuntimeError(f"无法加载 MVS SDK，请检查 MVS_PYTHON_PATH 和 SDK 库路径。{exc}") from exc
        finally:
            sys.path.pop(0)

        self.sdk = sdk
        self.camera = sdk.MvCamera()
        self.initialized = self.created = self.opened = self.streaming = False
        self.previous_trigger = None
        try:
            check_mvs(self.camera.MV_CC_Initialize(), "初始化 SDK")
            self.initialized = True
            devices = sdk.MV_CC_DEVICE_INFO_LIST()
            check_mvs(self.camera.MV_CC_EnumDevices(sdk.MV_GIGE_DEVICE | sdk.MV_USB_DEVICE, devices), "枚举相机")
            if index >= devices.nDeviceNum:
                raise RuntimeError(f"MVS 找到 {devices.nDeviceNum} 台相机，无法打开编号 {index}。")
            check_mvs(self.camera.MV_CC_CreateHandle(devices.pDeviceInfo[index].contents), "创建相机句柄")
            self.created = True
            check_mvs(self.camera.MV_CC_OpenDevice(), "打开相机（请先断开 MVS 客户端）")
            self.opened = True
            trigger = sdk.MVCC_ENUMVALUE()
            check_mvs(self.camera.MV_CC_GetEnumValue("TriggerMode", trigger), "读取触发模式")
            if trigger.nCurValue != sdk.MV_TRIGGER_MODE_OFF:
                check_mvs(self.camera.MV_CC_SetEnumValue("TriggerMode", sdk.MV_TRIGGER_MODE_OFF), "关闭触发模式")
                self.previous_trigger = trigger.nCurValue
            check_mvs(self.camera.MV_CC_SetGrabStrategy(sdk.MV_GrabStrategy_LatestImagesOnly), "设置取帧策略")
            check_mvs(self.camera.MV_CC_StartGrabbing(), "开始采集")
            self.streaming = True
        except BaseException:
            self.release()
            raise

    def read(self):
        sdk = self.sdk
        frame = sdk.MV_FRAME_OUT()
        deadline = time.monotonic() + 3.0
        while True:
            result = self.camera.MV_CC_GetImageBuffer(frame, 200)
            if result & 0xffffffff != sdk.MV_E_NODATA:
                check_mvs(result, "读取图像")
                break
            if time.monotonic() >= deadline:
                return False, None

        try:
            info = frame.stFrameInfo
            width, height = info.nWidth, info.nHeight
            if not frame.pBufAddr or width <= 0 or height <= 0 or info.nFrameLen <= 0:
                raise RuntimeError("MVS 返回空帧。")
            frame_bytes = width * height * 3
            if info.enPixelType == sdk.PixelType_Gvsp_BGR8_Packed:
                if info.nFrameLen < frame_bytes:
                    raise RuntimeError("MVS 的 BGR 帧数据不完整。")
                data = ctypes.string_at(frame.pBufAddr, frame_bytes)
                return True, np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3).copy()

            image = np.empty((height, width, 3), dtype=np.uint8)
            convert = sdk.MV_CC_PIXEL_CONVERT_PARAM()
            convert.nWidth, convert.nHeight = width, height
            convert.pSrcData, convert.nSrcDataLen = frame.pBufAddr, info.nFrameLen
            convert.enSrcPixelType = info.enPixelType
            convert.enDstPixelType = sdk.PixelType_Gvsp_BGR8_Packed
            convert.pDstBuffer = image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
            convert.nDstBufferSize = image.nbytes
            check_mvs(self.camera.MV_CC_ConvertPixelType(convert), "转换图像为 BGR")
            if convert.nDstLen != image.nbytes:
                raise RuntimeError("MVS 转色后的 BGR 帧长度异常。")
            return True, image
        finally:
            self.camera.MV_CC_FreeImageBuffer(frame)

    def release(self):
        if self.streaming:
            self.camera.MV_CC_StopGrabbing()
            self.streaming = False
        if self.opened:
            if self.previous_trigger is not None:
                self.camera.MV_CC_SetEnumValue("TriggerMode", self.previous_trigger)
            self.camera.MV_CC_CloseDevice()
            self.opened = False
        if self.created:
            self.camera.MV_CC_DestroyHandle()
            self.created = False
        if self.initialized:
            self.camera.MV_CC_Finalize()
            self.initialized = False
