import win32gui
import win32ui
from ctypes import windll
import numpy as np
import cv2

class WindowCapture:
    def __init__(self, window_title):
        self.window_title = window_title
        self.hwnd = win32gui.FindWindow(None, self.window_title)
        
        if not self.hwnd:
            raise Exception(f"Window with title '{self.window_title}' not found.")

    def grab_frame(self):
        # Get the window size and dimensions
        left, top, right, bot = win32gui.GetWindowRect(self.hwnd)
        width = right - left
        height = bot - top

        # If the window is completely collapsed, return None
        if width == 0 or height == 0:
            return None

        # Get the device context (DC) of the window
        hwndDC = win32gui.GetWindowDC(self.hwnd)
        mfcDC  = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()

        # Create a bitmap object to hold the captured image data
        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
        saveDC.SelectObject(saveBitMap)

        result = windll.user32.PrintWindow(self.hwnd, saveDC.GetSafeHdc(), 2)

        frame = None
        if result == 1:
            # Extract the bitmap data
            bmpinfo = saveBitMap.GetInfo()
            bmpstr = saveBitMap.GetBitmapBits(True)
            
            # Convert the raw bytes into a numpy array for OpenCV
            frame = np.frombuffer(bmpstr, dtype=np.uint8)
            frame.shape = (bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4)
            
            # Drop the Alpha channel to convert BGRA to standard BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, hwndDC)

        return frame