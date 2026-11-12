import depthai as dai
import cv2
import numpy as np

with dai.Pipeline(True) as pipeline:
    thermal = pipeline.create(dai.node.Thermal)
    qTemp = thermal.temperature.createOutputQueue()
    pipeline.start()
    cv2.namedWindow("thermal", cv2.WINDOW_NORMAL)

    while pipeline.isRunning():
        temp = qTemp.get().getCvFrame().astype(np.float32)   # °C per pixel

        # Option A: auto-scale to the hottest/coldest in view
        norm = cv2.normalize(temp, None, 0, 255, cv2.NORM_MINMAX)

        # Option B (better for spotting people): fixed 20–38 °C range
        # norm = np.clip((temp - 20) / (38 - 20) * 255, 0, 255)

        colour = cv2.applyColorMap(norm.astype(np.uint8), cv2.COLORMAP_INFERNO)
        cv2.imshow("thermal", colour)
        if cv2.waitKey(1) == ord("q"):
            break