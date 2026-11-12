import time
from c4002 import C4002Sensor

PORT = "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A50285BI-if00-port0"   # or the /dev/serial/by-id/... path

with C4002Sensor(port=PORT, baudrate=115200) as sensor:
    for _ in range(20):
        data = sensor.read_packet()
        if data:
            print(data.target_state_name, data.ambient_light_lux)
        time.sleep(0.5)