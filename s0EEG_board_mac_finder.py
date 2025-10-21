"""
Notes:
A bleak scan goes through Bluetooth Low Energy devices like the OpenBCI ganglion
-Wraps the window bluetooth APIs to scan for BLE devices and connect by its address
-Reads and writes data points and stream EEG samples

Bluetooth Low Energy device is a power efficient version of Bluetooth
RSSI is returned which is the signal strength
MAC address is a unique hardware identifier
UUID is a unique identifier in software to find objects or devices
Persisting means to save information for later use
"""

from bleak import BleakScanner
import asyncio

#runs for 5 seconds
async def main():
    devices = await BleakScanner.discover()
    for device in devices:
        print(f"Name: {device.name}, MAC Address: {device.address}")

asyncio.run(main())

#currently returns: D5:A4:BE:DD:BC:89