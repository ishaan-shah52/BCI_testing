from bleak import BleakScanner
import asyncio

async def main():
    devices = await BleakScanner.discover()
    for device in devices:
        print(f"Name: {device.name}, MAC Address: {device.address}")

asyncio.run(main())

#currently returns: D5:A4:BE:DD:BC:89