import bleak
import asyncio


# Device instance
class DeviceModel:
    # deviceName
    deviceName = "My device"

    # Device Data Dictionary
    deviceData = {}

    # Is the device turned on?
    isOpen = False

    # Temporary array
    TempBytes = []

    def __init__(self, deviceName, BLEDevice, callback_method):
        print("Initialize device model")
        self.deviceName = deviceName
        self.BLEDevice = BLEDevice
        self.client = None
        self.writer_characteristic = None
        self.isOpen = False
        self.callback_method = callback_method
        self.deviceData = {}

    # Set device data
    def set(self, key, value):
        # Saving device data to key values
        self.deviceData[key] = value

    # Obtain device data
    def get(self, key):
        # Get data from the key value, return None if there is no data
        if key in self.deviceData:
            return self.deviceData[key]
        else:
            return None

    # Delete device data
    def remove(self, key):
        # Deleting a device key
        del self.deviceData[key]

    # open Device
    async def openDevice(self):
        print("Opening device......")
        # Obtain the services and characteristic of the device
        async with bleak.BleakClient(self.BLEDevice, timeout=15) as client:
            self.client = client
            self.isOpen = True
            # Device UUID constant
            target_service_uuid = "0000ffe5-0000-1000-8000-00805f9a34fb"
            target_characteristic_uuid_read = "0000ffe4-0000-1000-8000-00805f9a34fb"
            target_characteristic_uuid_write = "0000ffe9-0000-1000-8000-00805f9a34fb"
            notify_characteristic = None

            print("Matching services......")
            for service in client.services:
                if service.uuid == target_service_uuid:
                    print(f"Service: {service}")
                    print("Matching characteristic......")
                    for characteristic in service.characteristics:
                        if characteristic.uuid == target_characteristic_uuid_read:
                            notify_characteristic = characteristic
                        if characteristic.uuid == target_characteristic_uuid_write:
                            self.writer_characteristic = characteristic
                    if notify_characteristic:
                        break

            if self.writer_characteristic:
                # Reading magnetic field quaternions
                print("Reading magnetic field quaternions")
                await asyncio.sleep(3)
                asyncio.create_task(self.sendDataTh())

            if notify_characteristic:
                print(f"Characteristic: {notify_characteristic}")
                # Set up notifications to receive data
                await client.start_notify(
                    notify_characteristic.uuid, self.onDataReceived
                )

                # Keep connected and open
                try:
                    while self.isOpen:
                        await asyncio.sleep(1)
                except asyncio.CancelledError:
                    pass
                finally:
                    # Stop notification on exit
                    await client.stop_notify(notify_characteristic.uuid)
            else:
                print("No matching services or characteristic found")

    # close Device
    def closeDevice(self):
        self.isOpen = False
        print("The device is turned off")

    async def sendDataTh(self):
        while self.isOpen:
            await self.readReg(0x3A)
            await asyncio.sleep(0.1)
            await self.readReg(0x51)
            await asyncio.sleep(0.1)

    # Serial port data processing
    def onDataReceived(self, sender, data):
        tempdata = bytes.fromhex(data.hex())
        for var in tempdata:
            self.TempBytes.append(var)
            if len(self.TempBytes) == 1 and self.TempBytes[0] != 0x55:
                del self.TempBytes[0]
                continue
            if len(self.TempBytes) == 2 and (
                self.TempBytes[1] != 0x61 and self.TempBytes[1] != 0x71
            ):
                del self.TempBytes[0]
                continue
            if len(self.TempBytes) == 20:
                self.processData(self.TempBytes)
                self.TempBytes.clear()

    # data analysis
    def processData(self, Bytes):
        if Bytes[1] == 0x61:
            Ax = self.getSignInt16(Bytes[3] << 8 | Bytes[2]) / 32768 * 16
            Ay = self.getSignInt16(Bytes[5] << 8 | Bytes[4]) / 32768 * 16
            Az = self.getSignInt16(Bytes[7] << 8 | Bytes[6]) / 32768 * 16
            Gx = self.getSignInt16(Bytes[9] << 8 | Bytes[8]) / 32768 * 2000
            Gy = self.getSignInt16(Bytes[11] << 8 | Bytes[10]) / 32768 * 2000
            Gz = self.getSignInt16(Bytes[13] << 8 | Bytes[12]) / 32768 * 2000
            AngX = self.getSignInt16(Bytes[15] << 8 | Bytes[14]) / 32768 * 180
            AngY = self.getSignInt16(Bytes[17] << 8 | Bytes[16]) / 32768 * 180
            AngZ = self.getSignInt16(Bytes[19] << 8 | Bytes[18]) / 32768 * 180
            self.set("AccX", round(Ax, 3))
            self.set("AccY", round(Ay, 3))
            self.set("AccZ", round(Az, 3))
            self.set("AsX", round(Gx, 3))
            self.set("AsY", round(Gy, 3))
            self.set("AsZ", round(Gz, 3))
            self.set("AngX", round(AngX, 3))
            self.set("AngY", round(AngY, 3))
            self.set("AngZ", round(AngZ, 3))
            self.callback_method(self)
        else:
            # magnetic field
            if Bytes[2] == 0x3A:
                Hx = self.getSignInt16(Bytes[5] << 8 | Bytes[4]) / 120
                Hy = self.getSignInt16(Bytes[7] << 8 | Bytes[6]) / 120
                Hz = self.getSignInt16(Bytes[9] << 8 | Bytes[8]) / 120
                self.set("HX", round(Hx, 3))
                self.set("HY", round(Hy, 3))
                self.set("HZ", round(Hz, 3))
            # Quaternion
            elif Bytes[2] == 0x51:
                Q0 = self.getSignInt16(Bytes[5] << 8 | Bytes[4]) / 32768
                Q1 = self.getSignInt16(Bytes[7] << 8 | Bytes[6]) / 32768
                Q2 = self.getSignInt16(Bytes[9] << 8 | Bytes[8]) / 32768
                Q3 = self.getSignInt16(Bytes[11] << 8 | Bytes[10]) / 32768
                self.set("Q0", round(Q0, 5))
                self.set("Q1", round(Q1, 5))
                self.set("Q2", round(Q2, 5))
                self.set("Q3", round(Q3, 5))
            else:
                pass

    # Obtain int16 signed number
    @staticmethod
    def getSignInt16(num):
        if num >= pow(2, 15):
            num -= pow(2, 16)
        return num

    # Send serial port data
    async def sendData(self, data):
        try:
            if self.client.is_connected and self.writer_characteristic is not None:
                await self.client.write_gatt_char(
                    self.writer_characteristic.uuid, bytes(data)
                )
        except Exception as ex:
            print(ex)

    # read register
    async def readReg(self, regAddr):
        # Encapsulate read instructions and send data to the serial port
        await self.sendData(self.get_readBytes(regAddr))

    # Write Register
    async def writeReg(self, regAddr, sValue):
        # unlock
        self.unlock()
        # Delay 100ms
        await asyncio.sleep(0.1)
        # Encapsulate write instructions and send data to the serial port
        await self.sendData(self.get_writeBytes(regAddr, sValue))
        # Delay 100ms
        await asyncio.sleep(0.1)
        # save
        self.save()

    # Read instruction package
    @staticmethod
    def get_readBytes(regAddr):
        # 初始化
        tempBytes = [None] * 5
        tempBytes[0] = 0xFF
        tempBytes[1] = 0xAA
        tempBytes[2] = 0x27
        tempBytes[3] = regAddr
        tempBytes[4] = 0
        return tempBytes

    # Write instruction package
    @staticmethod
    def get_writeBytes(regAddr, rValue):
        # 初始化
        tempBytes = [None] * 5
        tempBytes[0] = 0xFF
        tempBytes[1] = 0xAA
        tempBytes[2] = regAddr
        tempBytes[3] = rValue & 0xFF
        tempBytes[4] = rValue >> 8
        return tempBytes

    # Unlock
    def unlock(self):
        cmd = self.get_writeBytes(0x69, 0xB588)
        self.sendData(cmd)

    # Save
    def save(self):
        cmd = self.get_writeBytes(0x00, 0x0000)
        self.sendData(cmd)
