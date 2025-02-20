import asyncio
import bleak
import device_model
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tqdm import tqdm


class DataCollector:
    def __init__(self):
        self.BLEDevice = None
        self.devices = []
        self.FEATURE_COUNT = 6
        self.model = load_model("modelo_gestos.h5")
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.array(
            [0.97076255, 0.9445494, 0.97515005, 0.9454171, 0.94420373, 0.9554501]
        )
        self.scaler.scale_ = np.array(
            [0.9701005, 0.94347113, 0.98497283, 0.9526623, 0.9534778, 0.96096903]
        )
        self.WINDOW_SIZE = 150
        self.y_pred = []
        self.y_true = []

    async def scan(self, devices):
        find = []
        print("Searching for Bluetooth devices......")
        try:
            self.devices = await bleak.BleakScanner.discover(timeout=20.0)
            print("Search ended")
            for device in devices:
                if device.name is not None and "WT" in device.name:
                    find.append(device)
                    print(device)
            if len(find) == 0:
                print("No devices found in this search!")
            else:
                user_input = input(
                    "Please enter the Mac address you want to connect to (e.g. DF:E9:1F:2C:BD:59)： "
                )
                for device in devices:
                    if device.address == user_input:
                        self.BLEDevice = device
                        break
        except Exception as ex:
            print(f"Bluetooth search failed to start: {ex}")

    async def scanByMac(self, device_mac: str):
        print("Searching for Bluetooth devices......")
        self.BLEDevice = await bleak.BleakScanner.find_device_by_address(
            device_mac, timeout=20
        )

    def updateData(self, DeviceModel, data_buffer: list):
        if len(data_buffer) == 0:
            self.y_true.append(
                int(input("Digite o número do gesto que será capturado: "))
            )
            for i in tqdm(range(0, 4), desc="Iniciando coleta em 4 segundos"):
                time.sleep(1)
        if len(data_buffer) == self.WINDOW_SIZE:
            self.y_pred.append(self.classify(data_buffer))
            data_buffer.clear()
            if len(self.y_pred) == 10:
                conf = confusion_matrix(self.y_true, self.y_pred)
                sns.heatmap(conf, annot=True, annot_kws={"size": 12})
                plt.show()
                self.y_pred = []
                self.y_true = []
        else:
            data_buffer.append(
                [
                    DeviceModel.get("AccX"),
                    DeviceModel.get("AccY"),
                    DeviceModel.get("AccZ"),
                    DeviceModel.get("AsX"),
                    DeviceModel.get("AsY"),
                    DeviceModel.get("AsZ"),
                ]
            )

    def classify(self, data_buffer: list):
        if len(data_buffer) == self.WINDOW_SIZE:
            print("Coleta concluída. Processando dados...")
            print(f"Número de amostras coletadas: {len(data_buffer)}")

            # Convert to numpy array
            data_array = np.array(data_buffer, dtype=np.float32)
            # Mostrar dados brutos para depuração
            print("Dados brutos (primeiras 5 amostras):")
            print(data_array[:5])

            # Normaliza os dados
            data_normalized = self.scaler.transform(data_array)

            # Mostrar dados normalizados para depuração
            print("Dados normalizados (primeiras 5 amostras):")
            print(data_normalized[:5])

            # Classifica o gesto
            data_normalized = data_normalized.reshape(
                1, self.WINDOW_SIZE, self.FEATURE_COUNT
            )
            prediction = self.model.predict(data_normalized)
            print("Distribuição das probabilidades:", prediction)
            gesture_detected: int = np.argmax(prediction, axis=1)[0]
            print(f"Gesto detectado: {gesture_detected}")

            # Plotar os dados
            self.plot_data(data_array, gesture_detected)
            return gesture_detected
        else:
            print(
                f"Dados insuficientes para classificação: {len(data_buffer)} amostras."
            )
            data_buffer.clear()
        return None

    def plot_data(self, data_array, gesture):
        plt.figure(figsize=(10, 6))
        x_data = range(len(data_array))
        for i in range(self.FEATURE_COUNT):
            plt.plot(x_data, data_array[:, i], label=f"Feature {i + 1}")
        plt.legend()
        plt.xlabel("Amostra")
        plt.ylabel("Valor")
        plt.title(f"Dados do Sensor - Gesto Detectado: {gesture}")
        plt.show()


if __name__ == "__main__":
    data_collector = DataCollector()
    # Method 1:Broadcast search and connect Bluetooth devices
    # asyncio.run(data_collector.scan())

    # Method 2: Specify MAC address to search and connect devices
    # asyncio.run(scanByMac("DF:DE:EA:97:12:6F"))
    asyncio.run(data_collector.scanByMac("C0:87:95:47:FC:4B"))

    if data_collector.BLEDevice is not None:
        data_collector.device = device_model.DeviceModel(
            "MyBLE5.0", data_collector.BLEDevice, data_collector.updateData
        )
        asyncio.run(data_collector.device.openDevice())
        # data_collector.collect_and_classify(device)
    else:
        print("BLE device was not found!!")
