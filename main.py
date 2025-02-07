import asyncio
import bleak
import device_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tqdm import tqdm


class DataCollector:
    def __init__(self):
        self.device = None
        self.devices = list()
        self.BLEDevice = None
        self.data_buffer = list()
        self.gesture_detected = None
        self.BLEDevice = None
        self.MODEL_PATH = "modelo_gestos.h5"
        self.FEATURE_COUNT = 6
        self.WINDOW_SIZE = 150
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.array(
            [0.97076255, 0.9445494, 0.97515005, 0.9454171, 0.94420373, 0.9554501]
        )
        self.scaler.scale_ = np.array(
            [0.9701005, 0.94347113, 0.98497283, 0.9526623, 0.9534778, 0.96096903]
        )
        self.data_buffer = list()
        self.test_data = dict(real=list(), predicted=list())
        self.model = load_model(self.MODEL_PATH)

    async def scan(self, devices):
        find = list()
        print("Searching for Bluetooth devices......")
        try:
            self.devices = await bleak.BleakScanner.discover(timeout=20.0)
            print("Search ended")
            for d in devices:
                if d.name is not None and "WT" in d.name:
                    find.append(d)
                    print(d)
            if len(find) == 0:
                print("No devices found in this search!")
            else:
                user_input = input(
                    "Please enter the Mac address you want to connect to (e.g. DF:E9:1F:2C:BD:59)："
                )
                for d in devices:
                    if d.address == user_input:
                        self.BLEDevice = d
                        break
        except Exception as ex:
            print("Bluetooth search failed to start")
            print(ex)

    async def scanByMac(self, device_mac):
        print("Searching for Bluetooth devices......")
        self.BLEDevice = await bleak.BleakScanner.find_device_by_address(
            device_mac, timeout=20
        )

    async def collect_and_classify(self, device, real):
        for gesture in real:
            for j in range(0, 2):  # 2 coletas por gesto
                self.data_buffer.clear()  # Limpa o buffer *antes* de coletar cada gesto
                print(
                    f"Iniciando coleta do gesto {gesture} (Coleta {j+1}) em 5 segundos"
                )
                for _ in tqdm(range(0, 5)):
                    await asyncio.sleep(1)  # Use asyncio.sleep

                while len(self.data_buffer) < self.WINDOW_SIZE:
                    await asyncio.sleep(0.01)  # Pausa para não consumir muitos recursos

                self.classify(gesture)
                self.data_buffer.clear()  # Limpa o buffer *após* classificar cada gesto

        self.plot_confusion_matrix(self.test_data)
        self.test_data["real"].clear()
        self.test_data["predicted"].clear()
        await device.closeDevice()  # Chamado *depois* de coletar todos os dados

    def updateData(self):
        if (
            self.device.get("AccX") is not None
        ):  # Verifica se os dados existem antes de adicionar
            self.data_buffer.append(
                [
                    self.device.get("AccX"),
                    self.device.get("AccY"),
                    self.device.get("AccZ"),
                    self.device.get("AsX"),
                    self.device.get("AsY"),
                    self.device.get("AsZ"),
                ]
            )

    def classify(self, gesture: int):
        if len(self.data_buffer) >= self.WINDOW_SIZE:
            self.test_data["real"].append(gesture)
            print("Coleta concluída. Processando dados...")
            print(f"Número de amostras coletadas: {len(self.data_buffer)}")
            # Convert to numpy array
            data_array = np.array(self.data_buffer, dtype=np.float32)

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

            gesture = np.argmax(prediction, axis=1)[0]
            gesture_detected = gesture
            self.test_data["predicted"].append(gesture_detected)
            print(f"Gesto detectado: {gesture}")

            # Plotar os dados
            self.plot_data(data_array, gesture)
            self.data_buffer.clear()
        else:
            print(
                f"Dados insuficientes para classificação: {len(self.data_buffer)} amostras."
            )
            # sleep(0.01)  # Evita uso excessivo de CPU

        self.data_buffer.clear()
        self.gesture_detected = None

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

    @staticmethod
    def plot_confusion_matrix(results_data: dict):
        df = pd.DataFrame(results_data, columns=["real", "predicted"])
        conf = pd.crosstab(
            df["real"], df["predicted"], rownames=["real"], colnames=["predicted"]
        )
        sns.heatmap(conf, annot=True, annot_kws={"size": 12})

    async def main(self):
        await self.scanByMac("DF:DE:EA:97:12:6F")

        if self.BLEDevice is not None:
            self.device = device_model.DeviceModel("MyBle5.0", self.BLEDevice, self.updateData)
            await self.device.openDevice()
        else:
            print("This BLEDevice was not found!!")


if __name__ == "__main__":
    data_collector = DataCollector.__new__(DataCollector)
    asyncio.run(data_collector.main())
