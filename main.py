import asyncio
import bleak
import device_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from time import sleep
from tqdm import tqdm

FEATURE_COUNT = 6
WINDOW_SIZE = 150

# Inicializa o scaler (substitua pelos valores usados no treinamento)
scaler = StandardScaler()
scaler.mean_ = np.array(
    [0.97076255, 0.9445494, 0.97515005, 0.9454171, 0.94420373, 0.9554501]
)
scaler.scale_ = np.array(
    [0.9701005, 0.94347113, 0.98497283, 0.9526623, 0.9534778, 0.96096903]
)


# Scan Bluetooth devices and filter names
async def scan():
    find = []
    print("Searching for Bluetooth devices......")
    try:
        devices = await bleak.BleakScanner.discover(timeout=20.0)
        print("Search ended")
        for device in devices:
            if device.name is not None and "WT" in device.name:
                find.append(device)
                print(device)
        if len(find) == 0:
            print("No devices found in this search!")
        else:
            user_input = input(
                "Please enter the Mac address you want to connect to (e.g. DF:E9:1F:2C:BD:59)："
            )
            for device in devices:
                if device.address == user_input:
                    return device
    except Exception as ex:
        print("Bluetooth search failed to start")
        print(ex)


# Specify MAC address to search and connect devices
async def scanByMac(device_mac):
    print("Searching for Bluetooth devices......")
    BLEDevice = await bleak.BleakScanner.find_device_by_address(device_mac, timeout=20)
    return BLEDevice


def colect(DeviceModel):
    coletas = 2
    data_buffer = []
    y_true = []
    y_pred = []
    for gesture in [i for i in range(0, 8)]:
        for j in range(0, coletas):
            y_true.append(gesture)
            while len(data_buffer) < WINDOW_SIZE:
                # if len(data_buffer) == 0:
                #     desc = f"Iniciando coleta do gesto {gesture} (Coleta {j + 1} de {coletas}) em 5 segundos"
                #     for _ in tqdm(range(0, 5), desc=desc):
                #         sleep(1)

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

                # sleep(0.05)

            y_pred.append(classify(data_buffer))
            data_buffer = []

    conf = confusion_matrix(y_true, y_pred) # , labels=[0, 1, 2, 3, 4, 5, 6, 7])
    sns.heatmap(conf, annot=True, annot_kws={"size": 12})
    plt.show()
    # plot_confusion_matrix(dict(real=y_true, predicted=y_pred))


def classify(data_buffer):
    if len(data_buffer) == WINDOW_SIZE:
        print("Coleta concluída. Processando dados...")
        print(f"Número de amostras coletadas: {len(data_buffer)}")
        # Convert to numpy array
        data_array = np.array(data_buffer, dtype=np.float32)

        # Mostrar dados brutos para depuração
        print("Dados brutos (primeiras 5 amostras):")
        print(data_array[:5])

        # Normaliza os dados
        data_normalized = scaler.transform(data_array)

        # Mostrar dados normalizados para depuração
        print("Dados normalizados (primeiras 5 amostras):")
        print(data_normalized[:5])

        # Classifica o gesto
        data_normalized = data_normalized.reshape(1, WINDOW_SIZE, FEATURE_COUNT)
        prediction = model.predict(data_normalized)
        print("Distribuição das probabilidades:", prediction)

        gesture_detected = np.argmax(prediction, axis=1)[0]
        print(f"Gesto detectado: {gesture_detected}")

        # Plotar os dados
        plot_data(data_array, gesture_detected)
        return gesture_detected
    else:
        print(f"Dados insuficientes para classificação: {len(data_buffer)} amostras.")
        sleep(0.01)  # Evita uso excessivo de CPU

    return None


def plot_data(data_array, gesture):
    plt.figure(figsize=(10, 6))
    x_data = range(len(data_array))
    for i in range(FEATURE_COUNT):
        plt.plot(x_data, data_array[:, i], label=f"Feature {i + 1}")
    plt.legend()
    plt.xlabel("Amostra")
    plt.ylabel("Valor")
    plt.title(f"Dados do Sensor - Gesto Detectado: {gesture}")
    plt.show()


def plot_confusion_matrix(results_data: dict):
    df = pd.DataFrame(results_data, columns=["real", "predicted"])
    conf = pd.crosstab(df["real"], df["predicted"], rownames=["real"], colnames=["predicted"])
    sns.heatmap(conf, annot=True, annot_kws={"size": 12})


if __name__ == "__main__":
    # Method 1:Broadcast search and connect Bluetooth devices
    # asyncio.run(scan())

    # # Method 2: Specify MAC address to search and connect devices
    # asyncio.run(scanByMac("DF:DE:EA:97:12:6F"))
    BLEDevice = asyncio.run(scanByMac("C0:87:95:47:FC:4B"))

    model = load_model("modelo_gestos.h5")

    if BLEDevice is not None:
        device = device_model.DeviceModel("MyBle5.0", BLEDevice, colect)
        asyncio.run(device.openDevice())
    else:
        print("This BLEDevice was not found!!")
