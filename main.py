import asyncio
import bleak
import device_model
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from time import sleep
from tqdm import tqdm

# 扫描到的设备 Scanned devices
devices = []

# 蓝牙设备 BLEDevice
BLEDevice = None

MODEL_PATH = "modelo_gestos.h5"
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

# Buffer para armazenar os dados
data_buffer = list()

# Variável para armazenar o gesto detectado
gesture_detected = None


# 扫描蓝牙设备并过滤名称
# Scan Bluetooth devices and filter names
async def scan():
    global devices
    global BLEDevice
    find = []
    print("Searching for Bluetooth devices......")
    try:
        devices = await bleak.BleakScanner.discover(timeout=20.0)
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
                    BLEDevice = d
                    break
    except Exception as ex:
        print("Bluetooth search failed to start")
        print(ex)


# 指定MAC地址搜索并连接设备
# Specify MAC address to search and connect devices
async def scanByMac(device_mac):
    global BLEDevice
    print("Searching for Bluetooth devices......")
    BLEDevice = await bleak.BleakScanner.find_device_by_address(device_mac, timeout=20)


# 数据更新时会调用此方法 This method will be called when data is updated
def updateData(DeviceModel):
    # 直接打印出设备数据字典 Directly print out the device data dictionary
    # print(DeviceModel.deviceData)

    # 获得X轴加速度 Obtain X-axis acceleration
    # print(DeviceModel.get("AccX"))

    global data_buffer, gesture_detected

    if len(data_buffer) == 0:
        for _ in tqdm(range(0, 5), desc="Iniciando coleta em 5 segundos"):
            sleep(1)

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

    if len(data_buffer) >= WINDOW_SIZE:
        classify()


def classify():
    global data_buffer, gesture_detected
    if len(data_buffer) >= WINDOW_SIZE:
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

        gesture = np.argmax(prediction, axis=1)[0]
        gesture_detected = gesture
        print(f"Gesto detectado: {gesture}")

        # Plotar os dados
        plot_data(data_array, gesture)
    else:
        print(f"Dados insuficientes para classificação: {len(data_buffer)} amostras.")
        sleep(0.01)  # Evita uso excessivo de CPU

    data_buffer.clear()
    gesture_detected = None


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


if __name__ == "__main__":
    # 方式一：广播搜索和连接蓝牙设备
    # Method 1:Broadcast search and connect Bluetooth devices
    # asyncio.run(scan())

    # # 方式二：指定MAC地址搜索并连接设备
    # # Method 2: Specify MAC address to search and connect devices
    asyncio.run(scanByMac("DF:DE:EA:97:12:6F"))

    model = load_model(MODEL_PATH)

    if BLEDevice is not None:
        # 创建设备 Create device
        device = device_model.DeviceModel("MyBle5.0", BLEDevice, updateData)
        # 开始连接设备 Start connecting devices
        asyncio.run(device.openDevice())
    else:
        print("This BLEDevice was not found!!")
