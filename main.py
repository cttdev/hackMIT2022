import time

import adafruit_scd30
import board
import cv2
import jetson_inference
import jetson_utils
from influxdb import InfluxDBClient
from twilio.rest import Client

CO2_MAX = 1900
TEMP_MAX = 40
DETECTION_THRESHOLD = 0.5

# A sliding averag filter
class SlidingAverage:
    def __init__(self,initial,samples):
        self.samples = samples
        self.buffer = [initial] * samples
        self.timestamps = [time.time()] * samples
        self.deltas = [0] * samples
        self.total = initial*samples
        self.totalder = 0 
    def update(self, value):
        self.total -= self.buffer.pop(0)
        self.totalder -= self.deltas.pop(0)
        self.buffer.append(value)
        self.total += value
        self.deltas.append((value-self.buffer[-1])/(time.time() - self.timestamps[-1]))
        self.totalder += (value-self.buffer[-1])/(time.time() - self.timestamps[-1])
        self.timestamps.pop(0)
        self.timestamps.append(time.time())
        return [self.total/self.samples,self.totalder/self.samples]


# Sending the messages
account_sid = 'sid here'
auth_token = 'auth token here'
client = Client(account_sid, auth_token)

def sendMessage(messsage):
    client.messages.create(
        body = messsage,
        from_ = 16073677216,
        to = 14256987268
    )


# Getting the enviromental data
scd = adafruit_scd30.SCD30(board.I2C())

def getEnvironmentData():
    print("Data available?", scd.data_available)
    print("CO2:", scd.CO2, "PPM")
    print("Temperature:", scd.temperature, "degrees C")
    print("Humidity:", scd.relative_humidity, "%%rH")
    return [scd.CO2, scd.temperature, scd.relative_humidity]


# Publishing the data to the dashboard
db_client = InfluxDBClient(host='localhost', port=8086)
db_client.switch_database("home")

def publishDashboardData(env_data, camera_data):
    series = []
    point = {
        "measurement": "co2",
        "tags": {
            "num_people": camera_data[0],
            "num_dogs": camera_data[1]
        },
        "fields": {
            "value": env_data[0]
        }
    }
    series.append(point)

    point = {
        "measurement": "temprature",
        "tags": {
            "num_people": camera_data[0],
            "num_dogs": camera_data[1]
        },
        "fields": {
            "value": env_data[1]
        }
    }
    series.append(point)

    point = {
        "measurement": "relative_humidity",
        "tags": {
            "num_people": camera_data[0],
            "num_dogs": camera_data[1]
        },
        "fields": {
            "value": env_data[2]
        }
    }
    series.append(point)

    num_entities = camera_data[0] + camera_data[1]
    point = {
        "measurement": "num_entities",
        "tags": {
            "num_people": camera_data[0],
            "num_dogs": camera_data[1]
        },
        "fields": {
            "value": num_entities
        }
    }
    series.append(point)

    db_client.write_points(series)


# Start by getting initial enviromental data for the filter
env_start = getEnvironmentData()

carbAvg = SlidingAverage(env_start[0],10)
tempAvg = SlidingAverage(env_start[1],10)
humAvg = SlidingAverage(env_start[2],10)

# Initialize the NN
net = jetson_inference.detectNet("ssd-mobilenet-v2", threshold=DETECTION_THRESHOLD)
camera = jetson_utils.gstCamera(1920, 1080, "/dev/video1")
display = jetson_utils.glDisplay()
lastMessage = 0
# Main loop
while display.IsOpen():
    # Read image from camera, process it, and update the display
    img, width, height = camera.CaptureRGBA()
    detections = net.Detect(img, width, height)
    display.RenderOnce(img, width, height)
    display.SetTitle("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
    
    # Parse the deteched objects
    detected_objects = [net.GetClassDesc(detection.ClassID) for detection in detections]
    num_people = detected_objects.count("person")
    num_dogs = detected_objects.count("dogs")

    # Pull enviromental data
    env = getEnvironmentData()
    carb, dcarb = carbAvg.update(env[0])
    temp, dtemp = tempAvg.update(env[1])
    hum, dhum = humAvg.update(env[2])

    # Publish data to dashboard
    publishDashboardData(env, [num_people, num_dogs])

    # Check if message needs to be sent
    if(num_people > 0 or num_dogs > 0):
        if(env[0] > CO2_MAX or env[1] > TEMP_MAX) and time.time() - lastMessage > 5:
            sendMessage('Warning: Unsafe conditions detected in your vehicle while a living entity is in the car. Please return immediately!')
            lastMessage = time.time()
