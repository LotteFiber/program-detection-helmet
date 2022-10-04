import paho.mqtt.client as mqtt
from PlateReadHOGOCR import startProgramDetection
import requests
import json

host = "localhost"
port = 1883

def on_connect(self, client, userdata, rc):
    print("MQTT Connected.")
    self.subscribe("start_Program_Detection")
    # self.publish("program_status_update","Hello")

def on_message(client, userdata,msg):
    print("On_message.")
    data = json.loads(msg.payload.decode("utf-8", "strict"))
    pathUrl = str("%s" % data['video_file'])	
    _id = str("%s" % data['id'])	
    print(pathUrl,_id)

    finaldata = startProgramDetection("http://localhost:5000/api/video/"+pathUrl)
    print(finaldata)
    for final_data in finaldata:
        response = requests.post('http://localhost:5000/api/insertdatabyvideo/', data=final_data)
        json_response = response.json()
        print("result => ",json_response)

    response = requests.put('http://localhost:5000/api/video/updateStatusVideo', data={ "_id":_id,"upload_by":"image" })
    json_response = response.json()
    print("status => ",json_response)
    print("End!")
        

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect(host)
client.loop_forever()
