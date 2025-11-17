# alert_system.py
import datetime, paho.mqtt.client as mqtt

def send_alert(event_type, message):
    print(f"[{datetime.datetime.now():%H:%M:%S}] ALERT: {event_type} - {message}")

def send_mqtt(event_type, message):
    client = mqtt.Client()
    client.connect("broker.hivemq.com", 1883, 60)
    payload = {"event": event_type, "message": message}
    client.publish("home/ai_ids/alerts", str(payload))
    client.disconnect()
