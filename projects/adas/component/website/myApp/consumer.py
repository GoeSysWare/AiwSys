import json
from channels.generic.websocket import WebsocketConsumer,AsyncWebsocketConsumer
from asgiref.sync import async_to_sync
import threading
from channels.layers import get_channel_layer

# from  cyber_py import cyber
# from unit_test_pb2 import ChatterBenchmark


# class ChatConsumer(WebsocketConsumer):
#     def connect(self):
#         self.accept()
#         listen = ListenThread()
#         listen.start()
#         global img,img_cout
#         while 1:
#             if img:
#                 self.send(json.dumps({
#                 'message': img,'img_cout':img_cout
#                 }))
#                 img=''
#                 print(img_cout) 

#     def disconnect(self, close_code):
#         pass

    # def receive(self, text_data):
    #     text_data_json = json.loads(text_data)
    #     message = text_data_json['message']
    #     self.send(text_data=json.dumps({
    #         'message': message
    #     }))
    #print(data)
# 推送consumer
chats=list()
class PushConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.group_name = 'image'
        await self.channel_layer.group_add(
        self.group_name,
        self.channel_name
        )
        await self.accept()
        # PushConsumer.chats[self.group_name].add(self)
        # print(PushConsumer.chats)

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(
        self.group_name,
        self.channel_name
        )

    # print(PushConsumer.chats)

    async def push_message(self, event):
        print(event['message']['cout'])
        await self.send(text_data=json.dumps({
        "message": event['message']['code'],
        "cout":event['message']['cout']
        # "event":event['message']
        }))
    # def chat_message(self, event):
    #     # Handles the "chat.message" event when it's sent to us.
    #     self.send(text_data=event["text"])

# 监听函数
# def create_listen():
#     cyber.init("web")
#     global name
#     name+=1
#     node = cyber.Node(str(name))
#     node.create_reader("channel/chatter",ChatterBenchmark, callback)
#     node.spin()

# # 监听回调函数
# def callback(data):
#     global img,img_cout
#     img=data.content
#     img_cout=data.seq

# # 监听线程
# class ListenThread(threading.Thread):
#     def run(self):
#         create_listen()



 
def push(username, event):
    channel_layer=get_channel_layer()
    async_to_sync(channel_layer.group_send)(
    username,
    {
      "type": "push.message",
      "message": event
    }
  )