from openai import OpenAI
import os

os.environ["OPENAI_API_KEY"] = "xxxxx"
os.environ["OPENAI_BASE_URL"] = "xxxx"

client = OpenAI()
# completion = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {
#                     "role": "user",
#                     "content": "Who are you",
#                 }
#             ]
#         )
# print(completion.choices[0].message)


class MyGPT:
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = OpenAI()


    def batch_chat(self, message, temperature=0.0, n=1):
        # print("====input_messages_list:", messages_list)
        res = []
        open_ai_messages_list = []

        open_ai_messages_list.append(
             {"role": "user", "content": message[0]}
            )
        # print("====open_ai_messages_list:",open_ai_messages_list)
        # print("重复回答n次，n是：", n)
        predictions = client.chat.completions.create(
            model= self.model_name, messages= open_ai_messages_list,max_tokens=1,n=n
            )
        # print("----type(predictions)\n",type(predictions)) #  <class 'openai.types.chat.chat_completion.ChatCompletion'>

        # print("-----predictions: ",predictions.choices[0].message.content ) # NO 对的
        res.append(predictions.choices[0].message.content)
        res.append(predictions.choices[1].message.content)
        # res.append(predictions.choices[2].message.content)
        print("-----res数组:", res) # ['Yes', 'No', 'Yes']

        return res



