from fastapi import FastAPI
import boto3
from google import genai
from pydantic import BaseModel
# import torch
from typing import List
import onnxruntime as ort
import numpy as np


ssm_client = boto3.client("ssm", region_name="us-east-1",
                          )

def get_parameter(name: str):
    try:
        response = ssm_client.get_parameter(Name=name, WithDecryption=True)
        return response["Parameter"]["Value"]
    except Exception as e:
        print(f"An Error occurred: {str(e)}")

app = FastAPI()
google_api_key = get_parameter("/simple-fastapi/GOOGLE_API_KEY")

gemini_client = genai.Client(api_key=google_api_key)


class Item(BaseModel):
    query: str

class PytorchItem(BaseModel):
    input: List[List[List[float]]]

@app.get("/")
def read_root():
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents="explain something interesting"
    )
    return {"message": response.text}

messages = []
@app.post("/")
def query_llm(item: Item):
    messages = messages[-6:]
    messages.append({"role":"user", "parts": [{"text":item.query}]})
    response = gemini_client.models.generate_content(
        model="gemini-1.5-flash-8b",
        contents=messages
    )
    messages.append({"role": "assistant", "parts": [{"text":response.text}]})
    context = "\n---\n".join([m["parts"][0]["text"] for m in messages])
    return {"message": response.text, "context": context}


# model = torch.jit.load("my_simple_nn.pt")
# model.eval()
# @app.post("/pytorch_simple_nn")
# def query_pytorch_nn(item: PytorchItem):
#     print("item")
#     print(item)
#     tensor = torch.tensor(item.input)
#     print(tensor)
#     print(tensor.shape)
#     with torch.no_grad():
#         result = model(tensor)
    
#     return {"result": result.tolist()}

session = ort.InferenceSession("my_simple_model.onnx")
@app.post("/onnx_simple_nn")
def query_onnx_nn(item: PytorchItem):
    input_array = np.array(item.input, dtype=np.float32).reshape(1, 4, 4)
    inputs = {session.get_inputs()[0].name: input_array}
    outputs = session.run(None, inputs)
    print("outputs")
    print(outputs[0])
    return {"prediction": outputs[0].tolist()}