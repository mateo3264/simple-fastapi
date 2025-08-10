from fastapi import FastAPI
import boto3
from google import genai
from pydantic import BaseModel


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

@app.get("/")
def read_root():
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents="explain something interesting"
    )
    return {"message": response.text}

@app.post("/")
def query_llm(item: Item):
    response = gemini_client.models.generate_content(
        model="gemini-1.5-flash-8b",
        contents=item.query
    )

    return {"message": response.text}
