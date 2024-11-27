from fastapi import FastAPI
import openai
from file_handler import process_file, save_tags_in_redis, save_task_in_redis_vectors
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import redis

app = FastAPI(swagger_ui_parameters={"deepLinking": False})
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)

@app.post('/upload')
async def upload_file(file):

    if file.filename == '':
        return {"error": "File has no name"}

    try:
        file_content = process_file(file)
        tags = await generate_tags_ai(file_content)

        return {"tags": tags}
    except Exception as e:
        return {"error": str(e)}

@app.post('/generate-tags')
async def generate_tags(text):
    try:
        tags = await generate_tags_ai(text)

        return {"tags": tags}
    except Exception as e:
        return {"error": str(e)}

class SaveTagsRequest(BaseModel):
    tags: list[str]
    contentId: int
    contentTitle: str

@app.post('/save_tags')
async def save_tags(request: SaveTagsRequest):
    try:
        response = save_task_in_redis_vectors(request)
        return response
    except Exception as e:
        return {"error": str(e)}


async def generate_tags_ai(content):
    #prompt = f"Analyze the following content and provide a list of 10 related tags:\n\n{content}"
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for generating content tags."},
            {"role": "user", "content": f"Generate highly relevant tags for this course description. These tags will be used to match user search queries and should directly relate to the course content. Avoid generic terms like 'general' or 'miscellaneous'.\n{content}"}
        ]
    )

    tags = response.choices[0].message.content.strip().split('\n')
    return [tag.strip() for tag in tags if tag.strip()][:10]



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)