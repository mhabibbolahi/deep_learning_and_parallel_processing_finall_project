import importlib
import shutil
import uvicorn
import numpy as np
from fastapi import FastAPI, Form, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI()

templates = Jinja2Templates(directory="templates")
files = {
    'TBP': ['tbp_1', 'tbp_2', 'tbp_3', 'tbp_4', 'tbp_5', 'tbp_6', 'tbp_7', 'tbp_8'],
    'PBP': ['pbp_1', 'pbp_2', 'pbp_3', 'pbp_4', 'pbp_5', 'pbp_6', 'pbp_7', 'pbp_8']
}
big_data_files_path = ['fcn_use_model','cnn_use_model']

class Item(BaseModel):
    part: str
    sections: int
    scenario: str


@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/submit-form/", response_class=HTMLResponse)
async def submit_form(request: Request, part: str = Form(...), sections: int = Form(...), scenario: str = Form(...)):
    my_module = importlib.import_module('Parallelism.' + files[part][sections - 1])
    function_name = scenario
    function_to_call = getattr(my_module, function_name)
    result = function_to_call()
    item = Item(part=part, sections=sections, scenario=scenario)
    return templates.TemplateResponse("result.html", {"request": request, "item": item, 'data': np.array(result)})


@app.post("/submit-json/", response_class=HTMLResponse)
async def submit_json(request: Request, item: Item):
    return templates.TemplateResponse("result.html", {"request": request, "item": item})


@app.post("/upload-file/", response_class=HTMLResponse)
async def upload_file(request: Request, network: int = Form(...), file: UploadFile = File(...)):
    file_location = f"static/{file.filename}"
    my_module = importlib.import_module(big_data_files_path[network-1])
    with open(file_location, "wb") as f:
        shutil.copyfileobj(file.file, f)
    if network == 2:
        result = my_module.main(file_location)
    else:
        result =''
        file_location = my_module.main(file_location)

    networks=['FCN','CNN']
    return templates.TemplateResponse("image_result.html",
                                      {"request": request, "file_url": file_location, "network": networks[network-1],"result":result})


app.mount("/static", StaticFiles(directory="static"), name="static")
if __name__ == "__main__":

    uvicorn.run(app, host="127.0.0.1", port=8000)
