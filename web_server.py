import io
from queue import Queue
from threading import Thread

import uvicorn
import yaml
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

from LaSSI.Configuration import SentenceRepresentation

app = FastAPI()

@app.get("/")
async def homepage():
    html = """
    <!DOCTYPE html>
<html>
<head>
    <title>Test</title>
</head>
<body>
    <h1>Send Request to Server</h1>
        <form action="" onsubmit="sendMessage(event)">
    <div class="formbuilder-text form-group field-text-1726609082834">
        <label for="text-1726609082834" class="formbuilder-text-label">Dataset Name
            <br>
        </label>
        <input type="text" class="form-control" name="text-1726609082834" access="false" id="text-1726609082834">
    </div>
    <div class="formbuilder-textarea form-group field-textarea-1726609047934">
        <label for="textarea-1726609047934" class="formbuilder-textarea-label">Sentences</label>
        <textarea type="textarea" class="form-control" name="textarea-1726609047934" access="false" id="textarea-1726609047934"></textarea>
    </div>
    <div class="formbuilder-select form-group field-select-1726609103516">
        <label for="select-1726609103516" class="formbuilder-select-label">Select</label>
        <select class="form-control" name="select-1726609103516" id="select-1726609103516">
            <option value="0" selected="true" id="select-1726609103516-0">FullText</option>
            <option value="1" id="select-1726609103516-1">Simplistic Graphs</option>
            <option value="2" id="select-1726609103516-2">Logical Graphs</option>
            <option value="3" id="select-1726609103516-3">Logical</option>
        </select>
    </div>
            <button>Send</button>
        </form>
    <ul id='messages'>
    </ul>
    <script>
        var ws = new WebSocket(`ws://localhost:5000/ws`);
        console.log("Connected")
        ws.onmessage = function (event) {
            var messages = document.getElementById('messages')
            var message = document.createElement('li')
            var content = document.createTextNode(event.data)
            message.appendChild(content)
            messages.appendChild(message)
        };
        console.log("Send Mesage")
        function sendMessage(event) {
            var input = document.getElementById("text-1726609082834").value;
            var separateLines = document.getElementById("textarea-1726609047934").value.split('\\n');
            var casus = document.getElementById("select-1726609103516").value;
            var obj = new Object();
   obj.dataset_name = input;
   obj.dataset_request  = separateLines;
   obj.sentence_representation = casus;
   var jsonString= JSON.stringify(obj);
            ws.send(jsonString)
            jsonString = ''
            event.preventDefault()
        }
    </script>
</body>
</html>"""
    return HTMLResponse(html)


class Logger():
    def __init__(self):
        self.q = Queue()
    def __call__(self, msg:str):
        self.q.put(msg)

def threaded_function(dataset_name, yaml_request, transformation, logger):
    with io.StringIO(yaml_request) as f:
        from LaSSI.LaSSI import LaSSI
        pipeline = LaSSI(dataset_name, "connection.yaml", transformation, f, logger)
        pipeline.run()
        pipeline.close()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger = Logger()
    rich_request = await websocket.receive_json()
    dataset_name = rich_request["dataset_name"]
    yaml_request = yaml.dump(rich_request["dataset_request"])
    transformation = SentenceRepresentation(int(rich_request["sentence_representation"]))
    thread = Thread(target=threaded_function, args=(dataset_name, yaml_request, transformation, logger))
    thread.start()
    while True:
        msg = logger.q.get()
        await websocket.send_text(msg)
        logger.q.task_done()
        if msg == "~~DONE~~":
            break
    thread.join()
    websocket.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)