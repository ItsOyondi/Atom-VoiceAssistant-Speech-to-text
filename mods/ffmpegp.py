from flask import Flask, render_template_string, request
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>Audio Recorder</title>
</head>
<body>
    <script>
    var my_div = document.createElement("DIV");
    var my_p = document.createElement("P");
    var my_btn = document.createElement("BUTTON");
    var t = document.createTextNode("Press to start recording");
    
    my_btn.appendChild(t);
    my_div.appendChild(my_btn);
    document.body.appendChild(my_div);
    
    var base64data = 0;
    var reader;
    var recorder, gumStream;
    var recordButton = my_btn;
    
    var handleSuccess = function(stream) {
      gumStream = stream;
      var options = {
        mimeType : 'audio/webm;codecs=opus'
      };            
      recorder = new MediaRecorder(stream);
      recorder.ondataavailable = function(e) {            
        var url = URL.createObjectURL(e.data);
        var preview = document.createElement('audio');
        preview.controls = true;
        preview.src = url;
        document.body.appendChild(preview);
    
        reader = new FileReader();
        reader.readAsDataURL(e.data); 
        reader.onloadend = function() {
          base64data = reader.result;

          // Send base64data to server
          fetch('/upload_audio', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({ audio: base64data })
          }).then(function(response) {
            if (response.ok) {
              console.log('Audio uploaded successfully');
              recordButton.innerText = "Recording saved!";
            } else {
              console.log('Upload failed');
              recordButton.innerText = "Upload failed";
            }
          });
        }
      };
      recorder.start();
    };
    
    recordButton.innerText = "Recording... press to stop";
    
    navigator.mediaDevices.getUserMedia({audio: true}).then(handleSuccess);
    
    function toggleRecording() {
      if (recorder && recorder.state == "recording") {
          recorder.stop();
          gumStream.getAudioTracks()[0].stop();
          recordButton.innerText = "Saving the recording... please wait!";
      }
    }
    
    recordButton.onclick = ()=>{
      toggleRecording();
    }
    </script>
</body>
</html>
""")

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    data = request.get_json()
    audio_base64 = data['audio']
    # The audio_base64 is a data URL like "data:audio/webm;base64,..."
    header, encoded = audio_base64.split(',', 1)
    audio_data = base64.b64decode(encoded)
    # Save the audio data to a file
    with open('recorded_audio.webm', 'wb') as f:
        f.write(audio_data)
    return '', 200

if __name__ == '__main__':
    app.run(debug=True)
