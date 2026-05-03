"""
NxV - Phone to Pi File Transfer
transfer_server.py

Run this on the Pi, then visit the URL on your phone browser
to upload your video file directly over WiFi.

Run:
  python transfer_server.py

Then on your phone browser:
  http://10.71.53.113:8080
"""



import os
from flask import Flask, request, redirect


app = Flask(__name__)

SAVE_DIR = "/home/emmanuel/camera_project/research"
os.makedirs(SAVE_DIR, exist_ok=True)

@app.route('/')
def index():
    return '''
    <html>
    <head>
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <style>
        body {
          font-family: monospace;
          background: #0d1117;
          color: #c9d1d9;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          min-height: 100vh;
          margin: 0;
          padding: 20px;
        }
        h2 { color: #00ff88; letter-spacing: 3px; margin-bottom: 30px; }
        input[type=file] { margin: 20px 0; color: #c9d1d9; width: 100%; }
        button {
          background: transparent;
          border: 1px solid #00ff88;
          color: #00ff88;
          padding: 12px 30px;
          font-family: monospace;
          font-size: 14px;
          letter-spacing: 2px;
          cursor: pointer;
          width: 100%;
          margin-top: 10px;
        }
        p { color: #4a5568; font-size: 12px; text-align: center; }
      </style>
    </head>
    <body>
      <h2>NxV FILE TRANSFER</h2>
      <p>Select your video file from your phone gallery</p>
      <form method="POST" action="/upload" enctype="multipart/form-data">
        <input type="file" name="file" accept="video/*,image/*"/>
        <button type="submit">UPLOAD TO PI</button>
      </form>
      <p style="margin-top:20px">Connected to Pi at 10.71.53.113:8080</p>
    </body>
    </html>
    '''

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file selected', 400

    f        = request.files['file']
    filename = f.filename

    if filename == '':
        return 'Empty filename', 400

    save_path = os.path.join(SAVE_DIR, filename)
    f.save(save_path)

    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"[Transfer] File saved → {save_path} ({size_mb:.1f} MB)")

    return f'''
    <html>
    <head>
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <style>
        body {{
          font-family: monospace;
          background: #0d1117;
          color: #c9d1d9;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          min-height: 100vh;
          margin: 0;
          padding: 20px;
          text-align: center;
        }}
        h2 {{ color: #00ff88; letter-spacing: 3px; }}
        p  {{ color: #4a5568; font-size: 13px; margin-top: 10px; }}
        .path {{ color: #4fc3f7; font-size: 12px; margin-top: 20px; }}
      </style>
    </head>
    <body>
      <h2>✓ UPLOAD COMPLETE</h2>
      <p>File: {filename}</p>
      <p>Size: {size_mb:.1f} MB</p>
      <div class="path">Saved to Pi:<br>/home/emmanuel/camera_project/research/{filename}</div>
      <p style="margin-top:30px">You can close this tab now<br>and go back to your Pi terminal</p>
    </body>
    </html>
    '''

if __name__ == '__main__':
    import socket
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "10.71.53.113"

    print(f"""
[NxV Transfer Server]
─────────────────────────────────
  Open this on your phone browser:
  http://{local_ip}:8080
─────────────────────────────────
  Waiting for upload...
  Files saved to: {SAVE_DIR}
  Press Ctrl+C when done
""")
    app.run(host='0.0.0.0', port=8080, debug=False)
