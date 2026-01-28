import webview
import threading
import time
import sys
from app import app

# Function to run Flask in a separate thread
def run_server():
    # Run in production mode (debug=False) for smoother app experience
    app.run(port=5000, debug=False)

if __name__ == '__main__':
    # Start Flask server in a daemon thread
    t = threading.Thread(target=run_server)
    t.daemon = True
    t.start()

    # Give server a moment to start
    time.sleep(1)

    # Create a native window pointing to the local server
    # fullscreen=True gives that immersive SOC feel
    webview.create_window('IntruGuard SOC', 'http://127.0.0.1:5000', fullscreen=False, width=1280, height=800)
    
    # Start the GUI loop
    webview.start()
