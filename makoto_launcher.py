!pip install streamlit -q
import os, time, subprocess

# Kill any existing streamlit
os.system("pkill -f streamlit 2>/dev/null")
time.sleep(2)

# Launch streamlit
subprocess.Popen(
    ["streamlit", "run", "makoto_dashboard.py",
     "--server.port", "8501",
     "--server.headless", "true",
     "--server.enableCORS", "false",
     "--server.enableXsrfProtection", "false"],
    stdout=open("/tmp/streamlit.log", "w"),
    stderr=subprocess.STDOUT
)

# Wait for server to actually bind
print("Starting Streamlit...")
for i in range(15):
    time.sleep(1)
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:8501", timeout=1)
        print(f"Server ready after {i+1}s")
        break
    except:
        print(f"Waiting... {i+1}s")

from google.colab.output import serve_kernel_port_as_window
serve_kernel_port_as_window(8501)
