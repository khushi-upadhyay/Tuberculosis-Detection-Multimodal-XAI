#!/usr/bin/env python3
"""
Simple web server for Regional TB Detection website
Run this script to host the website locally
"""

import http.server
import socketserver
import webbrowser
import os
import sys
from pathlib import Path

def start_server(port=8000):
    """Start a simple HTTP server to host the website"""
    
    # Change to website directory
    website_dir = Path(__file__).parent
    os.chdir(website_dir)
    
    # Create server
    handler = http.server.SimpleHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"🌐 Regional TB Detection Website")
            print(f"📍 Serving at: http://localhost:{port}")
            print(f"📂 Directory: {website_dir}")
            print(f"🚀 Opening browser...")
            print(f"🛑 Press Ctrl+C to stop the server")
            
            # Open browser automatically
            webbrowser.open(f'http://localhost:{port}')
            
            # Start serving
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print(f"\n👋 Server stopped.")
        sys.exit(0)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"❌ Port {port} is already in use.")
            print(f"🔄 Trying port {port + 1}...")
            start_server(port + 1)
        else:
            print(f"❌ Error starting server: {e}")
            sys.exit(1)

if __name__ == "__main__":
    start_server()
