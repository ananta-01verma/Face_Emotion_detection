#!/usr/bin/env python3
"""
Script to forcefully stop the Face Emotion Detection application
and release camera resources
"""

import cv2
import os
import subprocess
import sys

def kill_python_processes():
    """Kill all Python processes"""
    try:
        if os.name == 'nt':  # Windows
            subprocess.run(['taskkill', '/f', '/im', 'python.exe'], 
                          capture_output=True, text=True)
            print("✅ Killed all Python processes")
        else:  # Linux/Mac
            subprocess.run(['pkill', '-f', 'python'], 
                          capture_output=True, text=True)
            print("✅ Killed all Python processes")
    except Exception as e:
        print(f"Error killing processes: {e}")

def release_camera():
    """Release camera resources"""
    try:
        # Try to release any open camera
        cap = cv2.VideoCapture(0)
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Camera resources released")
    except Exception as e:
        print(f"Error releasing camera: {e}")

def kill_port_5000():
    """Kill processes on port 5000"""
    try:
        if os.name == 'nt':  # Windows
            # Find process using port 5000
            result = subprocess.run(['netstat', '-ano'], 
                                  capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if ':5000' in line and 'LISTENING' in line:
                    parts = line.split()
                    if len(parts) > 4:
                        pid = parts[-1]
                        subprocess.run(['taskkill', '/f', '/pid', pid], 
                                      capture_output=True, text=True)
                        print(f"✅ Killed process {pid} on port 5000")
        else:  # Linux/Mac
            subprocess.run(['lsof', '-ti:5000', '|', 'xargs', 'kill', '-9'], 
                          shell=True, capture_output=True, text=True)
            print("✅ Killed processes on port 5000")
    except Exception as e:
        print(f"Error killing port 5000: {e}")

def main():
    print("🛑 Forcefully stopping Face Emotion Detection...")
    
    # Release camera resources
    release_camera()
    
    # Kill processes on port 5000
    kill_port_5000()
    
    # Kill all Python processes
    kill_python_processes()
    
    print("✅ Application stopped forcefully!")
    print("📹 Camera should now be released")

if __name__ == "__main__":
    main()
