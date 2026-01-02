"""
Module for text-to-speech conversion.
"""

import pyttsx3
import time
import os
import subprocess
import platform
import sys
from utils import logger
from config import settings

def speak(text):
    """
    Convert text to speech and play it using multiple fallback methods.
    
    Args:
        text (str): Text to be spoken
    """
    if not text:
        return
    
    # Log the text being spoken
    logger.speech(text)
    
    # Try all available speech methods until one works
    success = False
    
    # Method 1: pyttsx3 (primary method)
    if not success:
        success = speak_with_pyttsx3(text)
    
    # Method 2: PowerShell Speech (Windows fallback)
    if not success and platform.system() == 'Windows':
        success = speak_with_powershell(text)
    
    # Method 3: Command-line tools
    if not success:
        success = speak_with_command_line(text)
    
    # If all speech methods fail, we at least showed the text in the console
    if not success:
        logger.error("All speech methods failed. Check your audio setup.")

def speak_with_pyttsx3(text):
    """Try to speak using pyttsx3."""
    try:
        # Create a new engine instance each time for reliability
        engine = pyttsx3.init()
        
        # Configure voice properties
        engine.setProperty('rate', settings.VOICE_RATE - 30)  # Slower for clarity
        engine.setProperty('volume', 1.0)  # Maximum volume
        
        # Set voice gender
        voices = engine.getProperty('voices')
        if voices and len(voices) > 1:
            if settings.VOICE_GENDER.lower() == 'female' and len(voices) > 1:
                engine.setProperty('voice', voices[1].id)
            else:
                engine.setProperty('voice', voices[0].id)
        
        # Add the text to speak
        engine.say(text)
        
        # Force synchronous operation and wait for completion
        engine.runAndWait()
            
        # Need to explicitly stop and dispose
        engine.stop()
        
        time.sleep(0.5)  # Small pause after speech
        return True
        
    except Exception as e:
        logger.error(f"pyttsx3 speech failed: {e}")
        return False

def speak_with_powershell(text):
    """Try to speak using Windows PowerShell."""
    try:
        # Escape single quotes in the text
        safe_text = text.replace("'", "''")
        
        # PowerShell command to speak text
        command = f'powershell -command "Add-Type -AssemblyName System.Speech; ' + \
                 f'$speak = New-Object System.Speech.Synthesis.SpeechSynthesizer; ' + \
                 f'$speak.Speak(\'{safe_text}\');"'
        
        # Execute the command
        subprocess.call(command, shell=True)
        return True
        
    except Exception as e:
        logger.error(f"PowerShell speech failed: {e}")
        return False

def speak_with_command_line(text):
    """Try to speak using platform-specific command-line tools."""
    system = platform.system()
    
    try:
        if system == 'Darwin':  # macOS
            # Use macOS say command (fixed escaping)
            safe_text = text.replace('"', '\\"')
            os.system(f'say "{safe_text}"')
            return True
            
        elif system == 'Linux':
            # Try espeak on Linux (fixed escaping)
            safe_text = text.replace('"', '\\"')
            os.system(f'espeak "{safe_text}"')
            return True
            
        elif system == 'Windows':
            # Additional Windows method - wscript
            # Replace quotes to avoid syntax errors
            safe_text = text.replace('"', '')
            
            vbs_script = 'Dim sapi\nSet sapi=CreateObject("SAPI.SpVoice")\n'
            vbs_script += f'sapi.Speak "{safe_text}"'
            
            # Write the script to a temporary file
            with open('speak_temp.vbs', 'w') as file:
                file.write(vbs_script)
                
            # Execute the script
            os.system('cscript //nologo speak_temp.vbs')
            
            # Clean up
            try:
                os.remove('speak_temp.vbs')
            except:
                pass
                
            return True
            
    except Exception as e:
        logger.error(f"Command-line speech failed: {e}")
        return False
    
    return False
