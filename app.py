import os
import json
import base64
import time
import subprocess
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
import requests
import azure.cognitiveservices.speech as speechsdk

# Traditional LangChain imports for conversation memory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request/response models
class ChatRequest(BaseModel):
    message: Optional[str] = None

class Message(BaseModel):
    text: str
    audio: str
    lipsync: dict
    facialExpression: str
    animation: str

class ChatResponse(BaseModel):
    messages: List[Message]

# Initialize environment variables and clients
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')
AZURE_SPEECH_KEY = os.getenv('AZURE_SPEECH_KEY')
AZURE_SPEECH_REGION = os.getenv('AZURE_SPEECH_REGION')

# Validate required environment variables
if not AZURE_OPENAI_API_KEY:
    raise EnvironmentError("AZURE_OPENAI_API_KEY environment variable is not set")
if not AZURE_OPENAI_ENDPOINT:
    raise EnvironmentError("AZURE_OPENAI_ENDPOINT environment variable is not set")
if not AZURE_OPENAI_DEPLOYMENT_NAME:
    raise EnvironmentError("AZURE_OPENAI_DEPLOYMENT_NAME environment variable is not set")
if not AZURE_SPEECH_KEY:
    raise EnvironmentError("AZURE_SPEECH_KEY environment variable is not set")
if not AZURE_SPEECH_REGION:
    raise EnvironmentError("AZURE_SPEECH_REGION environment variable is not set")

# Configure Azure Speech Service
speech_config = speechsdk.SpeechConfig(
    subscription=AZURE_SPEECH_KEY,
    region=AZURE_SPEECH_REGION
)
speech_config.speech_synthesis_voice_name = "en-US-GuyNeural"  # Male voice
speech_config.set_speech_synthesis_output_format(
    speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm  # 16kHz for better lip sync compatibility
)
print(f"Azure Speech Service configured with region: {AZURE_SPEECH_REGION}")
if AZURE_SPEECH_KEY:
    print("Azure Speech Key is set")

# Initialize Azure OpenAI client
llm = AzureChatOpenAI(
    openai_api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    temperature=0.7
)

# Traditional conversation memory setup
conversation_template = """You are StackBot, a warm and empathetic virtual companion.
Keep responses under 50 words. Express emotions in your responses that can be reflected
in facial expressions like 'smile', 'sad', 'neutral', etc.
Do not use any emojis, emoticons, or special characters.
Use only plain text with words to express emotions and feelings.
Remember our conversation history and build upon it naturally.

Current conversation:
{history}
Human: {input}
StackBot:"""

conversation_prompt = PromptTemplate(
    input_variables=["history", "input"], 
    template=conversation_template
)

def exec_command(command):
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        if e.stderr:
            print(f"Command error output: {e.stderr}")
        return None

def audio_file_to_base64(file_path):
    try:
        if not os.path.exists(file_path):
            print(f"Audio file not found: {file_path}")
            return None
            
        with open(file_path, 'rb') as audio_file:
            audio_data = audio_file.read()
            if not audio_data:
                print("Audio file is empty")
                return None
                
            base64_audio = base64.b64encode(audio_data).decode('utf-8')
            if not base64_audio:
                print("Base64 encoding failed")
                return None
                
            return base64_audio
    except Exception as e:
        print(f"Error reading audio file: {e}")
        return None

def read_json_transcript(file_path):
    try:
        with open(file_path, 'r') as json_file:
            return json.load(json_file)
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return None

# Optimized lip sync function with longer timeout and better error handling
def lip_sync_message_fast(message):
    try:
        input_wav = f'audios/{message}.wav'
        output_json = f'audios/{message}.json'
        
        if not os.path.exists(input_wav):
            print(f"WAV file not found: {input_wav}")
            return None
        
        # Check if audio file is valid
        try:
            file_size = os.path.getsize(input_wav)
            if file_size < 1000:  # Less than 1KB is probably not valid audio
                print(f"Audio file too small: {file_size} bytes")
                return None
        except OSError:
            print("Error checking audio file size")
            return None
        
        # Try to generate lip sync data with extended timeout
        rhubarb_path = './bin/Rhubarb-Lip-Sync-1.13.0-Linux/rhubarb'
            
        if os.path.exists(rhubarb_path):
            os.chmod(rhubarb_path, 0o755)
            
            print(f"Starting lip sync generation for {input_wav}")
            
            # Use simpler Rhubarb settings with longer timeout
            sync_result = exec_command(
                f'timeout 30s {rhubarb_path} '  # Increased timeout to 30 seconds
                f'-f json '
                f'-o {output_json} '
                f'{input_wav}'  # Simplified command - removed complex flags
            )
            
            if sync_result is not None and os.path.exists(output_json):
                # Verify the JSON file is valid
                try:
                    with open(output_json, 'r') as f:
                        json_data = json.load(f)
                        if 'mouthCues' in json_data:
                            print(f"Lip sync generated successfully with {len(json_data['mouthCues'])} mouth cues")
                            return True
                        else:
                            print("Generated JSON doesn't contain mouthCues")
                            return None
                except json.JSONDecodeError:
                    print("Generated file is not valid JSON")
                    return None
            else:
                print("Lip sync generation failed or timed out")
                return None
        else:
            print("Rhubarb binary not found")
            return None
            
    except Exception as e:
        print(f"Error in lip sync process: {e}")
        return None

def text_to_speech(text, output_path, max_retries=2):
    for attempt in range(max_retries):
        try:
            audio_config = speechsdk.AudioConfig(filename=output_path)
            speech_synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=speech_config, 
                audio_config=audio_config
            )

            escaped_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            ssml = f"""
            <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
                <voice name="en-US-GuyNeural">
                    <prosody rate="0%" pitch="0%" volume="+10%">
                        {escaped_text}
                    </prosody>
                </voice>
            </speak>
            """

            result = speech_synthesizer.speak_ssml(ssml)

            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                print("Speech synthesis succeeded")
                
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    print(f"Audio file created successfully: {output_path}")
                    return True
                else:
                    print("Audio file is missing or empty")
                    continue
            else:
                print(f"Speech synthesis failed: {result.reason}")
                return False

        except Exception as e:
            print(f"TTS error: {str(e)}")
            return False

    return False

@app.get("/")
def hello():
    return "Hello World!"

@app.get("/voices")
def get_voices():
    return {
        "voices": [
            {"name": "Jenny", "id": "en-US-JennyNeural"},
            {"name": "Guy", "id": "en-US-GuyNeural"},
            {"name": "Aria", "id": "en-US-AriaNeural"},
        ]
    }

def process_lip_sync_data(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        if 'mouthCues' not in data:
            return data
            
        # Sort mouth cues by time
        cues = sorted(data['mouthCues'], key=lambda x: x['start'])
        
        # Smooth transitions
        for i in range(1, len(cues)):
            if cues[i]['start'] > cues[i-1]['end']:
                cues[i-1]['end'] = cues[i]['start']
            
            if cues[i]['value'] != cues[i-1]['value']:
                mid_time = (cues[i]['start'] + cues[i-1]['end']) / 2
                cues[i-1]['end'] = mid_time
                cues[i]['start'] = mid_time
        
        data['mouthCues'] = cues
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        return data
    except Exception as e:
        print(f"Error processing lip sync data: {e}")
        return None

# Updated Connection Manager with conversation memory
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.conversations: Dict[WebSocket, ConversationChain] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Create a new conversation chain with memory for this connection
        memory = ConversationBufferMemory()
        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            prompt=conversation_prompt,
            verbose=True  # Enable to see conversation history in logs
        )
        self.conversations[websocket] = conversation
        print(f"Created new conversation chain for connection")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        # Clean up conversation memory
        if websocket in self.conversations:
            del self.conversations[websocket]
            print("Cleaned up conversation memory")

    def get_conversation(self, websocket: WebSocket) -> ConversationChain:
        return self.conversations.get(websocket)

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    print("WebSocket connection established")
    
    try:
        while True:
            try:
                data = await websocket.receive_text()
                request_data = json.loads(data)
                
                if not isinstance(request_data, dict) or 'message' not in request_data:
                    await websocket.send_json({
                        "error": "Invalid message format",
                        "messages": []
                    })
                    continue

                user_message = request_data["message"]
                print(f"Processing user message: '{user_message}'")

                # Get the conversation chain for this connection
                conversation = manager.get_conversation(websocket)
                if not conversation:
                    print("No conversation found for this connection")
                    await websocket.send_json({
                        "messages": [{
                            "text": "I apologize, but I lost our conversation context. Could you try again?",
                            "audio": "",
                            "lipsync": {"mouthCues": []},
                            "facialExpression": "sad",
                            "animation": "Talking_0"
                        }]
                    })
                    continue

                try:
                    # Use the conversation chain with memory
                    response = conversation.predict(input=user_message)
                    assistant_message = response.strip()
                    print(f"LLM Response: {assistant_message}")
                    
                    # Debug: Print conversation history
                    print(f"Conversation buffer: {conversation.memory.buffer}")
                    
                except Exception as e:
                    print(f"Error calling conversation chain: {e}")
                    await websocket.send_json({
                        "messages": [{
                            "text": "I apologize, but I'm having trouble thinking right now. Could you try again?",
                            "audio": "",
                            "lipsync": {"mouthCues": []},
                            "facialExpression": "sad",
                            "animation": "Talking_0"
                        }]
                    })
                    continue

                # Ensure audios directory exists
                os.makedirs("audios", exist_ok=True)

                # Generate unique filename for this response
                timestamp = int(time.time() * 1000)
                output_path = f"audios/message_{timestamp}.wav"
                
                # Generate speech
                start_time = time.time()
                audio_data = None
                lipsync_data = {"mouthCues": []}
                
                if text_to_speech(assistant_message, output_path):
                    tts_time = time.time() - start_time
                    print(f"TTS completed in {tts_time:.2f}s")
                    
                    audio_data = audio_file_to_base64(output_path)
                    
                    # Generate lip sync with improved timeout
                    lip_sync_start = time.time()
                    lip_sync_result = lip_sync_message_fast(f"message_{timestamp}")
                    if lip_sync_result:
                        processed_data = process_lip_sync_data(f"audios/message_{timestamp}.json")
                        if processed_data:
                            lipsync_data = processed_data
                            lip_sync_time = time.time() - lip_sync_start
                            print(f"Lip sync completed in {lip_sync_time:.2f}s")
                        else:
                            print("Failed to process lip sync data")
                    else:
                        print("Failed to generate lip sync")
                    
                    # Send response immediately
                    await websocket.send_json({
                        "messages": [{
                            "text": assistant_message,
                            "audio": audio_data,
                            "lipsync": lipsync_data,
                            "facialExpression": "smile",
                            "animation": "Talking_0"
                        }]
                    })
                    
                    # Clean up old files in background
                    try:
                        for old_file in os.listdir("audios"):
                            if old_file.startswith("message_") and not old_file.startswith(f"message_{timestamp}"):
                                file_path = os.path.join("audios", old_file)
                                if os.path.getctime(file_path) < time.time() - 300:  # Delete files older than 5 minutes
                                    os.remove(file_path)
                    except Exception as cleanup_error:
                        print(f"Cleanup error: {cleanup_error}")
                        
                else:
                    # Send response without audio if TTS failed
                    await websocket.send_json({
                        "messages": [{
                            "text": assistant_message,
                            "audio": "",
                            "lipsync": {"mouthCues": []},
                            "facialExpression": "neutral",
                            "animation": "Talking_0"
                        }]
                    })

            except WebSocketDisconnect:
                print("WebSocket disconnected during message processing")
                break
            except Exception as e:
                print(f"Error processing message: {e}")
                try:
                    await websocket.send_json({
                        "messages": [{
                            "text": "I apologize, but something went wrong. Please try again.",
                            "audio": "",
                            "lipsync": {"mouthCues": []},
                            "facialExpression": "sad",
                            "animation": "Talking_0"
                        }]
                    })
                except Exception as send_err:
                    print(f"Error sending error message: {send_err}")
                    break

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    finally:
        manager.disconnect(websocket)
        print("WebSocket connection cleaned up")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3001)