import os
import asyncio
import requests
from livekit import rtc
from dotenv import load_dotenv
import pyaudio
import threading
import sys
import jwt
import numpy as np

running = True

class LiveKitAudioClient:
    def __init__(self):
        load_dotenv()
        
        self.jwt_token = os.getenv("JWT_TOKEN")
        self.fastapi_url = os.getenv("FASTAPI_URL", "http://localhost:8000")
        self.participant_name = os.getenv("PARTICIPANT_NAME", "client")
        
        if not self.jwt_token:
            raise ValueError("JWT_TOKEN environment variable is required")
        
        self.log_jwt_details()
        self.check_audio_devices()

    def log_jwt_details(self):
        try:
            decoded = jwt.decode(self.jwt_token, options={"verify_signature": False})
            print("JWT Token Details (FastAPI Auth):")
            print(f"  Room: {decoded.get('room', 'Not specified')}")
            print(f"  Participant Identity: {decoded.get('sub', 'Not specified')}")
            print(f"  Can Publish: {decoded.get('video', {}).get('can_publish', 'Not specified')}")
            print(f"  Can Subscribe: {decoded.get('video', {}).get('can_subscribe', 'Not specified')}")
        except Exception as e:
            print(f"Error decoding JWT token (FastAPI Auth): {e}")

    def check_audio_devices(self):
        p = pyaudio.PyAudio()
        print("Available audio devices:")
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            print(f"Device {i}: {device_info['name']}, "
                  f"Input Channels: {device_info['maxInputChannels']}, "
                  f"Output Channels: {device_info['maxOutputChannels']}, "
                  f"Sample Rate: {device_info['defaultSampleRate']}")
        p.terminate()

    def get_livekit_token(self) -> str:
        headers = {
            "Authorization": f"Bearer {self.jwt_token}",
            "Content-Type": "application/json"
        }
        response = requests.post(
            f"{self.fastapi_url}/gettoken/",
            headers=headers,
            timeout=10
        )
        if response.status_code == 200:
            token = response.text.strip('"')
            try:
                decoded = jwt.decode(token, options={"verify_signature": False})
                print("LiveKit Token Details:")
                print(f"  Room: {decoded.get('video', {}).get('room', 'Not specified')}")
                print(f"  Identity: {decoded.get('sub', 'Not specified')}")
                print(f"  Can Publish: {decoded.get('video', {}).get('can_publish', 'Not specified')}")
                print(f"  Can Subscribe: {decoded.get('video', {}).get('can_subscribe', 'Not specified')}")
            except Exception as e:
                print(f"Error decoding LiveKit token: {e}")
            print("Successfully retrieved LiveKit token")
            return token
        else:
            raise Exception(f"Failed to get LiveKit token: {response.status_code}")

    async def connect_with_audio(self):
        global running
        running = True
        capture_task = None
        
        try:
            livekit_token = self.get_livekit_token()
            room = rtc.Room()
            
            # Event handlers
            @room.on("participant_connected")
            def on_participant_connected(participant: rtc.RemoteParticipant):
                print(f"Participant connected: {participant.identity}, SID: {participant.sid}")

            @room.on("participant_disconnected")
            def on_participant_disconnected(participant: rtc.RemoteParticipant):
                print(f"Participant disconnected: {participant.identity}")

            @room.on("track_published")
            def on_track_published(publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
                print(f"Track published: {publication.sid}, Kind: {publication.kind}, "
                      f"Participant: {participant.identity}")
                # Auto-subscribe to audio tracks
                if publication.kind == rtc.TrackKind.KIND_AUDIO:
                    print("Remote audio track detected - auto-subscribing!")
                    publication.set_subscribed(True)

            @room.on("track_subscribed")
            def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
                print(f"Track subscribed: {track.sid}, Kind: {track.kind}, "
                      f"Participant: {participant.identity}")
                if track.kind == rtc.TrackKind.KIND_AUDIO:
                    print("Subscribed to remote audio track - starting playback!")
                    audio_stream = rtc.AudioStream(track, sample_rate=48000, num_channels=1)
                    asyncio.create_task(self.play_audio_stream(audio_stream))

            @room.on("connected")
            def on_connected():
                print("Room connected successfully!")

            @room.on("disconnected")
            def on_disconnected(reason):
                print(f"Room disconnected: {reason}")

            print(f"Connecting to LiveKit server...")
            await room.connect(
                url=os.getenv("LIVEKIT_URL", "wss://my-first-project-7ci5luaf.livekit.cloud"),
                token=livekit_token
            )
            print(f"Connected to room: {room.name}")
            print(f"Local participant identity: {room.local_participant.identity}")
            
            # Wait a moment for the room to fully initialize
            await asyncio.sleep(1)
            
            # Create and publish audio track
            audio_source = rtc.AudioSource(sample_rate=48000, num_channels=1)
            track = rtc.LocalAudioTrack.create_audio_track("microphone", audio_source)
            
            # Create publish options
            options = rtc.TrackPublishOptions()
            options.source = rtc.TrackSource.SOURCE_MICROPHONE
            
            publication = await room.local_participant.publish_track(track, options)
            print(f"Audio track published successfully: SID {publication.sid}")
            
            # Start audio capture in async task instead of thread
            capture_task = asyncio.create_task(self.capture_audio(audio_source))
            
            print("Audio system ready! You should now be able to communicate with the agent.")
            print("Press Ctrl+C to disconnect.")
            
            # Keep the connection alive and monitor status
            try:
                while running:
                    participants = len(room.remote_participants)
                    if participants > 0:
                        print(f"Active participants: {participants}")
                    await asyncio.sleep(10)
            except (KeyboardInterrupt, asyncio.CancelledError):
                print("\nShutting down...")
                running = False
                
        except Exception as e:
            print(f"Connection error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            running = False
            if capture_task and not capture_task.done():
                capture_task.cancel()
                try:
                    await capture_task
                except asyncio.CancelledError:
                    pass
            try:
                await room.disconnect()
                print("Disconnected from room")
            except:
                pass

    async def capture_audio(self, audio_source: rtc.AudioSource):
        """Capture audio from microphone and send to LiveKit"""
        global running
        p = pyaudio.PyAudio()
        
        # Audio parameters
        SAMPLE_RATE = 48000
        FRAME_SIZE = 960  # 20ms at 48kHz
        INPUT_DEVICE = 9  # Your microphone device
        
        stream = None
        frame_count = 0
        
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=FRAME_SIZE,
                input_device_index=INPUT_DEVICE
            )
            
            print(f"Microphone capture started (device {INPUT_DEVICE})")
            
            while running:
                try:
                    # Read audio data
                    data = stream.read(FRAME_SIZE, exception_on_overflow=False)
                    
                    # Create AudioFrame and send to LiveKit
                    frame = rtc.AudioFrame(
                        data=data,
                        sample_rate=SAMPLE_RATE,
                        num_channels=1,
                        samples_per_channel=FRAME_SIZE
                    )
                    
                    await audio_source.capture_frame(frame)
                    frame_count += 1
                    
                    # Log progress less frequently
                    if frame_count % 50 == 0:
                        print(f"Captured {frame_count} audio frames")
                        
                except Exception as e:
                    print(f"Audio capture error: {e}")
                    break
                    
        except Exception as e:
            print(f"Failed to initialize audio capture: {e}")
        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except:
                    pass
            p.terminate()
            print(f"Audio capture stopped. Total frames: {frame_count}")

    async def play_audio_stream(self, audio_stream: rtc.AudioStream):
        """Play audio received from LiveKit"""
        global running
        p = pyaudio.PyAudio()
        
        # Audio parameters
        SAMPLE_RATE = 48000
        OUTPUT_DEVICE = 12  # Your speaker device
        
        output_stream = None
        frame_count = 0
        
        try:
            output_stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                output=True,
                output_device_index=OUTPUT_DEVICE
            )
            
            print(f"Audio playback started (device {OUTPUT_DEVICE})")
            
            async for event in audio_stream:
                if not running:
                    break
                    
                if hasattr(event, 'frame'):
                    frame = event.frame
                    try:
                        # Convert frame data to bytes and play
                        audio_data = bytes(frame.data)
                        output_stream.write(audio_data)
                        frame_count += 1
                        
                        if frame_count % 50 == 0:
                            print(f"Played {frame_count} audio frames")
                            
                    except Exception as e:
                        print(f"Audio playback error: {e}")
                        
        except Exception as e:
            print(f"Audio playback initialization error: {e}")
        finally:
            if output_stream:
                try:
                    output_stream.stop_stream()
                    output_stream.close()
                except:
                    pass
            p.terminate()
            print(f"Audio playback stopped. Total frames: {frame_count}")

async def main():
    client = LiveKitAudioClient()
    
    print("Starting LiveKit audio client...")
    await client.connect_with_audio()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
    finally:
        running = False