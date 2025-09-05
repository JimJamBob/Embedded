# This script connects to a LiveKit server using the provided URL and TOKEN,
# captures audio from the microphone, and publishes it as a track.
#
# Dependencies:
# pip install livekit sounddevice numpy
#
# Note: Ensure your system has a microphone available.
# Run this script in an environment where audio devices are accessible.

import asyncio
import numpy as np
import sounddevice as sd
from livekit import rtc

# LiveKit connection details (provided)
URL = "wss://my-first-project-7ci5luaf.livekit.cloud"
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJuYW1lIjoibmFtZSIsInZpZGVvIjp7InJvb21Kb2luIjp0cnVlLCJyb29tIjoiMTIzIiwiY2FuUHVibGlzaCI6dHJ1ZSwiY2FuU3Vic2NyaWJlIjp0cnVlLCJjYW5QdWJsaXNoRGF0YSI6dHJ1ZX0sInJvb21Db25maWciOnsiYWdlbnRzIjpbeyJhZ2VudE5hbWUiOiJBc3Npc3RhbnQiLCJtZXRhZGF0YSI6InRlc3QtbWV0YWRhdGEifV19LCJzdWIiOiJpZGVudGl0eSIsImlzcyI6IkFQSWtlTWtWWlBMVVduQyIsIm5iZiI6MTc1NzAyNDY0OSwiZXhwIjoxNzU3MDQ2MjQ5fQ.vRiKK8JDMbsdKOytIf7PTbahu8HtGCArMALLlf_MZxk"

# Audio configuration
AUDIO_SAMPLE_RATE = 48000
AUDIO_CHANNELS = 12
AUDIO_BLOCK_SIZE = AUDIO_SAMPLE_RATE // 100  # 10ms frames

async def main():
    room = rtc.Room()

    # Connect to the room
    await room.connect(URL, TOKEN)
    print("Connected to room")

    local_participant = room.local_participant

    # Set up audio source and track
    audio_source = rtc.AudioSource(AUDIO_SAMPLE_RATE, AUDIO_CHANNELS)
    audio_track = rtc.LocalAudioTrack.create_audio_track("microphone", audio_source)
    audio_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
    await local_participant.publish_track(audio_track, audio_options)
    print("Published audio track")

    # Audio capture callback
    def audio_callback(indata, frames, time, status):
        if status:
            print(status)
        # Ensure indata is a 2D array and convert to int16, then to bytes
        if indata.ndim == 2:  # Handle multi-channel data
            audio_data = (indata.reshape(-1) * 32767).astype(np.int16).tobytes()
        else:
            audio_data = (indata * 32767).astype(np.int16).tobytes()
        print(f"Audio data type: {type(audio_data)}, length: {len(audio_data)}")  # Debug print
        frame = rtc.AudioFrame(AUDIO_SAMPLE_RATE, AUDIO_CHANNELS, AUDIO_BLOCK_SIZE, audio_data)
        asyncio.create_task(audio_source.capture_frame(frame))

    # Start audio input stream
    audio_stream = sd.InputStream(
        samplerate=AUDIO_SAMPLE_RATE,
        channels=AUDIO_CHANNELS,
        dtype='float32',  # sounddevice uses float32 by default
        blocksize=AUDIO_BLOCK_SIZE,
        callback=audio_callback
    )
    audio_stream.start()

    # Keep the connection open (adjust sleep time as needed)
    try:
        await asyncio.sleep(3600)  # Run for 1 hour or until interrupted
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        audio_stream.stop()
        await room.disconnect()
        print("Disconnected")

if __name__ == "__main__":
    asyncio.run(main())