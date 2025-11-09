from list_devices import list_audio_devices
from auth import generate_token
import numpy as np
import sounddevice as sd
from livekit.rtc import apm
from livekit import rtc
from dotenv import load_dotenv
import signal
import threading
import time
import sys
import argparse
import asyncio
import logging
import os
cat << EOF > audio3.py
#!/usr/bin/env python3

if sys.platform.startswith('win'):
    import msvcrt
else:
    import select
    import termios
    import tty


load_dotenv()

LIVEKIT_URL = os.environ.get("LIVEKIT_URL")
ROOM_NAME = os.environ.get("ROOM_NAME")

SAMPLE_RATE = 48000
INPUT_CHANNELS = 1
OUTPUT_CHANNELS = 2
FRAME_SAMPLES = 480
BLOCKSIZE = 960  # 20ms at 48kHz

MAX_AUDIO_BAR = 20
INPUT_DB_MIN = -70.0
INPUT_DB_MAX = 0.0
FPS = 16


def _esc(*codes: int) -> str:
    return "\033[" + ";".join(str(c) for c in codes) + "m"


def _normalize_db(amplitude_db: float, db_min: float, db_max: float) -> float:
    amplitude_db = max(db_min, min(amplitude_db, db_max))
    return (amplitude_db - db_min) / (db_max - db_min)


class AudioStreamer:
    def __init__(self, enable_aec: bool = True, loop: asyncio.AbstractEventLoop = None):
        self.enable_aec = enable_aec
        self.running = True
        self.logger = logging.getLogger(__name__)
        self.loop = loop

        self.is_muted = False
        self.mute_lock = threading.Lock()

        self.input_callback_count = 0
        self.output_callback_count = 0
        self.frames_processed = 0
        self.frames_sent_to_livekit = 0
        self.dropped_frames = 0
        self.last_debug_time = time.time()

        self.input_stream = None
        self.output_stream = None

        self.source = rtc.AudioSource(SAMPLE_RATE, INPUT_CHANNELS)
        self.room = None

        self.audio_processor = None
        if enable_aec:
            print("[INIT] Creating APM with AEC...")
            self.audio_processor = apm.AudioProcessingModule(
                echo_cancellation=True,
                noise_suppression=True,
                high_pass_filter=True,
                auto_gain_control=True
            )
            print("[INIT] APM created successfully")

        self.output_buffer = bytearray()
        self.output_lock = threading.Lock()
        self.audio_input_queue = asyncio.Queue(maxsize=500)

        self.output_delay = 0.0
        self.input_delay = 0.0

        self.micro_db = INPUT_DB_MIN
        self.input_device_name = "Unknown"

        self.participants = {}
        self.participants_lock = threading.Lock()

        self.meter_running = True
        self.keyboard_thread = None

        self.stdout_lock = threading.Lock()
        self.active_remote_participant_id = None
        self.remote_playback_enabled = True

    def start_audio_devices(self):
        print("\n[START] === STARTING AUDIO DEVICES ===")
        try:
            print("[DEV] Listing all audio devices...")
            list_audio_devices()

            input_device = 1   # USB mic (hw:1,0)
            output_device = 0  # WM8962 (hw:0,0)
            print(
                f"[DEV] Using input: {input_device}, output: {output_device}")
            device_info = sd.query_devices(input_device)

            self.input_device_name = device_info.get("name", "Microphone")
            print(f"[DEV] Device info: {device_info}")

            print(
                f"[STREAM] Creating INPUT stream: rate={SAMPLE_RATE}, ch={INPUT_CHANNELS}, block={BLOCKSIZE}")
            self.input_stream = sd.InputStream(
                callback=self._input_callback,
                dtype="int16",
                channels=INPUT_CHANNELS,
                device=input_device,
                samplerate=SAMPLE_RATE,
                blocksize=BLOCKSIZE,
            )
            self.input_stream.start()
            print("[STREAM] INPUT stream STARTED")

            print(
                f"[STREAM] Creating OUTPUT stream: rate={SAMPLE_RATE}, ch={OUTPUT_CHANNELS}, block={BLOCKSIZE}")
            self.output_stream = sd.OutputStream(
                callback=self._output_callback,
                dtype="int16",
                channels=OUTPUT_CHANNELS,
                device=output_device,
                samplerate=SAMPLE_RATE,
                blocksize=BLOCKSIZE,
            )
            self.output_stream.start()
            print("[STREAM] OUTPUT stream STARTED")

            time.sleep(0.2)
            print(f"[STATUS] Input active: {self.input_stream.active}")
            print(f"[STATUS] Output active: {self.output_stream.active}")

        except Exception as e:
            print(f"[FATAL] FAILED to start audio devices: {e}")
            import traceback
            traceback.print_exc()
            raise

    def stop_audio_devices(self):
        print("\n[STOP] === STOPPING AUDIO DEVICES ===")
        self.meter_running = False
        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()
            print("[STOP] Input stream closed")
        if self.output_stream:
            self.output_stream.stop()
            self.output_stream.close()
            print("[STOP] Output stream closed")
        print("Audio devices stopped")

    def toggle_mute(self):
        with self.mute_lock:
            self.is_muted = not self.is_muted
            print(f"[MUTE] Microphone {'MUTED' if self.is_muted else 'LIVE'}")

    def start_keyboard_handler(self):
        """Start keyboard input handler in a separate thread"""
        def keyboard_handler():
            is_windows = sys.platform.startswith('win')
            try:
                if not is_windows:
                    # Save original terminal settings
                    old_settings = termios.tcgetattr(sys.stdin)
                    tty.setraw(sys.stdin.fileno())

                while self.running:
                    if is_windows:
                        if msvcrt.kbhit():
                            key = msvcrt.getch().decode('ascii')
                            if key.lower() == 'm':
                                self.toggle_mute()
                            elif key.lower() == 'q':
                                print("Quit requested by user")
                                self.running = False
                                break
                            elif key == '\x03':  # Ctrl+C
                                break
                        time.sleep(0.1)
                    else:
                        if select.select([sys.stdin], [], [], 0.1)[0]:
                            key = sys.stdin.read(1)
                            if key.lower() == 'm':
                                self.toggle_mute()
                            elif key.lower() == 'q':
                                print("Quit requested by user")
                                self.running = False
                                break
                            elif key == '\x03':  # Ctrl+C
                                break

            except Exception as e:
                print(f"Keyboard handler error: {e}")
            finally:
                # Restore terminal settings on non-Windows
                if not is_windows:
                    try:
                        termios.tcsetattr(
                            sys.stdin, termios.TCSADRAIN, old_settings)
                    except:
                        pass

        self.keyboard_thread = threading.Thread(
            target=keyboard_handler, daemon=True)
        self.keyboard_thread.start()
        print("Keyboard handler started - Press 'm' to toggle mute, 'q' to quit")

    def stop_keyboard_handler(self):
        """Stop keyboard handler"""
        if self.keyboard_thread and self.keyboard_thread.is_alive():
            # The thread will stop when self.running = False
            self.keyboard_thread.join(timeout=1.0)

    def _input_callback(self, indata: np.ndarray, frame_count: int, time_info, status) -> None:
        self.input_callback_count += 1
        if self.input_callback_count <= 5 or self.input_callback_count % 100 == 0:
            print(f"[IN] #{self.input_callback_count} | frames={frame_count} | shape={indata.shape} | max={np.max(np.abs(indata))} | mean={np.mean(np.abs(indata)):.2f}")

        if status:
            print(f"[IN] STATUS WARNING: {status}")

        if not self.running:
            return

        with self.mute_lock:
            is_muted = self.is_muted

        processed_indata = indata.copy()
        if is_muted:
            processed_indata.fill(0)

        self.input_delay = time_info.currentTime - time_info.inputBufferAdcTime
        total_delay = self.output_delay + self.input_delay

        if self.audio_processor:
            try:
                self.audio_processor.set_stream_delay_ms(
                    int(total_delay * 1000))
            except RuntimeError as e:
                if not hasattr(self, '_delay_error_logged'):
                    print(f"Failed to set APM stream delay: {e}")
                    self._delay_error_logged = True

        num_frames = frame_count // FRAME_SAMPLES

        for i in range(num_frames):
            start = i * FRAME_SAMPLES
            end = start + FRAME_SAMPLES
            if end > frame_count:
                break

            original_chunk = indata[start:end, 0]
            capture_chunk = processed_indata[start:end, 0]

            capture_frame = rtc.AudioFrame(
                data=capture_chunk.tobytes(),
                samples_per_channel=FRAME_SAMPLES,
                sample_rate=SAMPLE_RATE,
                num_channels=1,
            )

            self.frames_processed += 1

            if self.audio_processor:
                try:
                    self.audio_processor.process_stream(capture_frame)
                except Exception as e:
                    if self.frames_processed <= 5:
                        print(f"Error processing audio stream with AEC: {e}")

            rms = np.sqrt(np.mean(original_chunk.astype(np.float32) ** 2))
            max_int16 = np.iinfo(np.int16).max
            self.micro_db = 20.0 * np.log10(rms / max_int16 + 1e-6)

            if self.audio_input_queue.full():
                try:
                    self.audio_input_queue.get_nowait()
                    self.dropped_frames += 1
                except:
                    pass

            if self.loop and not self.loop.is_closed():
                try:
                    self.loop.call_soon_threadsafe(
                        self.audio_input_queue.put_nowait, capture_frame)
                    self.frames_sent_to_livekit += 1
                except Exception as e:
                    self.dropped_frames += 1
                    if self.frames_processed <= 5:
                        print(f"Failed to queue audio frame: {e}")

    def _output_callback(self, outdata: np.ndarray, frame_count: int, time_info, status) -> None:
        self.output_callback_count += 1
        if self.output_callback_count <= 5 or self.output_callback_count % 100 == 0:
            print(
                f"[OUT] #{self.output_callback_count} | frames={frame_count} | buffer_size={len(self.output_buffer)}")

        if status:
            print(f"[OUT] STATUS WARNING: {status}")

        if not self.running:
            outdata.fill(0)
            return

        self.output_delay = time_info.outputBufferDacTime - time_info.currentTime

        bytes_needed = frame_count * 2  # mono int16 bytes

        with self.output_lock:
            if len(self.output_buffer) < bytes_needed:
                available_bytes = len(self.output_buffer)
                if available_bytes > 0:
                    mono = np.frombuffer(
                        self.output_buffer[:available_bytes], dtype=np.int16)
                    num_samples = len(mono)
                    outdata[:num_samples, 0] = mono
                    outdata[:num_samples, 1] = mono
                    outdata[num_samples:, :] = 0
                    del self.output_buffer[:available_bytes]
                else:
                    outdata.fill(0)
            else:
                chunk = self.output_buffer[:bytes_needed]
                mono = np.frombuffer(chunk, dtype=np.int16)
                outdata[:, 0] = mono
                outdata[:, 1] = mono
                del self.output_buffer[:bytes_needed]

        if self.audio_processor:
            num_chunks = frame_count // FRAME_SAMPLES
            for i in range(num_chunks):
                start = i * FRAME_SAMPLES
                end = start + FRAME_SAMPLES
                if end > frame_count:
                    break

                render_chunk = np.mean(
                    outdata[start:end, :], axis=1).astype(np.int16)
                render_frame = rtc.AudioFrame(
                    data=render_chunk.tobytes(),
                    samples_per_channel=FRAME_SAMPLES,
                    sample_rate=SAMPLE_RATE,
                    num_channels=1,
                )
                try:
                    self.audio_processor.process_reverse_stream(render_frame)
                except Exception as e:
                    if self.output_callback_count <= 5:
                        print(f"Error processing reverse stream with AEC: {e}")

    def print_audio_meter(self):
        if not self.meter_running:
            return
        self._print_simple_meter()

    def _print_simple_meter(self):
        amplitude_db = _normalize_db(self.micro_db, INPUT_DB_MIN, INPUT_DB_MAX)
        nb_bar = round(amplitude_db * MAX_AUDIO_BAR)
        color_code = 31 if amplitude_db > 0.75 else 33 if amplitude_db > 0.5 else 32
        bar = "#" * nb_bar + "-" * (MAX_AUDIO_BAR - nb_bar)

        with self.mute_lock:
            status = "MUTED" if self.is_muted else "LIVE"

        meter_text = f"I:{self.input_callback_count} O:{self.output_callback_count} Q:{self.audio_input_queue.qsize()} Db:{self.micro_db:.1f} [{bar}] {status}"

        with self.stdout_lock:
            sys.stdout.write(f"\033[2K\r{meter_text}")
            sys.stdout.flush()

    def init_terminal(self):
        with self.stdout_lock:
            sys.stdout.write("\033[?25l")
            sys.stdout.flush()

    def restore_terminal(self):
        with self.stdout_lock:
            sys.stdout.write("\033[2K\r\033[?25h")
            sys.stdout.flush()


async def main(participant_name: str, enable_aec: bool = True):
    print("=== LIVEKIT AUDIO STREAMER (DEBUG MODE) ===")
    loop = asyncio.get_running_loop()

    if not LIVEKIT_URL or not ROOM_NAME:
        print("[ERROR] Missing LIVEKIT_URL or ROOM_NAME in .env")
        return

    streamer = AudioStreamer(enable_aec, loop=loop)
    room = rtc.Room(loop=loop)
    streamer.room = room

    async def audio_processing_task():
        sent = 0
        print("[TASK] Audio processing started")
        while streamer.running:
            try:
                frames = []
                frame = await asyncio.wait_for(streamer.audio_input_queue.get(), timeout=0.1)
                frames.append(frame)
                while len(frames) < 20 and not streamer.audio_input_queue.empty():
                    try:
                        frames.append(streamer.audio_input_queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                for f in frames:
                    await streamer.source.capture_frame(f)
                    sent += 1
                if sent <= 5 or sent % 100 == 0:
                    print(
                        f"[SEND] Sent {sent} frames | Queue size: {streamer.audio_input_queue.qsize()} | Dropped: {streamer.dropped_frames}")
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"[SEND] Error: {e}")
                break
        print("[TASK] Audio processing ended")

    async def meter_task():
        print("[TASK] Meter started")
        while streamer.running and streamer.meter_running:
            streamer.print_audio_meter()
            await asyncio.sleep(1 / FPS)
        print("[TASK] Meter ended")

    async def receive_audio_frames(stream: rtc.AudioStream, participant: rtc.RemoteParticipant):
        recv = 0
        pid = participant.sid
        name = participant.identity or pid[:8]
        print(f"[RECV] From {name}")
        async for frame_event in stream:
            recv += 1
            if recv <= 5 or recv % 100 == 0:
                print(
                    f"[RECV] Frame {recv} from {name}, len={len(frame_event.frame.data)}")
            if streamer.active_remote_participant_id == pid:
                audio_samples = np.frombuffer(
                    frame_event.frame.data, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_samples.astype(np.float32)**2))
                db = 20 * np.log10(rms / 32767 + 1e-6)
                with streamer.participants_lock:
                    streamer.participants[pid] = {
                        'name': name, 'db_level': db, 'last_update': time.time()}
                with streamer.output_lock:
                    streamer.output_buffer.extend(frame_event.frame.data)
        print(f"[RECV] Ended for {name}, total {recv}")
        with streamer.participants_lock:
            streamer.participants.pop(pid, None)

    @room.on("track_subscribed")
    def on_track_subscribed(track, pub, participant):
        print(
            f"[EVENT] Subscribed track {pub.sid} from {participant.identity}, kind={track.kind}")
        if track.kind == rtc.TrackKind.KIND_AUDIO and streamer.active_remote_participant_id is None:
            streamer.active_remote_participant_id = participant.sid
            print(f"[PLAY] Activating {participant.identity}")
            audio_stream = rtc.AudioStream(
                track, sample_rate=SAMPLE_RATE, num_channels=1)
            asyncio.create_task(receive_audio_frames(
                audio_stream, participant))

    @room.on("connected")
    def on_connected():
        print("[LIVEKIT] Connected to room")

    @room.on("disconnected")
    def on_disconnected(reason):
        print(f"[LIVEKIT] Disconnected: {reason}")

    try:
        print("[MAIN] Starting devices...")
        streamer.start_audio_devices()
        print("[MAIN] Starting keyboard...")
        streamer.start_keyboard_handler()
        streamer.init_terminal()

        print("[LIVEKIT] Generating token...")
        token = generate_token(ROOM_NAME, participant_name, participant_name)

        print("[LIVEKIT] Connecting...")
        await room.connect(LIVEKIT_URL, token)
        print("[LIVEKIT] Connected")

        print("[LIVEKIT] Publishing mic...")
        track = rtc.LocalAudioTrack.create_audio_track("mic", streamer.source)
        await room.local_participant.publish_track(track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
        print("[LIVEKIT] Published")

        audio_task = asyncio.create_task(audio_processing_task())
        meter_task = asyncio.create_task(meter_task())

        print("=== Streaming active. Ctrl+C to stop ===")
        while streamer.running:
            await asyncio.sleep(1)
    except Exception as e:
        print(f"[FATAL] Main error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        streamer.running = False
        try:
            audio_task.cancel()
        except:
            pass
        try:
            meter_task.cancel()
        except:
            pass
        streamer.stop_audio_devices()
        streamer.stop_keyboard_handler()
        await room.disconnect()
        streamer.restore_terminal()
        print("[MAIN] Cleanup complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="yocto-streamer")
    parser.add_argument("--disable-aec", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.name, not args.disable_aec))
EOF
