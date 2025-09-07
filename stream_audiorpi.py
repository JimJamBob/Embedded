#!/usr/bin/env python3
# Audio streaming script for Raspberry Pi OS
# Dependencies (install with):
#   sudo apt update && sudo apt upgrade
#   sudo apt install python3-pip libportaudio2
#   pip3 install livekit livekit-api sounddevice python-dotenv numpy
# Ensure .env file contains: LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET, ROOM_NAME

import os
import logging
import asyncio
import argparse
import sys
import time
import threading
import signal
import select
import termios
import tty
from dotenv import load_dotenv
from livekit import rtc
from livekit.rtc import apm
import sounddevice as sd
import numpy as np
from auth import generate_token

load_dotenv()
LIVEKIT_URL = os.environ.get("LIVEKIT_URL")
ROOM_NAME = os.environ.get("ROOM_NAME")

SAMPLE_RATE = 48000  # 48kHz; can change to 44100 if needed for USB audio devices
NUM_CHANNELS = 1
FRAME_SAMPLES = 480  # 10ms at 48kHz for APM
BLOCKSIZE = 2048  # 42ms buffer for lower latency on Raspberry Pi

# dB meter settings
MAX_AUDIO_BAR = 20
INPUT_DB_MIN = -70.0
INPUT_DB_MAX = 0.0
FPS = 16

def _esc(*codes: int) -> str:
    return "\033[" + ";".join(str(c) for c in codes) + "m"

def _normalize_db(amplitude_db: float, db_min: float, db_max: float) -> float:
    amplitude_db = max(db_min, min(amplitude_db, db_max))
    return (amplitude_db - db_min) / (db_max - db_min)

def list_audio_devices():
    """List available audio devices for debugging"""
    logger = logging.getLogger(__name__)
    logger.info("Available audio devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if isinstance(device, dict):
            logger.info(f"Device {i}: {device['name']}, "
                        f"Input Channels: {device['max_input_channels']}, "
                        f"Output Channels: {device['max_output_channels']}, "
                        f"Sample Rate: {device['default_samplerate']}")

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
        self.last_debug_time = time.time()
        self.input_stream = None
        self.output_stream = None
        self.source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
        self.room = None
        self.audio_processor = None
        if enable_aec:
            self.logger.info("Initializing Audio Processing Module with Echo Cancellation")
            self.audio_processor = apm.AudioProcessingModule(
                echo_cancellation=True,
                noise_suppression=True,
                high_pass_filter=True,
                auto_gain_control=True
            )
        self.output_buffer = bytearray()
        self.output_lock = threading.Lock()
        self.audio_input_queue = asyncio.Queue(maxsize=50)  # Reduced for Raspberry Pi
        self.output_delay = 0.0
        self.input_delay = 0.0
        self.micro_db = INPUT_DB_MIN
        self.input_device_name = "Microphone"
        self.participants = {}
        self.participants_lock = threading.Lock()
        self.meter_running = True
        self.keyboard_thread = None
        self.stdout_lock = threading.Lock()
        self.meter_line_reserved = False
        self.active_remote_participant_id = None
        self.remote_playback_enabled = True

    def start_audio_devices(self):
        """Initialize and start audio input/output devices for Raspberry Pi"""
        try:
            self.logger.info("Starting audio devices...")
            list_audio_devices()

            # Dynamically select devices; prefer USB or bcm2835 (RPi onboard audio)
            input_device = None
            output_device = None
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if isinstance(device, dict):
                    if device['max_input_channels'] >= NUM_CHANNELS and 'usb' in device['name'].lower():
                        input_device = i
                        self.input_device_name = device['name']
                        break
                    elif device['max_input_channels'] >= NUM_CHANNELS and 'bcm2835' in device['name'].lower():
                        input_device = i
                        self.input_device_name = device['name']

            for i, device in enumerate(devices):
                if isinstance(device, dict):
                    if device['max_output_channels'] >= NUM_CHANNELS and 'usb' in device['name'].lower():
                        output_device = i
                        break
                    elif device['max_output_channels'] >= NUM_CHANNELS and 'bcm2835' in device['name'].lower():
                        output_device = i

            # Fallback to default devices if none found
            if input_device is None:
                input_device = sd.default.device[0]
                self.input_device_name = sd.query_devices(input_device)['name']
            if output_device is None:
                output_device = sd.default.device[1]

            self.logger.info(f"Selected input device: {self.input_device_name} (ID: {input_device})")
            self.logger.info(f"Selected output device: {sd.query_devices(output_device)['name']} (ID: {output_device})")

            device_info = sd.query_devices(input_device)
            if device_info['max_input_channels'] < NUM_CHANNELS:
                self.logger.warning(f"Input device only has {device_info['max_input_channels']} channels, need {NUM_CHANNELS}")

            self.logger.info(f"Creating input stream: rate={SAMPLE_RATE}, channels={NUM_CHANNELS}, blocksize={BLOCKSIZE}")
            try:
                self.input_stream = sd.InputStream(
                    callback=self._input_callback,
                    dtype="int16",
                    channels=NUM_CHANNELS,
                    device=input_device,
                    samplerate=SAMPLE_RATE,
                    blocksize=BLOCKSIZE,
                )
                self.input_stream.start()
                self.logger.info(f"Started audio input: {self.input_device_name}")
            except sd.PortAudioError as e:
                self.logger.error(f"Failed to start input stream: {e}. Check ALSA configuration or try a different device.")
                raise

            self.logger.info(f"Creating output stream: rate={SAMPLE_RATE}, channels={NUM_CHANNELS}, blocksize={BLOCKSIZE}")
            try:
                self.output_stream = sd.OutputStream(
                    callback=self._output_callback,
                    dtype="int16",
                    channels=NUM_CHANNELS,
                    device=output_device,
                    samplerate=SAMPLE_RATE,
                    blocksize=BLOCKSIZE,
                )
                self.output_stream.start()
                self.logger.info("Started audio output")
            except sd.PortAudioError as e:
                self.logger.error(f"Failed to start output stream: {e}. Check ALSA configuration or try a different device.")
                raise

            time.sleep(0.1)
            self.logger.info(f"Input stream active: {self.input_stream.active}")
            self.logger.info(f"Output stream active: {self.output_stream.active}")

        except Exception as e:
            self.logger.error(f"Failed to start audio devices: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def stop_audio_devices(self):
        """Stop and cleanup audio devices"""
        self.logger.info("Stopping audio devices...")
        self.meter_running = False

        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()
            self.input_stream = None
            self.logger.info("Stopped input stream")

        if self.output_stream:
            self.output_stream.stop()
            self.output_stream.close()
            self.output_stream = None
            self.logger.info("Stopped output stream")

        self.logger.info("Audio devices stopped")

    def toggle_mute(self):
        """Toggle microphone mute state"""
        with self.mute_lock:
            self.is_muted = not self.is_muted
            status = "MUTED" if self.is_muted else "LIVE"
            self.logger.info(f"Microphone {status}")

    def start_keyboard_handler(self):
        """Start keyboard input handler in a separate thread"""
        def keyboard_handler():
            try:
                old_settings = termios.tcgetattr(sys.stdin)
                tty.setraw(sys.stdin.fileno())
                while self.running:
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1)
                        if key.lower() == 'm':
                            self.toggle_mute()
                        elif key.lower() == 'q':
                            self.logger.info("Quit requested by user")
                            self.running = False
                            break
                        elif key == '\x03':
                            break
            except Exception as e:
                self.logger.error(f"Keyboard handler error: {e}")
            finally:
                try:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                except:
                    pass

        self.keyboard_thread = threading.Thread(target=keyboard_handler, daemon=True)
        self.keyboard_thread.start()
        self.logger.info("Keyboard handler started - Press 'm' to toggle mute, 'q' to quit")

    def stop_keyboard_handler(self):
        """Stop keyboard handler"""
        pass

    def _input_callback(self, indata: np.ndarray, frame_count: int, time_info, status) -> None:
        """Sounddevice input callback - processes microphone audio"""
        self.input_callback_count += 1

        current_time = time.time()
        if current_time - self.last_debug_time > 10.0:  # Reduced frequency for Raspberry Pi
            self.logger.info(f"Input callback stats: called {self.input_callback_count} times, "
                           f"processed {self.frames_processed} frames, "
                           f"sent {self.frames_sent_to_livekit} to LiveKit")
            self.last_debug_time = current_time

        if status:
            self.logger.warning(f"Input callback status: {status}")

        if not self.running:
            return

        if self.input_callback_count <= 3:
            self.logger.info(f"Input callback #{self.input_callback_count}: "
                           f"frame_count={frame_count}, indata.shape={indata.shape}")

        with self.mute_lock:
            is_muted = self.is_muted

        processed_indata = indata.copy()
        if is_muted:
            processed_indata.fill(0)

        self.input_delay = time_info.currentTime - time_info.inputBufferAdcTime
        total_delay = self.output_delay + self.input_delay

        if self.audio_processor:
            try:
                self.audio_processor.set_stream_delay_ms(int(total_delay * 1000))
            except RuntimeError as e:
                if not hasattr(self, '_delay_error_logged'):
                    self.logger.warning(f"Failed to set APM stream delay: {e}")
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
                num_channels=NUM_CHANNELS,
            )

            self.frames_processed += 1

            if self.audio_processor:
                try:
                    self.audio_processor.process_stream(capture_frame)
                except Exception as e:
                    if self.frames_processed <= 5:
                        self.logger.warning(f"Error processing audio stream with AEC: {e}")

            rms = np.sqrt(np.mean(original_chunk.astype(np.float32) ** 2))
            max_int16 = np.iinfo(np.int16).max
            self.micro_db = 20.0 * np.log10(rms / max_int16 + 1e-6)

            if self.loop and not self.loop.is_closed():
                try:
                    queue_size = self.audio_input_queue.qsize()
                    if queue_size > 40:
                        self.logger.warning(f"Audio input queue getting full: {queue_size} items")
                    self.loop.call_soon_threadsafe(self.audio_input_queue.put_nowait, capture_frame)
                    self.frames_sent_to_livekit += 1
                except Exception as e:
                    if self.frames_processed <= 5:
                        self.logger.warning(f"Failed to queue audio frame: {e}")

    def _output_callback(self, outdata: np.ndarray, frame_count: int, time_info, status) -> None:
        """Sounddevice output callback - plays received audio"""
        self.output_callback_count += 1

        if status:
            self.logger.warning(f"Output callback status: {status}")

        if not self.running:
            outdata.fill(0)
            return

        self.output_delay = time_info.outputBufferDacTime - time_info.currentTime

        with self.output_lock:
            bytes_needed = frame_count * 2
            if len(self.output_buffer) < bytes_needed:
                available_bytes = len(self.output_buffer)
                if available_bytes > 0:
                    outdata[:available_bytes // 2, 0] = np.frombuffer(
                        self.output_buffer[:available_bytes],
                        dtype=np.int16,
                        count=available_bytes // 2,
                    )
                    outdata[available_bytes // 2:, 0] = 0
                    del self.output_buffer[:available_bytes]
                else:
                    outdata.fill(0)
            else:
                chunk = self.output_buffer[:bytes_needed]
                outdata[:, 0] = np.frombuffer(chunk, dtype=np.int16, count=frame_count)
                del self.output_buffer[:bytes_needed]

        if self.audio_processor:
            num_chunks = frame_count // FRAME_SAMPLES
            for i in range(num_chunks):
                start = i * FRAME_SAMPLES
                end = start + FRAME_SAMPLES
                if end > frame_count:
                    break
                render_chunk = outdata[start:end, 0]
                render_frame = rtc.AudioFrame(
                    data=render_chunk.tobytes(),
                    samples_per_channel=FRAME_SAMPLES,
                    sample_rate=SAMPLE_RATE,
                    num_channels=NUM_CHANNELS,
                )
                try:
                    self.audio_processor.process_reverse_stream(render_frame)
                except Exception as e:
                    if self.output_callback_count <= 5:
                        self.logger.warning(f"Error processing reverse stream with AEC: {e}")

    def print_audio_meter(self):
        """Print dB meter with live/mute indicator"""
        if not self.meter_running:
            return
        self._print_simple_meter()

    def _print_simple_meter(self):
        """Simple terminal meter display"""
        if not self.meter_running:
            return

        meter_parts = []
        status_info = f"I:{self.input_callback_count} O:{self.output_callback_count} Q:{self.audio_input_queue.qsize()} P:{len(self.participants)} "

        amplitude_db = _normalize_db(self.micro_db, db_min=INPUT_DB_MIN, db_max=INPUT_DB_MAX)
        nb_bar = round(amplitude_db * MAX_AUDIO_BAR)
        color_code = 31 if amplitude_db > 0.75 else 33 if amplitude_db > 0.5 else 32
        bar = "#" * nb_bar + "-" * (MAX_AUDIO_BAR - nb_bar)

        with self.mute_lock:
            is_muted = self.is_muted
        live_indicator = f"{_esc(90)}●{_esc(0)} " if is_muted else f"{_esc(1, 38, 2, 255, 0, 0)}●{_esc(0)} "

        local_part = f"{live_indicator}Mic[{self.micro_db:6.1f}]{_esc(color_code)}[{bar}]{_esc(0)}"
        meter_parts.append(local_part)

        current_time = time.time()
        with self.participants_lock:
            for participant_id, info in list(self.participants.items()):
                if current_time - info['last_update'] > 5.0:
                    del self.participants[participant_id]
                    continue
                participant_amplitude_db = _normalize_db(info['db_level'], db_min=INPUT_DB_MIN, db_max=INPUT_DB_MAX)
                participant_nb_bar = round(participant_amplitude_db * (MAX_AUDIO_BAR // 2))
                participant_color_code = 31 if participant_amplitude_db > 0.75 else 33 if participant_amplitude_db > 0.5 else 32
                participant_bar = "#" * participant_nb_bar + "-" * ((MAX_AUDIO_BAR // 2) - participant_nb_bar)
                participant_indicator = f"{_esc(94)}●{_esc(0)} "
                participant_part = f"{participant_indicator}{info['name'][:6]}[{info['db_level']:6.1f}]{_esc(participant_color_code)}[{participant_bar}]{_esc(0)}"
                meter_parts.append(participant_part)

        meter_text = status_info + " ".join(meter_parts)
        with self.stdout_lock:
            sys.stdout.write(f"\033[2K\r\033[?25l{meter_text}")
            sys.stdout.flush()

    def init_terminal(self):
        """Initialize terminal for stable UI display"""
        with self.stdout_lock:
            sys.stdout.write("\033[?25l")
            sys.stdout.flush()

    def restore_terminal(self):
        """Restore terminal to normal state"""
        with self.stdout_lock:
            sys.stdout.write("\033[2K\r\033[?25h")
            sys.stdout.flush()

async def main(participant_name: str, enable_aec: bool = True):
    logger = logging.getLogger(__name__)
    logger.info("=== STARTING AUDIO STREAMER ===")
    loop = asyncio.get_running_loop()

    if not LIVEKIT_URL or not ROOM_NAME:
        logger.error("Missing LIVEKIT_URL or ROOM_NAME environment variables")
        return

    streamer = AudioStreamer(enable_aec, loop=loop)
    room = rtc.Room(loop=loop)
    streamer.room = room

    async def audio_processing_task():
        frames_sent = 0
        logger.info("Audio processing task started")
        while streamer.running:
            try:
                frame = await asyncio.wait_for(streamer.audio_input_queue.get(), timeout=1.0)
                await streamer.source.capture_frame(frame)
                frames_sent += 1
                if frames_sent <= 5 or frames_sent % 100 == 0:
                    logger.info(f"Sent frame {frames_sent} to LiveKit")
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in audio processing: {e}")
                break
        logger.info(f"Audio processing task ended. Total frames sent: {frames_sent}")

    async def meter_task():
        logger.info("Meter task started")
        while streamer.running and streamer.meter_running:
            streamer.print_audio_meter()
            await asyncio.sleep(1 / FPS)
        logger.info("Meter task ended")

    async def receive_audio_frames(stream: rtc.AudioStream, participant: rtc.RemoteParticipant):
        frames_received = 0
        participant_id = participant.sid
        participant_name = participant.identity or f"User_{participant.sid[:8]}"
        logger.info(f"Receiving audio from participant: {participant_name} ({participant_id})")
        async for frame_event in stream:
            if not streamer.running:
                break
            frames_received += 1
            if frames_received <= 5 or frames_received % 100 == 0:
                logger.info(f"Received frame {frames_received} from {participant_name}")
            if streamer.active_remote_participant_id == participant_id and streamer.remote_playback_enabled:
                frame_data = frame_event.frame.data
                if len(frame_data) > 0:
                    audio_samples = np.frombuffer(frame_data, dtype=np.int16)
                    if len(audio_samples) > 0:
                        rms = np.sqrt(np.mean(audio_samples.astype(np.float32) ** 2))
                        max_int16 = np.iinfo(np.int16).max
                        participant_db = 20.0 * np.log10(rms / max_int16 + 1e-6)
                        with streamer.participants_lock:
                            streamer.participants[participant_id] = {
                                'name': participant_name,
                                'db_level': participant_db,
                                'last_update': time.time()
                            }
                with streamer.output_lock:
                    streamer.output_buffer.extend(frame_event.frame.data.tobytes())
        logger.info(f"Audio receive task ended for {participant_name}. Total frames received: {frames_received}")
        with streamer.participants_lock:
            if participant_id in streamer.participants:
                del streamer.participants[participant_id]

    @room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant):
        logger.info(f"Track subscribed: {publication.sid} from participant {participant.sid}")
        if track.kind == rtc.TrackKind.KIND_AUDIO and streamer.active_remote_participant_id is None:
            streamer.active_remote_participant_id = participant.sid
            audio_stream = rtc.AudioStream(track, sample_rate=SAMPLE_RATE, num_channels=NUM_CHANNELS)
            asyncio.ensure_future(receive_audio_frames(audio_stream, participant))

    @room.on("track_published")
    def on_track_published(publication, participant):
        logger.info(f"Track published: {publication.sid} from participant {participant.sid}")

    @room.on("participant_connected")
    def on_participant_connected(participant):
        logger.info(f"Participant connected: {participant.sid}")
        with streamer.participants_lock:
            streamer.participants[participant.sid] = {
                'name': participant.identity or f"User_{participant.sid[:8]}",
                'db_level': INPUT_DB_MIN,
                'last_update': time.time()
            }

    @room.on("participant_disconnected")
    def on_participant_disconnected(participant):
        logger.info(f"Participant disconnected: {participant.sid}")
        with streamer.participants_lock:
            if participant.sid in streamer.participants:
                del streamer.participants[participant.sid]
        if streamer.active_remote_participant_id == participant.sid:
            streamer.active_remote_participant_id = None
            with streamer.output_lock:
                streamer.output_buffer.clear()

    @room.on("connected")
    def on_connected():
        logger.info("Successfully connected to LiveKit room")

    @room.on("disconnected")
    def on_disconnected(reason):
        logger.info(f"Disconnected from LiveKit room: {reason}")

    try:
        streamer.start_audio_devices()
        streamer.start_keyboard_handler()
        streamer.init_terminal()
        token = generate_token(ROOM_NAME, participant_name, participant_name)
        await room.connect(LIVEKIT_URL, token)
        logger.info(f"Connected to room: {room.name}")

        track = rtc.LocalAudioTrack.create_audio_track("mic", streamer.source)
        options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        publication = await room.local_participant.publish_track(track, options)
        logger.info(f"Published track: {publication.sid}")

        audio_task = asyncio.create_task(audio_processing_task())
        meter_display_task = asyncio.create_task(meter_task())

        while streamer.running:
            await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        streamer.running = False
        if 'audio_task' in locals():
            audio_task.cancel()
            try:
                await audio_task
            except asyncio.CancelledError:
                pass
        if 'meter_display_task' in locals():
            meter_display_task.cancel()
            try:
                await meter_display_task
            except asyncio.CancelledError:
                pass
        streamer.stop_audio_devices()
        streamer.stop_keyboard_handler()
        await room.disconnect()
        streamer.restore_terminal()
        logger.info("=== CLEANUP COMPLETE ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LiveKit audio streaming for Raspberry Pi")
    parser.add_argument("--name", "-n", type=str, default="rpi-streamer", help="Participant name")
    parser.add_argument("--disable-aec", action="store_true", help="Disable acoustic echo cancellation")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("stream_audio.log"),
            logging.StreamHandler() if args.debug else []
        ]
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    main_task = loop.create_task(main(args.name, enable_aec=not args.disable_aec))

    def shutdown():
        main_task.cancel()

    def signal_handler(sig, frame):
        loop.call_soon_threadsafe(shutdown)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        loop.run_until_complete(main_task)
    except asyncio.CancelledError:
        pass
    finally:
        loop.close()