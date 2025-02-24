#!/usr/bin/env python3
import argparse
import json
import logging
import threading
import time
import platform
import numpy as np
import sounddevice as sd
import librosa
import noisereduce as nr
import tkinter as tk
from tkinter import ttk

# Default configuration with noise reduction settings.
DEFAULT_CONFIG = {
    "pitch": 0.0,
    "samplerate": 44100,
    "channels": 1,
    "blocksize": 1024,
    "input_device": None,
    "output_device": None,
    "enable_noise_reduction": True,
    "calibration_duration": 3.0  # seconds for ambient noise calibration
}

# Preset configurations for 15 human vocals.
DEFAULT_PRESETS = {
    "Male Bass":      {"pitch": -4.0, "enable_noise_reduction": True},
    "Male Baritone":  {"pitch": -2.0, "enable_noise_reduction": True},
    "Male Tenor":     {"pitch": 0.0,  "enable_noise_reduction": True},
    "Male High":      {"pitch": 2.0,  "enable_noise_reduction": True},
    "Female Alto":    {"pitch": -2.0, "enable_noise_reduction": True},
    "Female Mezzo":   {"pitch": 0.0,  "enable_noise_reduction": True},
    "Female Soprano": {"pitch": 2.0,  "enable_noise_reduction": True},
    "Child Male":     {"pitch": 4.0,  "enable_noise_reduction": True},
    "Child Female":   {"pitch": 4.0,  "enable_noise_reduction": True},
    "Elderly Male":   {"pitch": -3.0, "enable_noise_reduction": True},
    "Elderly Female": {"pitch": -1.0, "enable_noise_reduction": True},
    "Narrator":       {"pitch": -1.0, "enable_noise_reduction": True},
    "News Anchor":    {"pitch": 0.0,  "enable_noise_reduction": True},
    "Podcast Host":   {"pitch": 0.0,  "enable_noise_reduction": True},
    "Opera Singer":   {"pitch": 3.0,  "enable_noise_reduction": True}
}

class Config:
    """Loads default configuration and applies JSON or CLI overrides."""
    def __init__(self, config_file=None):
        self.config = DEFAULT_CONFIG.copy()
        if config_file:
            try:
                with open(config_file, "r") as f:
                    file_config = json.load(f)
                self.config.update(file_config)
                logging.info("Loaded configuration from %s", config_file)
            except Exception as e:
                logging.error("Error reading config file: %s", e)

    def get(self, key):
        return self.config.get(key)

    def update(self, key, value):
        self.config[key] = value

class VoiceModulator:
    """Production‐grade voice modulator with enhanced audio capture (noise cancellation and calibration)."""
    def __init__(self, config: Config):
        self.config = config
        self.pitch = config.get("pitch")
        self.samplerate = config.get("samplerate")
        self.channels = config.get("channels")
        self.blocksize = config.get("blocksize")
        self.input_device = config.get("input_device")
        self.output_device = config.get("output_device")
        self.enable_noise_reduction = config.get("enable_noise_reduction")
        self.calibration_duration = config.get("calibration_duration")
        self.noise_profile = None
        self.running = False
        self.audio_stream = None
        self.param_lock = threading.Lock()

    def calibrate_noise_profile(self):
        """Record ambient noise to build a noise profile."""
        logging.info("Calibrating noise for %.2f seconds. Please remain silent...", self.calibration_duration)
        try:
            recording = sd.rec(int(self.calibration_duration * self.samplerate),
                               samplerate=self.samplerate,
                               channels=self.channels,
                               dtype="float32")
            sd.wait()
            # Use first channel as noise profile.
            self.noise_profile = recording[:, 0]
            logging.info("Noise calibration completed.")
        except Exception as e:
            logging.error("Noise calibration failed: %s", e)
            self.noise_profile = None

    def audio_callback(self, indata, outdata, frames, time_info, status):
        if status:
            logging.warning("Audio stream status: %s", status)
        try:
            processed = np.empty_like(indata)
            with self.param_lock:
                current_pitch = self.pitch
                nr_enabled = self.enable_noise_reduction
                noise_profile = self.noise_profile

            for ch in range(indata.shape[1]):
                channel_audio = indata[:, ch]
                # Apply noise reduction if enabled and calibrated.
                if nr_enabled and noise_profile is not None:
                    try:
                        reduced_audio = nr.reduce_noise(y=channel_audio,
                                                        sr=self.samplerate,
                                                        y_noise=noise_profile)
                    except Exception as nr_err:
                        logging.error("Noise reduction error: %s", nr_err)
                        reduced_audio = channel_audio
                else:
                    reduced_audio = channel_audio

                # Apply pitch shifting.
                shifted = librosa.effects.pitch_shift(reduced_audio, self.samplerate, n_steps=current_pitch)
                if len(shifted) < len(channel_audio):
                    shifted = np.pad(shifted, (0, len(channel_audio) - len(shifted)), mode="constant")
                elif len(shifted) > len(channel_audio):
                    shifted = shifted[:len(channel_audio)]
                processed[:, ch] = shifted
            outdata[:] = processed
        except Exception as e:
            logging.error("Error in audio callback: %s", e)
            outdata[:] = indata  # fallback pass-through

    def list_devices(self):
        """List available input/output devices."""
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            logging.info("Device %d: %s", i, dev["name"])
        return devices

    def select_devices(self):
        """Select audio devices if not preset."""
        devices = self.list_devices()
        if self.input_device is None:
            try:
                inp = input("Enter input device index (or press Enter for default): ")
                if inp.strip():
                    self.input_device = int(inp)
            except ValueError:
                logging.info("Invalid input. Using default input device.")
        if self.output_device is None:
            try:
                outp = input("Enter output device index (or press Enter for default): ")
                if outp.strip():
                    self.output_device = int(outp)
            except ValueError:
                logging.info("Invalid input. Using default output device.")

    def start(self):
        """Start the audio stream and perform initial noise calibration if enabled."""
        self.running = True
        self.select_devices()
        if self.enable_noise_reduction:
            self.calibrate_noise_profile()
        try:
            self.audio_stream = sd.Stream(
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                dtype="float32",
                channels=self.channels,
                callback=self.audio_callback,
                device=(self.input_device, self.output_device)
            )
            self.audio_stream.start()
            logging.info("Voice modulator started.")
            while self.running:
                time.sleep(0.5)
        except Exception as e:
            logging.error("Error starting audio stream: %s", e)
        finally:
            self.stop()

    def stop(self):
        """Stop the audio stream."""
        self.running = False
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
        logging.info("Voice modulator stopped.")

# Custom logging filter to limit console output.
class LessThanFilter(logging.Filter):
    def __init__(self, max_level):
        super().__init__()
        self.max_level = max_level
    def filter(self, record):
        return record.levelno < self.max_level

def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.addFilter(LessThanFilter(logging.ERROR))
    console_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console_handler.setFormatter(console_formatter)
    file_handler = logging.FileHandler("voice_modulator.log", mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.handlers = []
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# GUI window for dynamic configuration.
class ConfigWindow(tk.Tk):
    def __init__(self, modulator: VoiceModulator, presets: dict):
        super().__init__()
        self.title("Voice Modulator Configuration")
        self.modulator = modulator
        self.presets = presets
        self.geometry("400x300")
        self.create_widgets()
        self.load_current_settings()

    def create_widgets(self):
        # Preset selection.
        preset_frame = ttk.LabelFrame(self, text="Presets")
        preset_frame.pack(padx=10, pady=10, fill="x")
        ttk.Label(preset_frame, text="Select Preset:").pack(side="left", padx=5, pady=5)
        self.preset_var = tk.StringVar()
        self.preset_menu = ttk.OptionMenu(preset_frame, self.preset_var,
                                          list(self.presets.keys())[0],
                                          *self.presets.keys(),
                                          command=self.on_preset_selected)
        self.preset_menu.pack(side="left", padx=5, pady=5)

        # Pitch control.
        control_frame = ttk.LabelFrame(self, text="Parameters")
        control_frame.pack(padx=10, pady=10, fill="x")
        ttk.Label(control_frame, text="Pitch (semitones):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.pitch_var = tk.DoubleVar()
        self.pitch_scale = ttk.Scale(control_frame, from_=-10, to=10, orient="horizontal",
                                     variable=self.pitch_var)
        self.pitch_scale.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        control_frame.columnconfigure(1, weight=1)

        # Noise reduction toggle.
        self.nr_var = tk.BooleanVar()
        self.nr_check = ttk.Checkbutton(control_frame, text="Enable Noise Reduction", variable=self.nr_var)
        self.nr_check.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="w")

        # Buttons for applying settings and calibrating noise.
        button_frame = ttk.Frame(self)
        button_frame.pack(padx=10, pady=10, fill="x")
        apply_btn = ttk.Button(button_frame, text="Apply Settings", command=self.apply_settings)
        apply_btn.pack(side="left", padx=5)
        calibrate_btn = ttk.Button(button_frame, text="Recalibrate Noise", command=self.recalibrate_noise)
        calibrate_btn.pack(side="left", padx=5)

    def load_current_settings(self):
        """Load current modulator settings into the GUI controls."""
        with self.modulator.param_lock:
            self.pitch_var.set(self.modulator.pitch)
            self.nr_var.set(self.modulator.enable_noise_reduction)

    def on_preset_selected(self, preset_name):
        """When a preset is chosen, update the GUI controls with the preset values."""
        preset = self.presets.get(preset_name)
        if preset:
            self.pitch_var.set(preset.get("pitch", 0.0))
            self.nr_var.set(preset.get("enable_noise_reduction", True))

    def apply_settings(self):
        """Apply the settings from the GUI to the modulator."""
        with self.modulator.param_lock:
            self.modulator.pitch = self.pitch_var.get()
            self.modulator.enable_noise_reduction = self.nr_var.get()
        logging.info("Applied settings: pitch=%f, noise reduction=%s",
                     self.modulator.pitch, self.modulator.enable_noise_reduction)

    def recalibrate_noise(self):
        """Trigger noise calibration in the modulator."""
        threading.Thread(target=self.modulator.calibrate_noise_profile, daemon=True).start()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Production Grade Voice Modulator with Interactive GUI")
    parser.add_argument("--config", type=str, help="Path to configuration JSON file", default=None)
    parser.add_argument("--pitch", type=float, help="Initial pitch shift (semitones)")
    parser.add_argument("--samplerate", type=int, help="Audio sample rate (Hz)")
    parser.add_argument("--channels", type=int, help="Number of audio channels")
    parser.add_argument("--blocksize", type=int, help="Audio processing block size")
    parser.add_argument("--input_device", type=int, help="Input device index")
    parser.add_argument("--output_device", type=int, help="Output device index")
    parser.add_argument("--nr", type=str, choices=["on", "off"], help="Initial noise reduction state", default=None)
    return parser.parse_args()

def main():
    setup_logging()
    
    # macOS-specific reminder for proper microphone access.
    if platform.system() == "Darwin":
        logging.info("Detected macOS environment. Ensure your application has microphone access via System Preferences > Security & Privacy > Microphone.")

    args = parse_arguments()
    config = Config(args.config)
    if args.pitch is not None:
        config.update("pitch", args.pitch)
    if args.samplerate is not None:
        config.update("samplerate", args.samplerate)
    if args.channels is not None:
        config.update("channels", args.channels)
    if args.blocksize is not None:
        config.update("blocksize", args.blocksize)
    if args.input_device is not None:
        config.update("input_device", args.input_device)
    if args.output_device is not None:
        config.update("output_device", args.output_device)
    if args.nr is not None:
        config.update("enable_noise_reduction", args.nr == "on")

    modulator = VoiceModulator(config)
    # Start the modulator in a background thread.
    modulator_thread = threading.Thread(target=modulator.start, daemon=True)
    modulator_thread.start()

    # Launch the interactive GUI window.
    app = ConfigWindow(modulator, DEFAULT_PRESETS)
    app.protocol("WM_DELETE_WINDOW", app.quit)
    app.mainloop()

    # When the GUI window is closed, signal the modulator to stop.
    modulator.stop()
    modulator_thread.join()

if __name__ == "__main__":
    main()
