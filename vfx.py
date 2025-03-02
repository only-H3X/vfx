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

# Default configuration with additional audio quality and noise reduction parameters.
DEFAULT_CONFIG = {
    "pitch": 0.0,
    "samplerate": 44100,
    "channels": 1,
    "blocksize": 1024,
    "input_device": None,
    "output_device": None,
    "enable_noise_reduction": True,
    "calibration_duration": 3.0,  # seconds for ambient noise calibration
    "enable_compression": False,
    "compression_gain": 2.0,      # gain for soft clipping compression
    "dither_level": 1e-6,         # amplitude of dithering noise
    # New noise reduction tuning parameters:
    "nr_prop_decrease": 1.0,      # How much to reduce the noise (1.0 means full reduction)
    "nr_n_std_thresh": 1.5        # Noise threshold in terms of standard deviations
}

# Preset configurations for 15 human vocals (basic settings).
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
    """Loads default configuration and applies JSON/CLI overrides."""
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
    """Production‑grade voice modulator with enhanced real‑time processing."""
    def __init__(self, config: Config):
        self.config = config
        # Basic settings.
        self.pitch = config.get("pitch")
        self.samplerate = config.get("samplerate")
        self.channels = config.get("channels")
        self.blocksize = config.get("blocksize")
        self.input_device = config.get("input_device")
        self.output_device = config.get("output_device")
        self.enable_noise_reduction = config.get("enable_noise_reduction")
        self.calibration_duration = config.get("calibration_duration")
        # New audio quality settings.
        self.enable_compression = config.get("enable_compression")
        self.compression_gain = config.get("compression_gain")
        self.dither_level = config.get("dither_level")
        # New noise reduction tuning parameters.
        self.nr_prop_decrease = config.get("nr_prop_decrease")
        self.nr_n_std_thresh = config.get("nr_n_std_thresh")
        self.noise_profile = None
        self.running = False
        self.audio_stream = None
        self.param_lock = threading.Lock()
        # For performance metrics.
        self.processing_times = []
        self.last_processing_time = 0.0

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

    def apply_compression_and_dither(self, audio):
        """Apply a simple soft-clip compression and add dithering noise."""
        if self.enable_compression:
            # Soft-clip using tanh.
            audio = np.tanh(self.compression_gain * audio) / np.tanh(self.compression_gain)
        if self.dither_level > 0:
            audio = audio + np.random.normal(0, self.dither_level, audio.shape)
        return audio

    def audio_callback(self, indata, outdata, frames, time_info, status):
        start_time = time.perf_counter()
        if status:
            logging.warning("Audio stream status: %s", status)
        try:
            processed = np.empty_like(indata)
            with self.param_lock:
                current_pitch = self.pitch
                nr_enabled = self.enable_noise_reduction
                noise_profile = self.noise_profile
            # Process each channel independently.
            for ch in range(indata.shape[1]):
                channel_audio = indata[:, ch]
                # Apply noise reduction if enabled and a noise profile exists.
                if nr_enabled and noise_profile is not None:
                    try:
                        reduced_audio = nr.reduce_noise(
                            y=channel_audio,
                            sr=self.samplerate,
                            y_noise=noise_profile,
                            prop_decrease=self.nr_prop_decrease,
                            n_std_thresh=self.nr_n_std_thresh
                        )
                    except Exception as nr_err:
                        logging.error("Noise reduction error: %s", nr_err)
                        reduced_audio = channel_audio
                else:
                    reduced_audio = channel_audio
                # Pitch shifting.
                shifted = librosa.effects.pitch_shift(reduced_audio, self.samplerate, n_steps=current_pitch)
                if len(shifted) < len(channel_audio):
                    shifted = np.pad(shifted, (0, len(channel_audio) - len(shifted)), mode="constant")
                elif len(shifted) > len(channel_audio):
                    shifted = shifted[:len(channel_audio)]
                # Apply compression and dithering.
                processed[:, ch] = self.apply_compression_and_dither(shifted)
            outdata[:] = processed
        except Exception as e:
            logging.error("Error in audio callback: %s", e)
            outdata[:] = indata  # fallback: pass input through
        end_time = time.perf_counter()
        processing_time = end_time - start_time
        with self.param_lock:
            self.processing_times.append(processing_time)
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
            self.last_processing_time = sum(self.processing_times) / len(self.processing_times)
        if processing_time > 0.05:  # 50ms threshold
            logging.warning("High processing time: %.3f s", processing_time)

    def list_devices(self):
        """List available input/output devices."""
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            logging.info("Device %d: %s", i, dev["name"])
        return devices

    def start(self):
        """Start the audio stream and perform initial noise calibration if enabled."""
        self.running = True
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
            logging.info("Voice modulator started with samplerate=%d, channels=%d, blocksize=%d, pitch=%f",
                         self.samplerate, self.channels, self.blocksize, self.pitch)
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

def get_input_devices():
    devices = sd.query_devices()
    input_devices = {"Default": "Default"}
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            input_devices[str(i)] = dev["name"]
    return input_devices

def get_output_devices():
    devices = sd.query_devices()
    output_devices = {"Default": "Default"}
    for i, dev in enumerate(devices):
        if dev["max_output_channels"] > 0:
            output_devices[str(i)] = dev["name"]
    return output_devices

# GUI window for dynamic configuration.
class ConfigWindow(tk.Tk):
    def __init__(self, modulator: VoiceModulator, presets: dict):
        super().__init__()
        self.title("Voice Modulator Configuration")
        self.modulator = modulator
        self.presets = presets
        self.geometry("600x600")
        self.modulator_thread = None
        self.create_widgets()
        self.load_basic_settings()
        self.load_advanced_settings()
        self.create_performance_panel()
        self.create_platform_info()
        self.update_performance_metrics()
        # Start the modulator stream.
        self.restart_stream()

    def create_widgets(self):
        # Basic Configuration.
        basic_frame = ttk.LabelFrame(self, text="Basic Configuration")
        basic_frame.pack(padx=10, pady=10, fill="x")
        ttk.Label(basic_frame, text="Select Preset:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.preset_var = tk.StringVar(value=list(self.presets.keys())[0])
        self.preset_menu = ttk.OptionMenu(basic_frame, self.preset_var,
                                          list(self.presets.keys())[0],
                                          *self.presets.keys(),
                                          command=self.on_preset_selected)
        self.preset_menu.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Label(basic_frame, text="Pitch (semitones):").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.pitch_var = tk.DoubleVar()
        self.pitch_scale = ttk.Scale(basic_frame, from_=-10, to=10, orient="horizontal",
                                     variable=self.pitch_var)
        self.pitch_scale.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.nr_var = tk.BooleanVar()
        self.nr_check = ttk.Checkbutton(basic_frame, text="Enable Noise Reduction", variable=self.nr_var)
        self.nr_check.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        basic_apply_btn = ttk.Button(basic_frame, text="Apply Basic Settings", command=self.apply_basic_settings)
        basic_apply_btn.grid(row=3, column=0, columnspan=2, padx=5, pady=5)
        basic_frame.columnconfigure(1, weight=1)

        # Advanced Configuration.
        adv_frame = ttk.LabelFrame(self, text="Advanced Configuration")
        adv_frame.pack(padx=10, pady=10, fill="x")
        ttk.Label(adv_frame, text="Samplerate (Hz):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.samplerate_var = tk.IntVar(value=self.modulator.samplerate)
        self.samplerate_entry = ttk.Entry(adv_frame, textvariable=self.samplerate_var)
        self.samplerate_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Label(adv_frame, text="Channels:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.channels_var = tk.IntVar(value=self.modulator.channels)
        self.channels_entry = ttk.Entry(adv_frame, textvariable=self.channels_var)
        self.channels_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ttk.Label(adv_frame, text="Blocksize:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.blocksize_var = tk.IntVar(value=self.modulator.blocksize)
        self.blocksize_entry = ttk.Entry(adv_frame, textvariable=self.blocksize_var)
        self.blocksize_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        ttk.Label(adv_frame, text="Calibration Duration (sec):").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.calibration_duration_var = tk.DoubleVar(value=self.modulator.calibration_duration)
        self.calibration_duration_entry = ttk.Entry(adv_frame, textvariable=self.calibration_duration_var)
        self.calibration_duration_entry.grid(row=3, column=1, padx=5, pady=5, sticky="ew")
        input_devs = get_input_devices()
        output_devs = get_output_devices()
        ttk.Label(adv_frame, text="Input Device:").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.input_device_var = tk.StringVar(value="Default")
        self.input_menu = ttk.OptionMenu(adv_frame, self.input_device_var, "Default", *list(input_devs.keys()))
        self.input_menu.grid(row=4, column=1, padx=5, pady=5, sticky="ew")
        ttk.Label(adv_frame, text="Output Device:").grid(row=5, column=0, padx=5, pady=5, sticky="w")
        self.output_device_var = tk.StringVar(value="Default")
        self.output_menu = ttk.OptionMenu(adv_frame, self.output_device_var, "Default", *list(output_devs.keys()))
        self.output_menu.grid(row=5, column=1, padx=5, pady=5, sticky="ew")
        # New advanced quality settings.
        ttk.Label(adv_frame, text="Compression Gain:").grid(row=6, column=0, padx=5, pady=5, sticky="w")
        self.compression_gain_var = tk.DoubleVar(value=self.modulator.compression_gain)
        self.compression_gain_entry = ttk.Entry(adv_frame, textvariable=self.compression_gain_var)
        self.compression_gain_entry.grid(row=6, column=1, padx=5, pady=5, sticky="ew")
        ttk.Label(adv_frame, text="Dither Level:").grid(row=7, column=0, padx=5, pady=5, sticky="w")
        self.dither_level_var = tk.DoubleVar(value=self.modulator.dither_level)
        self.dither_level_entry = ttk.Entry(adv_frame, textvariable=self.dither_level_var)
        self.dither_level_entry.grid(row=7, column=1, padx=5, pady=5, sticky="ew")
        ttk.Label(adv_frame, text="Enable Compression:").grid(row=8, column=0, padx=5, pady=5, sticky="w")
        self.compression_enabled_var = tk.BooleanVar(value=self.modulator.enable_compression)
        self.compression_enabled_check = ttk.Checkbutton(adv_frame, text="Compression", variable=self.compression_enabled_var)
        self.compression_enabled_check.grid(row=8, column=1, padx=5, pady=5, sticky="w")
        # New noise reduction parameters.
        ttk.Label(adv_frame, text="NR Prop Decrease:").grid(row=9, column=0, padx=5, pady=5, sticky="w")
        self.nr_prop_decrease_var = tk.DoubleVar(value=self.modulator.nr_prop_decrease)
        self.nr_prop_decrease_entry = ttk.Entry(adv_frame, textvariable=self.nr_prop_decrease_var)
        self.nr_prop_decrease_entry.grid(row=9, column=1, padx=5, pady=5, sticky="ew")
        ttk.Label(adv_frame, text="NR N Std Thresh:").grid(row=10, column=0, padx=5, pady=5, sticky="w")
        self.nr_n_std_thresh_var = tk.DoubleVar(value=self.modulator.nr_n_std_thresh)
        self.nr_n_std_thresh_entry = ttk.Entry(adv_frame, textvariable=self.nr_n_std_thresh_var)
        self.nr_n_std_thresh_entry.grid(row=10, column=1, padx=5, pady=5, sticky="ew")
        adv_btn_frame = ttk.Frame(adv_frame)
        adv_btn_frame.grid(row=11, column=0, columnspan=2, pady=10)
        restart_btn = ttk.Button(adv_btn_frame, text="Restart Stream", command=self.restart_stream)
        restart_btn.pack(side="left", padx=5)
        stop_btn = ttk.Button(adv_btn_frame, text="Stop Stream", command=self.stop_stream)
        stop_btn.pack(side="left", padx=5)
        adv_frame.columnconfigure(1, weight=1)
        basic_frame.columnconfigure(1, weight=1)

    def create_performance_panel(self):
        perf_frame = ttk.LabelFrame(self, text="Performance Metrics")
        perf_frame.pack(padx=10, pady=10, fill="x")
        self.performance_label = ttk.Label(perf_frame, text="Avg Processing Time: N/A")
        self.performance_label.pack(padx=5, pady=5)

    def create_platform_info(self):
        platform_info = f"Platform: {platform.system()} {platform.release()}"
        self.platform_label = ttk.Label(self, text=platform_info)
        self.platform_label.pack(padx=10, pady=5)

    def load_basic_settings(self):
        with self.modulator.param_lock:
            self.pitch_var.set(self.modulator.pitch)
            self.nr_var.set(self.modulator.enable_noise_reduction)

    def load_advanced_settings(self):
        self.samplerate_var.set(self.modulator.samplerate)
        self.channels_var.set(self.modulator.channels)
        self.blocksize_var.set(self.modulator.blocksize)
        self.calibration_duration_var.set(self.modulator.calibration_duration)
        self.input_device_var.set("Default" if self.modulator.input_device is None else str(self.modulator.input_device))
        self.output_device_var.set("Default" if self.modulator.output_device is None else str(self.modulator.output_device))
        self.compression_gain_var.set(self.modulator.compression_gain)
        self.dither_level_var.set(self.modulator.dither_level)
        self.compression_enabled_var.set(self.modulator.enable_compression)
        self.nr_prop_decrease_var.set(self.modulator.nr_prop_decrease)
        self.nr_n_std_thresh_var.set(self.modulator.nr_n_std_thresh)

    def on_preset_selected(self, preset_name):
        preset = self.presets.get(preset_name)
        if preset:
            self.pitch_var.set(preset.get("pitch", 0.0))
            self.nr_var.set(preset.get("enable_noise_reduction", True))
            # Immediately update the modulator with the preset values.
            self.apply_basic_settings()

    def apply_basic_settings(self):
        with self.modulator.param_lock:
            self.modulator.pitch = self.pitch_var.get()
            self.modulator.enable_noise_reduction = self.nr_var.get()
        logging.info("Applied basic settings: pitch=%f, noise reduction=%s",
                     self.modulator.pitch, self.modulator.enable_noise_reduction)

    def restart_stream(self):
        logging.info("Restarting audio stream with new advanced settings...")
        self.stop_stream()
        with self.modulator.param_lock:
            self.modulator.samplerate = int(self.samplerate_var.get())
            self.modulator.channels = int(self.channels_var.get())
            self.modulator.blocksize = int(self.blocksize_var.get())
            self.modulator.calibration_duration = float(self.calibration_duration_var.get())
            inp = self.input_device_var.get()
            self.modulator.input_device = None if inp == "Default" else int(inp)
            outp = self.output_device_var.get()
            self.modulator.output_device = None if outp == "Default" else int(outp)
            self.modulator.compression_gain = float(self.compression_gain_var.get())
            self.modulator.dither_level = float(self.dither_level_var.get())
            self.modulator.enable_compression = self.compression_enabled_var.get()
            self.modulator.nr_prop_decrease = float(self.nr_prop_decrease_var.get())
            self.modulator.nr_n_std_thresh = float(self.nr_n_std_thresh_var.get())
        self.modulator.running = True
        self.modulator_thread = threading.Thread(target=self.modulator.start, daemon=True)
        self.modulator_thread.start()

    def stop_stream(self):
        if self.modulator.running:
            self.modulator.stop()
        if self.modulator_thread and self.modulator_thread.is_alive():
            self.modulator_thread.join()
            logging.info("Audio stream stopped.")

    def update_performance_metrics(self):
        with self.modulator.param_lock:
            avg_time_ms = self.modulator.last_processing_time * 1000
        self.performance_label.config(text=f"Avg Processing Time: {avg_time_ms:.2f} ms")
        self.after(1000, self.update_performance_metrics)

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
    if platform.system() == "Darwin":
        logging.info("Running on macOS. Ensure microphone access is granted in System Preferences > Security & Privacy > Microphone.")
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
    app = ConfigWindow(modulator, DEFAULT_PRESETS)
    app.protocol("WM_DELETE_WINDOW", app.quit)
    app.mainloop()
    modulator.stop()
    if app.modulator_thread and app.modulator_thread.is_alive():
        app.modulator_thread.join()

if __name__ == "__main__":
    main()
