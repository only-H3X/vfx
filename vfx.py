import numpy as np
import sounddevice as sd
from scipy.signal import get_window

##############################
# CONFIGURATION
##############################
SAMPLE_RATE = 44100     # Audio sample rate (Hz)
BLOCK_SIZE = 1024       # Frames per chunk/block
PITCH_SHIFT = 1.2       # Factor > 1.0 = shift up,  < 1.0 = shift down
WINDOW_TYPE = 'hann'    # Window function type
OVERLAP_FACTOR = 4      # STFT overlap (4 => 75% overlap if BLOCK_SIZE is the FFT size)

##############################
# GLOBAL BUFFERS
##############################
# We’ll maintain some buffers for the phase-vocoder approach.
# Each block we’ll do an STFT, shift frequencies, then iSTFT.

# Window for STFT
window = get_window(WINDOW_TYPE, BLOCK_SIZE, fftbins=True).astype(np.float32)

# Overlap buffers
# We keep extra overlap frames from the previous block to allow smooth iSTFT.
hop_size = BLOCK_SIZE // OVERLAP_FACTOR
prev_input = np.zeros(BLOCK_SIZE, dtype=np.float32)
# For reconstruction overlap
output_buffer = np.zeros(BLOCK_SIZE * 2, dtype=np.float32)

# Phase accumulator for each frequency bin in the STFT
phase_accumulator = np.zeros(BLOCK_SIZE, dtype=np.float32)


def phase_vocoder_shift_frequency(stft_frame, phase_accum, shift):
    """
    Phase vocoder pitch shift on a single STFT frame.
    stft_frame: complex spectrum of size BLOCK_SIZE
    phase_accum: array to hold accumulated phases between frames
    shift: pitch shift factor
    Returns a shifted frame (complex spectrum) and updated phase accumulator.
    """
    # Magnitude and phase
    magnitudes = np.abs(stft_frame)
    phases = np.angle(stft_frame)

    # Compute instantaneous frequency
    # freq(omega) = dPhi/dt, approximate difference from last angle
    delta = phases - phase_accum

    # Wrap around -pi..pi
    delta = np.mod(delta + np.pi, 2.0 * np.pi) - np.pi

    # True frequency
    # Because we hop by hop_size, scale the delta by hop_size to get the real step
    # This references the standard phase vocoder eqn:
    #    freq(omega_k) = (omega_k + delta_k / hop_size)
    # but to keep it simpler, we just add the scaled delta.
    time_scale = hop_size
    instantaneous_freq = (phase_accum + delta) / time_scale

    # Now multiply by the pitch shift factor
    # And re-accumulate phases for the next block
    phase_accum_new = phase_accum + shift * (instantaneous_freq * hop_size)

    # Create the new complex spectrum with the old magnitudes but shifted phases
    shifted_frame = magnitudes * np.exp(1.0j * phase_accum_new)

    return shifted_frame, phase_accum_new


def process_audio_individual_block(input_block, pitch_shift_factor):
    """
    Process a single block of audio samples through an STFT-based pitch shifter.
    input_block: array of shape (BLOCK_SIZE,) float32
    pitch_shift_factor: float
    Returns a block of the same size (BLOCK_SIZE,) float32
    """
    global prev_input, output_buffer, phase_accumulator

    # Overlap input with previous tail
    current_input = np.concatenate((prev_input[-(BLOCK_SIZE - hop_size):], input_block))

    # Windowed input
    windowed_input = current_input * window

    # Forward STFT (complex spectrum)
    spectrum = np.fft.rfft(windowed_input, n=BLOCK_SIZE)

    # Pitch shift via phase vocoder
    shifted_spectrum, new_phase_accum = phase_vocoder_shift_frequency(
        spectrum, phase_accumulator[:len(spectrum)], pitch_shift_factor
    )
    # Update phase accumulator for freq bins used
    phase_accumulator[:len(shifted_spectrum)] = new_phase_accum

    # Inverse STFT
    time_domain = np.fft.irfft(shifted_spectrum, n=BLOCK_SIZE)

    # Multiply by window again
    time_domain_windowed = time_domain * window

    # Overlap-add into output_buffer
    # The first `hop_size` samples overlap with the tail from the last block
    out_block = output_buffer[:BLOCK_SIZE] + time_domain_windowed
    output_buffer[:BLOCK_SIZE] = 0.0  # Reset for next time
    output_buffer[:BLOCK_SIZE] = out_block

    # Shift the buffer left by hop_size
    processed = output_buffer[:hop_size].copy()
    output_buffer = np.roll(output_buffer, -hop_size)
    output_buffer[-hop_size:] = 0.0

    # Save the last full input block for next time’s overlap
    prev_input = current_input.copy()

    # Return the newly processed chunk (hop_size in length),
    # but we want BLOCK_SIZE frames to fill the same shape:
    # We'll pad if needed. This is one approach – you can also design
    # your system to output a smaller chunk per iteration (hop size).
    out_padded = np.zeros(BLOCK_SIZE, dtype=np.float32)
    out_padded[:min(hop_size, BLOCK_SIZE)] = processed[:min(hop_size, BLOCK_SIZE)]
    return out_padded


def audio_callback(indata, outdata, frames, time_info, status):
    """sounddevice callback function for real-time processing"""
    if status:
        print(status, flush=True)

    # Convert to float32 if not already
    input_block = indata[:, 0].astype(np.float32)  # Take mono (channel 0)
    processed_block = process_audio_individual_block(input_block, PITCH_SHIFT)

    # If you want stereo output, duplicate processed_block or add other channels
    outdata[:, 0] = processed_block
    # If you want the second channel silent or the same, do:
    if outdata.shape[1] > 1:
        outdata[:, 1] = processed_block


def main():
    print("Starting Voice Modulator...")
    print("Press Ctrl+C to stop.")

    with sd.Stream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        dtype='float32',
        channels=1,       # 1 input channel (mono mic)
        channels_in=1,    # mono input
        channels_out=2,   # 2 output channels for stereo
        callback=audio_callback
    ):
        # Keep the stream open and running until user stops
        while True:
            pass


if __name__ == "__main__":
    main()
