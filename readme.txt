pip install numpy sounddevice
sound device needs PortAudio (only if not windows)
macOS brew install PortAudio


Run demo with
python dtmf_live.py

Useful flags:
python dtmf_live.py --list        # list input devices
python dtmf_live.py --device 1    # pick arctis 5 chat
python dtmf_live.py --rate 16000  # use a different sample rate