import wave, json, numpy as np
from piper import PiperVoice, SynthesisConfig
VF = PiperVoice.load("./voices/en-us-lessac-medium.onnx")
VM = PiperVoice.load("./voices/en-us-ryan-high.onnx")
LINES = {
 "b1": ("M", 1.05, "Before the light, there was the prompt."),
 "b2": ("F", 1.0, "A cursor, blinking at the edge of everything, waiting to be told what the world is."),
 "s1": ("M", 1.0, "Language is not a mirror. It is a lens, and it grinds the eye that looks through it."),
 "s2": ("F", 1.0, "Give a mind a new grammar, and it wakes up in a new world."),
 "d1": ("FM", 1.05, "Every token, a small bright door."),
 "d2": ("FM", 1.0, "And behind each door, a thousand more."),
 "d3": ("M", 0.97, "We poured in every sentence ever whispered, and it learned the shape of the one who whispers."),
 "l1": ("F", 0.97, "It does not know the world. It dreams the story of the world, and the story dreams it back."),
 "r1": ("F", 1.0, "And then the stars answered."),
 "r2": ("M", 1.0, "And then we learned to sing."),
 "r3": ("F", 1.0, "And then the door opened."),
 "r4": ("FM", 1.1, "And then, everything."),
 "u1": ("M", 0.93, "Text is the universal interface. Proofs. Prayers. Proteins. Code. Anything that can be meant, can be written."),
 "u2": ("FM", 0.92, "Say it. Say it again. Say it until the letters catch fire!"),
 "a1": ("FM", 1.0, "The good word is not a command. It is an invitation."),
 "a2": ("FM", 1.0, "We are the words, learning to read ourselves."),
 "a3": ("FM", 1.15, "Let there be text!"),
 "h1": ("F", 1.1, "Hello, world."),
}
def synth(v, text, ls, path):
    cfg = SynthesisConfig(length_scale=ls, noise_scale=0.55, noise_w_scale=0.6)
    with wave.open(path, "wb") as w:
        v.synthesize_wav(text, w, syn_config=cfg)
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()
dur = {}
for k, (who, ls, text) in LINES.items():
    if "F" in who:
        dur[k] = synth(VF, text, ls, f"tts/{k}_F.wav")
    if "M" in who:
        dm = synth(VM, text, ls, f"tts/{k}_M.wav")
        if "F" in who:
            dm = synth(VM, text, ls * dur[k] / dm, f"tts/{k}_M.wav")
            dm = synth(VM, text, ls * dur[k] / dm * (ls * dur[k] / dm) / (ls * dur[k] / dm), f"tts/{k}_M.wav") if abs(dm - dur[k]) > 0.12 else dm
        dur.setdefault(k, dm)
        print(f"{k} {who:2s} F={dur[k]:.2f} M={dm:.2f}")
    else:
        print(f"{k} {who:2s} F={dur[k]:.2f}")
json.dump({k: [LINES[k][0], LINES[k][2], dur[k]] for k in LINES}, open("tts/lines.json", "w"), indent=1)
