from argparse import ArgumentParser
import wave
from os import path
import numpy as np
from dsp import split_bands, addnbuffers, stretch_ir
from PIL import Image as Im


def apply_ir(audio_in_path, audio_out_path, args, ir=None):
    # load audio
    with wave.open(audio_in_path, 'rb') as ai:
        params = ai.getparams()
        framebytes = ai.readframes(params.nframes * params.nchannels)
    dt = np.dtype(np.int16)
    dt = dt.newbyteorder('<')
    frames = np.frombuffer(framebytes, dtype=dt)

    bands = split_bands(frames, params.framerate)

    # load ir
    if ir is None:
        ir_length_seconds = 0.05
        ir_img = Im.open(path.join(args.working_directory, args.ir_in))
        irs = np.array(ir_img.getdata()).reshape((*ir_img.size, 3))
        irs = irs / 255
        ir = irs[:, 49, :]
    else:
        ir_length_seconds = args.time_interval

    results = []
    for dx, band in enumerate(bands):
        bir = ir[:, dx]
        bir = stretch_ir(bir, ir_length_seconds, params.framerate)
        bres = np.convolve(band, bir).astype(np.int16)
        results.append(bres)
    result = addnbuffers(results)

    params = params._replace(nframes=len(result))
    framebytes_out = result.tobytes()
    with wave.open(audio_out_path, 'wb') as ao:
        ao.setparams(params)
        ao.writeframes(framebytes_out)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--working_directory', type=str, default=r'C:\Users\lucad\downloads')
    parser.add_argument('-i', '--ir_in', type=str, default="raw_ir_groundt.png")
    parser.add_argument('-a', '--audio_in', type=str, default="in.wav")
    parser.add_argument('-o', '--audio_out', type=str, default="out.wav")

    args = parser.parse_args()

    apply_ir(path.join(args.working_directory, args.audio_in), path.join(args.working_directory, args.audio_out), args)




