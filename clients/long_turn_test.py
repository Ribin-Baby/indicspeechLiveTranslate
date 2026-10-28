"""A long turn: Ben talks English for ~35 s (11 meeting sentences with short pauses) while Asha
hears Hindi and says nothing. Hindi speech runs longer than the English it translates, so the
translations can fall behind; this reports by how much, per utterance and overall.
    docker run --rm --network host -v $PWD/clients:/clients -v $PWD/testdata:/td -w /clients \\
        indic-s2s:0.1 python /clients/long_turn_test.py
"""
import argparse, asyncio, statistics, time
from call_test import Caller, read_wav, wait_for


async def main(a):
    room = f"long-turn-{int(time.time())}"
    A, B = Caller("Asha", "hi", "hi", "female"), Caller("Ben", "en", "en", "male")
    for p in (A, B):
        await p.join(a.url, a.api, room)
    assert await wait_for(lambda: any(len(m.get("callers", [])) == 2 for m in A.msgs("status")), 20), "no translator"
    await asyncio.sleep(1.0)
    pcm, sr = read_wav(a.english)
    t0 = time.time()
    await B.say(pcm, sr, then_silence_s=1.0)
    finals = lambda: [m for m in B.msgs("transcript", t0) if m["final"] and m["text"] and not m.get("echo")]
    voiced = lambda: [m for m in A.msgs("metrics", t0) if m["listener"] == A.identity]
    await wait_for(lambda: finals() and len(voiced()) >= len(finals()), 120)
    ms = [m["audio_start_ms"] for m in voiced() if m.get("audio_start_ms") is not None]
    for m in voiced():
        print(f"  {m['utt']:>28}: speech starts {m['audio_start_ms']:>5} ms after that sentence ended ({m['audio_s']} s of audio)")
    done = max((ts for ts, _, loud in A.received if loud), default=t0)    # last speech-level frame Asha got
    print(f"\n{len(finals())} sentences, {len(ms)} voiced; delay median {statistics.median(ms):.0f} ms, max {max(ms)} ms; "
          f"last translated audio ended {done - (t0 + len(pcm) / sr):.1f} s after Ben stopped")
    for p in (A, B):
        await p.room.disconnect()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://127.0.0.1:7880")
    p.add_argument("--api", default="http://127.0.0.1:8090")
    p.add_argument("--english", default="/td/monologue_en_16k.wav")
    asyncio.run(main(p.parse_args()))
