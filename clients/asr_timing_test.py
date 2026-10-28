"""How soon after a caller stops talking their final transcript arrives (captions topic), e.g. to
see what background noise removal (BNR_GRPC_URL) adds. Needs no translation or speech service.
    docker run --rm --network host -v $PWD/clients:/clients -v $PWD/testdata:/td -w /clients \\
        indic-s2s:0.1 python /clients/asr_timing_test.py --runs 5
"""
import argparse, asyncio, statistics, time
from call_test import Caller, read_wav, wait_for
from call_test_en_hi import trimmed


async def main(a):
    room = f"asr-timing-{int(time.time())}"
    A, B = Caller("Asha", "hi", "hi", "female"), Caller("Ben", "en", "en", "male")
    for p in (A, B):
        await p.join(a.url, a.api, room)
    assert await wait_for(lambda: any(len(m.get("callers", [])) == 2 for m in A.msgs("status")), 20), "no translator"
    await asyncio.sleep(1.0)
    def finals(who, since):
        return [(ts, m) for ts, m in who.messages
                if ts >= since and m.get("type") == "transcript" and m.get("final") and m.get("text")]

    delays = []
    for i in range(a.runs):
        for who, clip in ((A, a.hindi), (B, a.english)):
            pcm, sr = trimmed(read_wav(clip))
            start = time.time()
            await who.say(pcm, sr, then_silence_s=1.5)
            spoken_end = start + len(pcm) / sr
            if await wait_for(lambda: finals(who, start), 10):
                ts, m = finals(who, start)[-1]               # the utterance's last final (a sentence may be its own)
                delays.append((ts - spoken_end) * 1000)
                print(f"{who.speak}: final transcript {delays[-1]:.0f} ms after speech end: {m['text'][:60]}")
            await asyncio.sleep(3.0)
    for p in (A, B):
        await p.room.disconnect()
    print(f"\nfinal transcript after speech end: median {statistics.median(delays):.0f} ms over {len(delays)} utterances")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://127.0.0.1:7880")
    p.add_argument("--api", default="http://127.0.0.1:8090")
    p.add_argument("--hindi", default="/td/sample_hindi.wav")
    p.add_argument("--english", default="/td/english_16k.wav")
    p.add_argument("--runs", type=int, default=3)
    asyncio.run(main(p.parse_args()))
