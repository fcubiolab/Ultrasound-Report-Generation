#!/usr/bin/env python

import os
import subprocess
import threading

java_path = subprocess.run(["where", "java"], capture_output=True, text=True)
print("使用中的 Java 路徑：", java_path.stdout)

METEOR_JAR = 'meteor-1.5.jar'

class Meteor:
    def __init__(self):
        self.meteor_cmd = ['java', '-jar', '-Xmx2G', METEOR_JAR,
                           '-', '-', '-stdio', '-l', 'en', '-norm']
        self._start_meteor()
        self.lock = threading.Lock()
        self._read_stderr()

    def _start_meteor(self):
        self.meteor_p = subprocess.Popen(
            self.meteor_cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1
        )

    def _restart_meteor(self):
        print("[INFO] Restarting METEOR subprocess...")
        try:
            self.meteor_p.kill()
            self.meteor_p.wait()
        except Exception:
            pass
        self._start_meteor()

    def _sanitize(self, text):
        return text.replace('\n', ' ').replace('\r', ' ').replace('|||', '').strip()

    def compute_score(self, gts, res):
        assert gts.keys() == res.keys()
        imgIds = gts.keys()
        scores = []

        eval_line = 'EVAL'
        with self.lock:
            try:
                for i in imgIds:
                    assert len(res[i]) == 1
                    stat = self._stat(res[i][0], gts[i])
                    eval_line += ' ||| {}'.format(stat)

                self._safe_write(eval_line)
                for _ in range(len(imgIds)):
                    scores.append(float(self.meteor_p.stdout.readline().decode('utf-8').strip()))
                score = float(self.meteor_p.stdout.readline().decode('utf-8').strip())

            except Exception as e:
                print(f"[METEOR ERROR] compute_score failed: {e}")
                score = 0.0
                scores = [0.0 for _ in imgIds]

        return score, scores

    def method(self):
        return "METEOR"

    def _stat(self, hypothesis_str, reference_list):
        hypothesis_str = self._sanitize(hypothesis_str)
        reference_list = [self._sanitize(r) for r in reference_list]
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))

        try:
            self._safe_write(score_line)
            return self.meteor_p.stdout.readline().decode('utf-8').strip()
        except Exception as e:
            print(f"[METEOR ERROR] _stat failed: {e}")
            return "0.0"

    def _score(self, hypothesis_str, reference_list):
        hypothesis_str = self._sanitize(hypothesis_str)
        reference_list = [self._sanitize(r) for r in reference_list]

        with self.lock:
            try:
                score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
                self._safe_write(score_line)
                stats = self.meteor_p.stdout.readline().decode('utf-8').strip()
                eval_line = 'EVAL ||| {}'.format(stats)
                self._safe_write(eval_line)
                _ = self.meteor_p.stdout.readline().decode('utf-8').strip()  # individual score
                score = float(self.meteor_p.stdout.readline().decode('utf-8').strip())
            except Exception as e:
                print(f"[METEOR ERROR] _score failed: {e}")
                score = 0.0

        return score

    def _safe_write(self, line):
        if self.meteor_p.stdin is None or self.meteor_p.poll() is not None:
            self._restart_meteor()
        try:
            self.meteor_p.stdin.write((line + '\n').encode('utf-8'))
            self.meteor_p.stdin.flush()
        except Exception as e:
            raise RuntimeError(f"Failed to write to METEOR subprocess: {e}")

    def _read_stderr(self):
        def reader(pipe):
            for line in iter(pipe.readline, b''):
                print("[METEOR STDERR]", line.decode('utf-8').strip())
        threading.Thread(target=reader, args=(self.meteor_p.stderr,), daemon=True).start()

    def close(self):
        with self.lock:
            try:
                if self.meteor_p.stdin:
                    self.meteor_p.stdin.close()
                self.meteor_p.kill()
                self.meteor_p.wait()
            except Exception:
                pass