"""Fake OpenAI-compatible server for rehearsing the whole pipeline without a GPU
or API budget. It recognises each prompt type in this repo and returns a
well-formed (meaningless) answer, so every stage, file format and metric can be
checked end to end in minutes.

  python sol/mock_llm.py 8099 &
  LLM_API_BASE=http://localhost:8099/v1 WORKERS=32 bash sol/run_fit_pipeline.sh
"""
import hashlib, json, re, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def h(s, n):
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % n


def answer(msgs):
    text = "\n".join(m["content"] for m in msgs)
    last = msgs[-1]["content"]
    ids = re.findall(r"^\[([^\]\s]+)\]", text, flags=re.M)
    if "generalising holdings" in text:                      # generalize.py
        ids = list(dict.fromkeys(ids))
        return json.dumps(dict(
            legal_question="Mock question?", rule={"if": "p_a and not p_b", "then": "outcome_c"},
            predicates={"p_a": "Condition A holds.", "p_b": "Condition B holds.", "outcome_c": "Outcome C follows."},
            definition="If A and not B, then C.", exceptions_or_limits=["B"], covered_members=ids,
            outlier_members=[], recurring_patterns=dict(common_material_facts=["f"], statutes_and_provisions=["s"],
                                                        typical_outcome="C", variations=["v"]),
            coherence=["high", "medium", "low"][h(",".join(ids), 3)], split_suggestion=None))
    if "Group them by the legal point" in last:              # refine.py split
        ids = list(dict.fromkeys(ids)); half = max(1, len(ids) // 2)
        return json.dumps(dict(groups=[dict(legal_point="x", members=ids[:half]), dict(legal_point="y", members=ids[half:])]))
    if "Classify the relationship" in last:                  # rule_conflicts.py
        return json.dumps(dict(relation=["SAME", "DIFFERENT", "REFINEMENT", "DIFFERENT", "CONFLICT"][h(last, 5)],
                               explanation="DISTINCTION", note="mock"))
    if "predicting how" in last:                             # loo_eval / validate predictor
        return json.dumps(dict(applies=True, predicted_conclusion="Mock conclusion."))
    if "SAME_POINT" in last:
        return ["SAME_POINT", "RELATED", "DIFFERENT"][h(last, 3)]
    if "EXACT" in last:
        return ["EXACT", "SUBSTANCE", "WRONG"][h(last, 3)]
    if "APPLIES, PARTIAL or NO" in last:
        return ["APPLIES", "PARTIAL", "NO"][h(last, 3)]
    if "MATCH, PARTIAL or MISMATCH" in last:
        return "MATCH"
    if "YES or NO" in last:
        return "YES"
    return "OK"


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        self._send(dict(object="list", data=[dict(id="mock")]))

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self._send(dict(choices=[dict(index=0, message=dict(role="assistant", content=answer(req["messages"])))]))


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 8099), H).serve_forever()
