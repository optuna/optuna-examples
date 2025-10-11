import subprocess, sys, os, re

def test_svm_tfidf_tweeteval_runs_quickly():
    # run 1 quick trial with a tiny train subset to keep CI fast
    cmd = [
        sys.executable,
        os.path.join("examples", "sklearn", "svm_tfidf_tweeteval_sentiment.py"),
        "--n-trials","1","--max-train","800","--seed","0",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    # should emit both "Best validation" and "Test set"
    assert "Best validation (1 - macroF1):" in out
    assert "=== Test set ===" in out
