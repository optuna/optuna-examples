import os, sys, subprocess

def test_cli_help_prints_without_running():
    script = os.path.join("examples","sklearn","svm_tfidf_tweeteval_sentiment.py")
    proc = subprocess.run([sys.executable, script, "--help"],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout
    # Should mention our CLI arguments
    assert "--n-trials" in out
    assert "--max-train" in out
