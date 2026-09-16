"""Deterministic synthetic support cases for the live pipeline notebook."""

SCENARIOS = [
    ("identity", "Login fails with AUTH-401.", "The SSO certificate expired.",
     "Renewed the SSO certificate and verified login."),
    ("uploads", "Uploads stall at 99 percent.", "A stale lock blocks the upload worker.",
     "Cleared the stale lock and restarted the upload worker."),
    ("database", "Database requests time out.", "An unindexed lookup scans the entire table.",
     "Added an index and verified database latency returned to normal."),
    ("storage", "New backups fail with insufficient space.", "Old snapshots filled the backup disk.",
     "Removed expired snapshots and verified the next backup completed."),
    ("network", "The application cannot reach its API hostname.", "A DNS record points to an old IP.",
     "Corrected the DNS record and verified API connectivity."),
]


def make_cases(count=100):
    """Deterministic support cases: 90 incidents and 10 information requests at count=100."""
    cases = []
    for i in range(count):
        product, symptom, cause, resolution = SCENARIOS[i % len(SCENARIOS)]
        rfi = i % 10 == 9
        notes = (
            [("request", "Please send the product brochure and pricing information.")]
            if rfi else [
                ("symptom", symptom),
                ("investigation", f"Support reproduced the {product} issue in region-{i % 3}."),
                ("hypothesis", "A recent deployment was suspected; rollback did not help."),
                ("finding", cause),
                ("resolution", resolution),
                ("confirmation", "Customer confirmed recovery; monitoring stayed healthy."),
            ]
        )
        cases.append({
            "id": f"CASE-{i + 1:03d}",
            "metadata": {"product": product, "region": f"region-{i % 3}", "synthetic": True},
            "artifacts": [
                {"sequence": n, "kind": kind, "text": text}
                for n, (kind, text) in enumerate(notes, 1)
            ],
        })
    return cases

