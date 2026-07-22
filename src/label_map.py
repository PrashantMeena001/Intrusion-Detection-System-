"""
src/label_map.py

Single source of truth for raw NSL-KDD attack_type -> 5-class mapping.
Copied exactly from preprocessing.ipynb Cell 2 — this file exists so
that mapping never has to be retyped (and risk drifting) across
scratch notebooks, sample_records generation, or the Streamlit app.
"""

ATTACK_MAP = {
    "normal": "Normal",
    "back": "DoS",
    "land": "DoS",
    "neptune": "DoS",
    "pod": "DoS",
    "smurf": "DoS",
    "teardrop": "DoS",
    "apache2": "DoS",
    "udpstorm": "DoS",
    "processtable": "DoS",
    "worm": "DoS",
    "ipsweep": "Probe",
    "nmap": "Probe",
    "portsweep": "Probe",
    "satan": "Probe",
    "mscan": "Probe",
    "saint": "Probe",
    "ftp_write": "R2L",
    "guess_passwd": "R2L",
    "imap": "R2L",
    "multihop": "R2L",
    "phf": "R2L",
    "spy": "R2L",
    "warezclient": "R2L",
    "warezmaster": "R2L",
    "sendmail": "R2L",
    "named": "R2L",
    "snmpgetattack": "R2L",
    "snmpguess": "R2L",
    "xlock": "R2L",
    "xsnoop": "R2L",
    "httptunnel": "R2L",
    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "perl": "U2R",
    "rootkit": "U2R",
    "ps": "U2R",
    "sqlattack": "U2R",
    "xterm": "U2R",
}

LABEL_MAP = {"Normal": 0, "DoS": 1, "Probe": 2, "R2L": 3, "U2R": 4}


def get_true_class(attack_type: str) -> str:
    """
    Map a raw attack_type string (e.g. 'neptune', 'satan') to its
    5-class name (e.g. 'DoS', 'Probe'). Raises if the attack_type is
    unrecognized rather than silently returning None — an unknown
    attack_type at test time is worth knowing about loudly, not
    something to paper over (matches the dropna-on-unmapped behavior
    in preprocessing.ipynb, just surfaced instead of silently dropped).
    """
    if attack_type not in ATTACK_MAP:
        raise KeyError(
            f"Unknown attack_type '{attack_type}' — not in ATTACK_MAP. "
            f"Check preprocessing.ipynb Cell 2 for new/renamed attack types."
        )
    return ATTACK_MAP[attack_type]


def get_true_label_int(attack_type: str) -> int:
    """Map a raw attack_type string directly to its integer label (0-4)."""
    return LABEL_MAP[get_true_class(attack_type)]
