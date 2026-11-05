"""Validate source hashes and the reported exact radial enclosures."""
from pathlib import Path
from fractions import Fraction as Q
import json, hashlib
if not __debug__:
    raise RuntimeError("Run with assertions enabled")
root=Path(__file__).resolve().parent
report=json.loads((root/'verification.json').read_text())
assert report['passed']
for name,digest in report['files_sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
for item in report['normalized_local_certificate']:
    assert item['passed']
    for gadget,lo,hi in [('A',Q(307,1000),Q(308,1000)),('B',Q(451,1000),Q(452,1000))]:
        bounds=item[gadget]
        assert Q(bounds['x_lo'][0])>lo
        assert Q(bounds['x_hi'][1])<hi
assert Q(308,451)<1 and Q(307,1000)/Q(452,1000)**2>1
print('PASS: report hashes and optional run-length radial bounds')

sage_path=root/'replication/verification_sage.json'
if sage_path.exists():
    sage=json.loads(sage_path.read_text())
    assert sage['passed']
    for name,digest in sage['files_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
    for key in ['uniform_rows','margin','strict_prefix_steps',
                'selected_columns_zeroindexed','burst_word','norms']:
        assert sage[key]==report[key], key
    assert all(item['passed'] for item in sage['normalized_local_certificate'])
    print('PASS: current Sage source hashes, itinerary, encoding, margin and norm agreement')
