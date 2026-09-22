#!/usr/bin/env python3
"""Verify the harvest against pinned external checkouts (never runtime input).

Usage: python3 tools/verify_sources.py /path/containing/SIMAF/SIMAFPort/...
The sources stay in their original repositories; only descriptors are shipped.
"""
import csv
import hashlib
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parents[1]
root = Path(sys.argv[1])
for pin in csv.DictReader((HERE / 'sources.tsv').open(), delimiter='\t'):
    data = (root / pin['repo'] / pin['path']).read_bytes()
    actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    assert actual == pin['blob'], (pin['path'], actual, pin['blob'])

abap = (root / 'SIMAFPort/V4/complete/CLASSES/TimeTracking/ZCL_TIME_TRACKING_DTO_POC.clas.abap.txt').read_text()
cs = (root / 'SIMAFPort/src/UniversalDtoPoc/TimeTracking/TimeTrackingDtos.cs').read_text()
schema_cs = (root / 'SIMAF/Schema/UniversalDtoPoc.TimeTracking.txt').read_text()
leaves = []
for group, body in re.findall(r'BEGIN OF (ty_s_\w+),(.*?)END OF \1', abap, re.S):
    leaves += [(group, name, typ) for name, typ in re.findall(r'^\s*(\w+)\s+TYPE\s+(\w+)', body, re.M) if not typ.startswith('ty_s_')]
descriptors = list(csv.DictReader((HERE / 'schema.tsv').open(), delimiter='\t'))
assert len(leaves) == len(descriptors) == 23
for ordinal, (leaf, field) in enumerate(zip(leaves, descriptors)):
    assert int(field['ordinal']) == ordinal
    assert leaf == (field['abap_group'], field['abap_name'], field['native_type'])
    for source in [cs, schema_cs]:
        typ = re.search(r'public (string\??|decimal|bool) ' + field['csharp_name'] + r' \{', source)[1]
        assert typ.endswith('?') == (field['optional'] == 'true')
    if field['width'] and field['native_type'] != 'pernr_d':
        before = cs[:cs.index('public string' + ('?' if field['optional'] == 'true' else '') + ' ' + field['csharp_name'])]
        assert re.findall(r'\[StringLength\((\d+)\)\]', before)[-1] == field['width']
    if field['native_type'] == 'pernr_d':
        assert field['width'] == '8' and '^\\d{8}$' in cs

# Source-derived disagreement is an oracle result, never a claimed parity pass.
processor = (root / 'SIMAFPort/V4/complete/CLASSES/TimeTracking/ZCL_TIME_DTO_PROCESSOR.clas.abap.txt').read_text()
hasher = (root / 'SMB-Core-Middleware/src/Core/Infrastructure/TimeTracking/TimeTrackingHasher.cs').read_text()
abap_order = re.findall(r'APPEND \|(\w+)=', processor)
cs_order = re.findall(r'sb.Append\((?:NormalizeField\()?entry\.(\w+)', hasher)
assert [n.lower() for n in abap_order] == [n.lower() for n in cs_order]
assert len(abap_order) == 16
assert 'value.Trim().ToUpperInvariant()' in hasher
assert 'APPEND |EntryID=' in processor
print('PASS: 23 ABAP/C# field bindings; 16 hash ordinals; source blobs pinned.')
print('FALSIFIED: hash bytes differ (ABAP named values, SMB bare normalized values).')
