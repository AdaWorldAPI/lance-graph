#!/usr/bin/env python3
"""W6: harvest pinned real Odoo inputs for the shared projection test.

Does not compile Odoo or merge its source metamodel with SAP's.
Usage: python3 tools/harvest_odoo_view.py /path/to/odoo-rs
"""
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

root = Path(sys.argv[1])
pins = {
    'crates/od-ontology/src/view_mask.rs': '723e50cf4ed929853369267e0e1d9061f6a07cd2',
    'data/account_move_form_view.xml': '32dcce31d739d85b34887c73f9aa993ccef35bfa',
    'data/slice_2.spo.ndjson': 'a854b501aa4a595385c7e60d4a22e6058ce3d315',
}
for path, sha in pins.items():
    data = (root/path).read_bytes()
    assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() == sha
source = (root/'crates/od-ontology/src/view_mask.rs').read_text()
assert 'lance_graph_contract::class_view::WideFieldMask::from_universe_present(' in source
universe = set()
for line in (root/'data/slice_2.spo.ndjson').read_text().splitlines():
    t = json.loads(line)
    if t['p'] == 'rdf:type' and t['o'] == 'ogit:Property' and t['s'].startswith('odoo:account_move.'):
        name = t['s'][len('odoo:account_move.'):]
        if '.' not in name: universe.add(name)
record = ET.fromstring((root/'data/account_move_form_view.xml').read_text())
arch = record.find("field[@name='arch']")
present = set()
def visit(node):
    for child in node:
        if child.tag == 'field':
            present.add(child.attrib['name'])
            # Descendants belong to a comodel, not account.move.
        else: visit(child)
visit(arch)
assert len(universe) == 216 and len(universe & present) == 57
out = Path(__file__).resolve().parents[1]/'fixtures/odoo-view.tsv'
out.write_text('field\tpresent\n'+''.join(f'{name}\t{int(name in present)}\n' for name in sorted(universe)))
print('PASS: pinned real account.move universe=216, view intersection=57; common WideFieldMask owner confirmed.')
