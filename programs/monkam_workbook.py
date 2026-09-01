"""Streaming reader for author-supplied XLSX numeric feature workbooks."""
from __future__ import annotations
import re, zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
import numpy as np

NS="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
def _column(ref):
    result=0
    for char in re.match(r"[A-Z]+",ref.upper()).group(0): result=result*26+ord(char)-64
    return result-1
def _strings(archive):
    if 'xl/sharedStrings.xml' not in archive.namelist(): return []
    root=ET.fromstring(archive.read('xl/sharedStrings.xml'))
    return [''.join(n.text or '' for n in item.findall(f'.//{{{NS}}}t'))
            for item in root.findall(f'{{{NS}}}si')]
def _value(cell,strings):
    value=cell.find(f'{{{NS}}}v')
    if value is None:
        inline=cell.find(f'.//{{{NS}}}t'); return inline.text if inline is not None else None
    raw=value.text or ''; return strings[int(raw)] if cell.attrib.get('t')=='s' else raw
def load_numeric_workbook(path: str|Path):
    path=Path(path)
    with zipfile.ZipFile(path) as archive:
        strings=_strings(archive)
        workbook=ET.fromstring(archive.read('xl/workbook.xml'))
        rels=ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))
        relmap={r.attrib['Id']:r.attrib['Target'] for r in rels}
        sheet=workbook.find(f'.//{{{NS}}}sheet')
        rid=sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
        target=relmap[rid].lstrip('/'); member=target if target.startswith('xl/') else 'xl/'+target
        header=None; rows=[]; labels=[]
        with archive.open(member) as stream:
            for _,row in ET.iterparse(stream,events=('end',)):
                if row.tag!=f'{{{NS}}}row': continue
                cells={_column(c.attrib['r']):_value(c,strings) for c in row.findall(f'{{{NS}}}c')}
                values=[cells.get(i) for i in range(max(cells,default=-1)+1)]
                if header is None: header=values; row.clear(); continue
                has_label=header[-1]=='label'; n=len(header)-int(has_label)
                rows.append([float(v) for v in values[:n]])
                if has_label: labels.append(str(values[n]))
                row.clear()
    return np.asarray(rows,dtype=np.float64), (np.asarray(labels,dtype=object) if labels else None), header
