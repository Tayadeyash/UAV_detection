"""Utilities for inspecting UAV box geometry before anchor tuning."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

def collect_box_stats(root):
    widths=[]; heights=[]; areas=[]
    for f in Path(root).glob('*.xml'):
        r=ET.parse(f).getroot(); size=r.find('size')
        W=float(size.findtext('width')) if size is not None else 1
        H=float(size.findtext('height')) if size is not None else 1
        for o in r.findall('object'):
            b=o.find('bndbox')
            if b is None: continue
            w=max(1,float(b.findtext('xmax'))-float(b.findtext('xmin')))
            h=max(1,float(b.findtext('ymax'))-float(b.findtext('ymin')))
            widths.append(w/W); heights.append(h/H); areas.append((w*h)/(W*H))
    return {'count':len(widths),'width_p50':float(np.percentile(widths,50)) if widths else 0,'height_p50':float(np.percentile(heights,50)) if heights else 0,'area_p10':float(np.percentile(areas,10)) if areas else 0}

if __name__ == '__main__':
    import argparse, pprint
    p=argparse.ArgumentParser(); p.add_argument('--data',default='dataset'); a=p.parse_args()
    pprint.pp(collect_box_stats(a.data))
