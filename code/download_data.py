"""Download the public dataset and require the exact archived checksum."""
from pathlib import Path
import urllib.request,hashlib
ROOT=Path(__file__).resolve().parents[1]
url='https://huggingface.co/datasets/potsawee/wiki_bio_gpt3_hallucination/resolve/main/dataset.json'
data=urllib.request.urlopen(url,timeout=90).read()
assert hashlib.sha256(data).hexdigest()=='9fea0263501d74b192b6f75dae5c25a67db7938e096df4aab18c65c42e83189c', 'Upstream changed: use the archived dataset or inspect version before continuing'
(ROOT/'data').mkdir(exist_ok=True);(ROOT/'data/dataset.json').write_bytes(data)
print('Downloaded and verified',len(data),'bytes')
