import os, sys
sys.path.append(os.path.abspath('../'))
from data.datasets import renderedLaTeXDataset

def test_renderedLaTeXDataset(dataset, processor):
    
    iter_ = iter(dataset)
    inputs, captions = next(iter_)
    inputs_2, captions_2 = next(iter_)
    assert ''.join(processor.batch_decode(captions, skip_special_tokens=True)) != ''.join(processor.batch_decode(captions_2, skip_special_tokens=True)), "Passed dataset yields repeat captions."

    print("renderedLaTeXDataset tests passed.")