from typing import Any

from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class SemanticVQVAEDataset(BaseSegDataset):
    METAINFO = dict(
        classes=('Background', 'Building', 'Road', 'Water', 'Barren', 'Forest', 'Agricultural'),
        palette=[[186, 245, 194], [255, 0, 0], [255, 255, 0], [0, 0, 255], [159, 129, 183], [0, 255, 0], [255, 195, 128]])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)

    def prepare_data(self, idx) -> Any:
        data_info = self.get_data_info(idx)
        results = self.pipeline(data_info)
        results['inputs'] = results['data_samples'].gt_sem_seg.data
        return results
