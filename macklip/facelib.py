"""Yüz bulma ve tanıma (dlib). face_recognition paketinin kullandığımız kısmı; o paket kurulurken
dlib'i kaynaktan derlemeye çalıştığı için hazır derlenmiş dlib-bin ile doğrudan çalışıyoruz."""
import warnings

import dlib
import numpy as np
from PIL import Image

with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # pkg_resources uyarısı
    import face_recognition_models as _models

_detector = dlib.get_frontal_face_detector()
_pose = dlib.shape_predictor(_models.pose_predictor_five_point_model_location())
_encoder = dlib.face_recognition_model_v1(_models.face_recognition_model_location())


def load_image_file(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def face_locations(image: np.ndarray, number_of_times_to_upsample: int = 1) -> list[tuple[int, int, int, int]]:
    """(üst, sağ, alt, sol) kutuları döndürür."""
    h, w = image.shape[:2]
    return [(max(r.top(), 0), min(r.right(), w), min(r.bottom(), h), max(r.left(), 0))
            for r in _detector(image, number_of_times_to_upsample)]


def face_encodings(image: np.ndarray, known_face_locations=None) -> list[np.ndarray]:
    locs = known_face_locations if known_face_locations is not None else face_locations(image)
    out = []
    for top, right, bottom, left in locs:
        shape = _pose(image, dlib.rectangle(left, top, right, bottom))
        out.append(np.array(_encoder.compute_face_descriptor(image, shape, 1)))
    return out


def face_distance(known: list[np.ndarray], candidate: np.ndarray) -> np.ndarray:
    if len(known) == 0:
        return np.empty(0)
    return np.linalg.norm(np.asarray(known) - candidate, axis=1)
