import pytest
from preprocessing import preprocess_image
import numpy as np

@pytest.fixture
def input_image():
    return np.zeros((100, 100, 3), dtype=np.uint8)

def test_grayscale_conversion(input_image):
    expected_shape = (100, 100)
    result_image = preprocess_image(input_image)
    assert result_image.shape == expected_shape
