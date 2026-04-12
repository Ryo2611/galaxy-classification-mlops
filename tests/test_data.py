"""
test_data.py — データパイプラインのユニットテスト

テスト対象:
  - safe_read_fits(): FITS ファイルの安全な読み込みと NaN 処理
  - make_rgb_components(): 3バンド合成の正常系・異常系
  - process_fits_to_rgb(): Lupton RGB 変換パイプライン全体

Usage:
    pytest tests/test_data.py -v
"""

import os
import sys
import tempfile
import numpy as np
import pytest
from PIL import Image

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.preprocess import safe_read_fits, make_rgb_components, process_fits_to_rgb


# =============================================================================
# フィクスチャ: テスト用ダミーFITSファイルの生成
# =============================================================================

@pytest.fixture
def dummy_fits_dir(tmp_path):
    """テスト用にダミーのFITSファイルを生成するフィクスチャ"""
    from astropy.io import fits

    objid = "testgalaxy001"
    bands = ["g", "r", "i"]
    shape = (64, 64)  # 小さい画像でテスト高速化

    for band in bands:
        band_dir = tmp_path / band
        band_dir.mkdir()

        # ランダムな天体画像のようなデータを生成
        np.random.seed(42)
        data = np.random.exponential(scale=100, size=shape).astype(np.float32)

        hdu = fits.PrimaryHDU(data=data)
        hdu.writeto(str(band_dir / f"{objid}_{band}.fits"), overwrite=True)

    return tmp_path, objid


@pytest.fixture
def dummy_fits_with_nan(tmp_path):
    """NaN を含むダミーFITSファイルを生成するフィクスチャ"""
    from astropy.io import fits

    band_dir = tmp_path / "test_nan"
    band_dir.mkdir()

    data = np.ones((32, 32), dtype=np.float32)
    # 一部をNaN / Infにする（実際のFITSでもセンサー不良で発生する）
    data[5:10, 5:10] = np.nan
    data[15, 15] = np.inf
    data[20, 20] = -np.inf

    hdu = fits.PrimaryHDU(data=data)
    filepath = str(band_dir / "nan_test.fits")
    hdu.writeto(filepath, overwrite=True)

    return filepath


# =============================================================================
# safe_read_fits のテスト
# =============================================================================

class TestSafeReadFits:
    """safe_read_fits() のテストスイート"""

    def test_reads_valid_fits(self, dummy_fits_dir):
        """正常なFITSファイルを正しく読み込めるか"""
        fits_dir, objid = dummy_fits_dir
        filepath = str(fits_dir / "g" / f"{objid}_g.fits")

        data = safe_read_fits(filepath)

        assert data is not None
        assert isinstance(data, np.ndarray)
        assert data.shape == (64, 64)

    def test_handles_nan_values(self, dummy_fits_with_nan):
        """NaN / Inf が 0 に置換されるか"""
        data = safe_read_fits(dummy_fits_with_nan)

        assert data is not None
        assert not np.any(np.isnan(data)), "NaN が残っている"
        assert not np.any(np.isinf(data)), "Inf が残っている"

    def test_returns_none_for_missing_file(self):
        """存在しないファイルの場合 None を返すか"""
        result = safe_read_fits("/nonexistent/path/to/file.fits")
        assert result is None

    def test_output_dtype(self, dummy_fits_dir):
        """出力データ型がnumpy arrayか"""
        fits_dir, objid = dummy_fits_dir
        filepath = str(fits_dir / "g" / f"{objid}_g.fits")

        data = safe_read_fits(filepath)
        assert isinstance(data, np.ndarray)


# =============================================================================
# make_rgb_components のテスト
# =============================================================================

class TestMakeRgbComponents:
    """make_rgb_components() のテストスイート"""

    def test_returns_three_bands(self, dummy_fits_dir):
        """i, r, g の3バンドがすべて返されるか"""
        fits_dir, objid = dummy_fits_dir

        i_data, r_data, g_data = make_rgb_components(str(fits_dir), objid)

        assert i_data is not None
        assert r_data is not None
        assert g_data is not None

    def test_correct_shapes(self, dummy_fits_dir):
        """全バンドが同じshapeか"""
        fits_dir, objid = dummy_fits_dir

        i_data, r_data, g_data = make_rgb_components(str(fits_dir), objid)

        assert i_data.shape == r_data.shape == g_data.shape

    def test_missing_band_returns_none(self, tmp_path):
        """1つでもバンドが欠損している場合 (None, None, None) を返すか"""
        # gバンドのみ作成（i, rは作成しない）
        from astropy.io import fits

        g_dir = tmp_path / "g"
        g_dir.mkdir()
        data = np.ones((32, 32), dtype=np.float32)
        hdu = fits.PrimaryHDU(data=data)
        hdu.writeto(str(g_dir / "missing_001_g.fits"), overwrite=True)

        i_data, r_data, g_data = make_rgb_components(str(tmp_path), "missing_001")

        assert i_data is None
        assert r_data is None
        assert g_data is None


# =============================================================================
# process_fits_to_rgb のテスト
# =============================================================================

class TestProcessFitsToRgb:
    """process_fits_to_rgb() のテストスイート"""

    def test_creates_output_directory(self, dummy_fits_dir):
        """出力ディレクトリが自動作成されるか"""
        fits_dir, _ = dummy_fits_dir
        output_dir = str(fits_dir / "output_rgb")

        process_fits_to_rgb(str(fits_dir), output_dir)

        assert os.path.exists(output_dir)

    def test_generates_png_files(self, dummy_fits_dir):
        """PNGファイルが生成されるか"""
        fits_dir, objid = dummy_fits_dir
        output_dir = str(fits_dir / "output_rgb")

        process_fits_to_rgb(str(fits_dir), output_dir)

        expected_file = os.path.join(output_dir, f"{objid}.png")
        assert os.path.exists(expected_file), f"出力ファイルが見つからない: {expected_file}"

    def test_output_is_valid_rgb_image(self, dummy_fits_dir):
        """出力がRGB画像として有効か"""
        fits_dir, objid = dummy_fits_dir
        output_dir = str(fits_dir / "output_rgb")

        process_fits_to_rgb(str(fits_dir), output_dir)

        img_path = os.path.join(output_dir, f"{objid}.png")
        img = Image.open(img_path)

        assert img.mode == "RGB"
        assert img.size[0] > 0
        assert img.size[1] > 0

    def test_handles_empty_raw_dir(self, tmp_path):
        """空のrawディレクトリでもクラッシュしないか"""
        output_dir = str(tmp_path / "output")
        # gバンドフォルダが存在しない場合
        process_fits_to_rgb(str(tmp_path), output_dir)
        # エラーなく完了すればOK
