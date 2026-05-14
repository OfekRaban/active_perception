"""Unit tests for the perception module (no GPU required, uses CPU)."""
import pytest
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from active_perception.model.perception_module import PerceptionModule, QueryAdapter
from active_perception.model.spatial_encoding import SpatialEncoding2D, SpatialEncodingMode


D_MODEL = 64   # small for testing
D_QUERY = 16
N_HEADS = 4
N_PATCHES = 49  # 7x7 grid
B = 2
K = 1           # one perception step


class TestQueryAdapter:
    def test_output_shape(self):
        adapter = QueryAdapter(D_MODEL, D_QUERY)
        x = torch.randn(B, K, D_MODEL)
        out = adapter(x)
        assert out.shape == (B, K, D_MODEL)

    def test_single_vector(self):
        adapter = QueryAdapter(D_MODEL, D_QUERY)
        x = torch.randn(D_MODEL)
        out = adapter(x)
        assert out.shape == (D_MODEL,)


class TestPerceptionModule:
    def setup_method(self):
        self.module = PerceptionModule(D_MODEL, D_QUERY, N_HEADS, residual=True)

    def test_batched_forward_shapes(self):
        h = torch.randn(B, K, D_MODEL)
        mem = torch.randn(B, N_PATCHES, D_MODEL)
        z, attn = self.module(h, mem)
        assert z.shape == (B, K, D_MODEL)
        assert attn.shape == (B, K, N_PATCHES)

    def test_unbatched_forward_shapes(self):
        h = torch.randn(K, D_MODEL)
        mem = torch.randn(N_PATCHES, D_MODEL)
        z, attn = self.module(h, mem)
        assert z.shape == (K, D_MODEL)
        assert attn.shape == (K, N_PATCHES)

    def test_shared_memory_broadcast(self):
        """Single visual memory shared across batch."""
        h = torch.randn(B, K, D_MODEL)
        mem = torch.randn(1, N_PATCHES, D_MODEL)  # single image
        z, attn = self.module(h, mem)
        assert z.shape == (B, K, D_MODEL)

    def test_attn_weights_sum_to_one(self):
        h = torch.randn(B, K, D_MODEL)
        mem = torch.randn(B, N_PATCHES, D_MODEL)
        _, attn = self.module(h, mem)
        sums = attn.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_no_attn_weights(self):
        h = torch.randn(B, K, D_MODEL)
        mem = torch.randn(B, N_PATCHES, D_MODEL)
        z, attn = self.module(h, mem, return_attn_weights=False)
        assert z.shape == (B, K, D_MODEL)
        # attn may be None or averaged

    def test_multi_step_K_gt_1(self):
        K2 = 3
        h = torch.randn(B, K2, D_MODEL)
        mem = torch.randn(B, N_PATCHES, D_MODEL)
        z, attn = self.module(h, mem)
        assert z.shape == (B, K2, D_MODEL)
        assert attn.shape == (B, K2, N_PATCHES)

    def test_residual_makes_z_closer_to_h(self):
        """With residual=True, z should be closer to h than without."""
        mod_res = PerceptionModule(D_MODEL, D_QUERY, N_HEADS, residual=True)
        mod_no_res = PerceptionModule(D_MODEL, D_QUERY, N_HEADS, residual=False)
        h = torch.randn(B, K, D_MODEL)
        mem = torch.randn(B, N_PATCHES, D_MODEL)
        z_res, _ = mod_res(h, mem)
        z_no, _ = mod_no_res(h, mem)
        dist_res = (z_res - h).norm()
        dist_no = (z_no - h).norm()
        # Residual should keep z closer to h (smaller perturbation at init)
        assert dist_res <= dist_no + 0.5  # allow small tolerance at random init


class TestSpatialEncoding2D:
    def _make_grid_thw(self, H, W):
        return torch.tensor([[1, H, W]])

    def test_none_mode_passthrough(self):
        enc = SpatialEncoding2D(D_MODEL, SpatialEncodingMode.NONE)
        mem = torch.randn(N_PATCHES, D_MODEL)
        out = enc(mem, self._make_grid_thw(7, 7))
        assert out is mem  # should return input unchanged

    def test_additive_sincos2d_shape(self):
        enc = SpatialEncoding2D(D_MODEL, SpatialEncodingMode.ADDITIVE_SINCOS2D)
        mem = torch.randn(N_PATCHES, D_MODEL)
        out = enc(mem, self._make_grid_thw(7, 7))
        assert out.shape == (N_PATCHES, D_MODEL)

    def test_concat_sincos2d_shape(self):
        enc = SpatialEncoding2D(D_MODEL, SpatialEncodingMode.CONCAT_SINCOS2D)
        mem = torch.randn(N_PATCHES, D_MODEL)
        out = enc(mem, self._make_grid_thw(7, 7))
        assert out.shape == (N_PATCHES, D_MODEL)

    def test_batched_additive(self):
        enc = SpatialEncoding2D(D_MODEL, SpatialEncodingMode.ADDITIVE_SINCOS2D)
        mem = torch.randn(B, N_PATCHES, D_MODEL)
        out = enc(mem, self._make_grid_thw(7, 7))
        assert out.shape == (B, N_PATCHES, D_MODEL)

    def test_pe_not_all_zeros(self):
        enc = SpatialEncoding2D(D_MODEL, SpatialEncodingMode.ADDITIVE_SINCOS2D)
        mem = torch.zeros(N_PATCHES, D_MODEL)
        out = enc(mem, self._make_grid_thw(7, 7))
        assert out.abs().sum() > 0  # PE was added
