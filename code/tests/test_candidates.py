"""Tests for flagging candidate ink regions."""

from __future__ import annotations

import numpy as np

from src.m4_ink_detection.candidates import find_candidates, to_scroll_coordinates


def _page(h=200, w=300):
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    coords = np.stack([np.full_like(yy, 50.0), yy + 1000, xx + 2000], axis=-1)
    return coords


def test_finds_a_confident_interior_region_and_reports_its_scan_position():
    coords = _page()
    prob = np.zeros((200, 300), np.float32)
    prob[80:100, 100:200] = 0.9  # a stroke-like 20x100 blob
    (c,) = find_candidates(prob, coords)
    assert c["bbox"] == [100, 80, 200, 100] and c["area_px"] == 2000 and c["elongation"] > 3
    assert abs(c["centre_zyx"][1] - 1089.5) < 1 and abs(c["centre_zyx"][2] - 2149.5) < 1
    native = to_scroll_coordinates(c, (16000, 7000, 7000))
    assert abs(native["centre_native_zyx"][0] - 16050) < 1e-6 and native["centre_native_um"][0] > 0


def test_rejects_specks_weak_regions_and_edges():
    coords = _page()
    prob = np.zeros((200, 300), np.float32)
    prob[10:14, 10:14] = 0.95        # speck
    prob[100:130, 50:120] = 0.55     # confident enough to threshold but weak on average (< min_mean)
    prob[2:40, 100:160] = 0.95       # touches the page border
    assert find_candidates(prob, coords) == []


def test_pixels_outside_the_page_and_empty_pages_are_safe():
    coords = _page()
    coords[:, 150:] = np.nan
    prob = np.full((200, 300), 0.9, np.float32)
    cands = find_candidates(prob, coords)
    assert all(c["bbox"][2] <= 150 for c in cands)
    assert find_candidates(prob, np.full((200, 300, 3), np.nan, np.float32)) == []


def test_best_candidates_come_first_and_the_list_is_capped():
    coords = _page(400, 600)
    prob = np.zeros((400, 600), np.float32)
    for i in range(12):
        prob[40 + 30 * i : 55 + 30 * i, 100:200] = 0.7 + 0.02 * i
    cands = find_candidates(prob, coords, max_per_page=5)
    assert len(cands) == 5 and [c["score"] for c in cands] == sorted([c["score"] for c in cands], reverse=True)
