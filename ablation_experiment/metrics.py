# -*- coding: utf-8 -*-
"""四個評估指標：EM, CM(Containment Match), ANLS, F1(字元級)"""

from collections import Counter


def normalize(s):
    return (s or "").strip()


def edit_distance(a, b):
    """標準 Levenshtein 編輯距離"""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def exact_match(pred, gt):
    """EM：整句完全一致才計 1 分"""
    return 1.0 if normalize(pred) == normalize(gt) else 0.0


def containment_match(pred, gt):
    """CM (Containment Match)：預測字串完整包含 ground truth 就計 1 分"""
    p, g = normalize(pred), normalize(gt)
    if not g:
        return 1.0 if not p else 0.0
    return 1.0 if g in p else 0.0


def anls(pred, gt, threshold=0.5):
    """ANLS：1 - 正規化編輯距離，低於門檻直接當 0（DocVQA 慣例）"""
    p, g = normalize(pred), normalize(gt)
    if not p and not g:
        return 1.0
    max_len = max(len(p), len(g), 1)
    nls = 1.0 - edit_distance(p, g) / max_len
    return nls if nls >= threshold else 0.0


def char_f1(pred, gt):
    """字元級 F1：以字元 multiset 重疊計算 precision / recall（中文沒有分詞邊界，用字元當作 token）"""
    p, g = normalize(pred), normalize(gt)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    cp, cg = Counter(p), Counter(g)
    overlap = sum((cp & cg).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def score_all(pred, gt, anls_threshold=0.5):
    return {
        "em": exact_match(pred, gt),
        "cm": containment_match(pred, gt),
        "anls": anls(pred, gt, anls_threshold),
        "f1": char_f1(pred, gt),
    }
