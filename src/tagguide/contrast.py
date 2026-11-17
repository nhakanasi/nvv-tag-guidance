"""Tag-local guidance: presence and type contrasts on the model's log-probabilities.

    presence  + l1 * m * (c - c_notag)    c_notag: the same text with the tag removed
    type      + l2 * m * (c - c_conf)     c_conf: the tag replaced by its confuser

c is the conditional log-probability, m the tag's window (1 inside, 0 outside). Each contrast needs one more
conditional pass (OmniVoice) or LM stream (CosyVoice 2) on the modified text; the model adapters run them and call
`texts` and `apply`.
"""


def texts(setting, text, notag, confuser):
    """The conditional inputs a setting needs, in the order `apply` expects: text, then notag if l1, confuser if l2."""
    l1, l2, _ = setting
    return [text] + ([notag] if l1 else []) + ([confuser] if l2 else [])


def apply(lp, c, contrasts, setting, m=None):
    """lp plus the contrast terms. contrasts: log-probabilities of the extra passes, in `texts` order; m: the window
    (broadcastable to c), or None where the caller already restricts the terms to the window."""
    l1, l2, _ = setting
    j = 0
    if l1:
        lp = lp + (l1 * m * (c - contrasts[j]) if m is not None else l1 * (c - contrasts[j]))
        j += 1
    if l2:
        lp = lp + (l2 * m * (c - contrasts[j]) if m is not None else l2 * (c - contrasts[j]))
    return lp
