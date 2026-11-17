"""Inline non-verbal tags in a TTS text: find them, remove them, swap them for their confuser."""
import re


class TagSet:
    """The bracketed tags a model knows, e.g. `[laughter]`, and the tag each one is swapped for in the type
    contrast. `confuser` maps tag names written without brackets: {"laughter": "sigh", ...}."""

    def __init__(self, confuser):
        names = "|".join(re.escape(t) for t in sorted(confuser, key=len, reverse=True))
        self.with_space = re.compile(rf"\s*\[({names})\]")  # a tag and the space before it
        self.span = re.compile(rf"\[({names})\]")
        self.confuser = dict(confuser)

    def has_tag(self, text):
        return self.with_space.search(text) is not None

    def count(self, text):
        return len(self.with_space.findall(text))

    def remove(self, text):
        """The text with every tag (and the space before it) removed: the presence contrast's input."""
        return self.with_space.sub("", text)

    def swap(self, text):
        """The text with every tag replaced by its confuser: the type contrast's input."""
        return self.with_space.sub(lambda m: m.group(0).replace(m.group(1), self.confuser[m.group(1)]), text)


def word_span(text, start, end):
    """Character span from the start of the word before [start, end) to the end of the word after it."""
    before = text[:start].rstrip()
    a = before.rfind(" ") + 1 if " " in before else 0
    rest = text[end:]
    nxt = rest.lstrip()
    lead = len(rest) - len(nxt)
    b = end + lead + (nxt.find(" ") if " " in nxt else len(nxt))
    return a, b
