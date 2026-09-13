"""Extract arXiv HTML sections with TeX math preserved via HTMLParser."""

from html.parser import HTMLParser
from pathlib import Path
import sys


class SectionText(HTMLParser):
    def __init__(self, ids):
        super().__init__()
        self.ids = set(ids)
        self.stack = []
        self.selected = None
        self.math_depth = None
        self.parts = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if attrs.get("id") in self.ids and self.selected is None:
            self.selected = len(self.stack)
            self.parts.append("\n\nSECTION " + attrs["id"] + "\n")
        if self.selected is not None:
            if tag == "math":
                self.parts.append(" $" + attrs.get("alttext", "") + "$ ")
                self.math_depth = len(self.stack)
            elif tag in ("p", "h1", "h2", "h3", "h4", "li", "tr"):
                self.parts.append("\n")
        if tag not in ("meta", "link", "img", "br", "hr", "input", "source", "wbr"):
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag not in self.stack:
            return
        at = len(self.stack) - 1 - self.stack[::-1].index(tag)
        self.stack = self.stack[:at]
        if self.math_depth is not None and at <= self.math_depth:
            self.math_depth = None
        if self.selected is not None and at <= self.selected:
            self.selected = None

    def handle_data(self, value):
        if self.selected is not None and self.math_depth is None:
            self.parts.append(value)


if __name__ == "__main__":
    parser = SectionText(sys.argv[2:])
    parser.feed(Path(sys.argv[1]).read_text())
    print("".join(parser.parts))
