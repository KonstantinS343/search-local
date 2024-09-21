# Author: Vodohleb04
class SnipetBounds:
    """
    Dataclass to save snippet bounds.

    snippet_start - index from the original text, that corresponds to the first char of the snippet
    snippet_end - index from the original text, that corresponds to the after-last char of the snippet

    snippet = original_text[snippet_start : snippet_end]  (original_text[snippet_end] is excluded from the snippet)
    """

    def __init__(self, snippet_start, snippet_end):
        self._snippet_start = snippet_start
        self._snippet_end = snippet_end

    
    @property
    def snippet_start(self) -> str:
        """
        snippet = original_text[snippet_start : snippet_end]  (original_text[snippet_end] is excluded from the snippet)

        returns: int - index from the original text, that corresponds to the first char of the snippet
        """
        return str(self._snippet_start.item())
    

    @property
    def snippet_end(self) -> str:
        """
        snippet = original_text[snippet_start : snippet_end]  (original_text[snippet_end] is excluded from the snippet)

        returns: int - index from the original text, that corresponds to the after-last char of the snippet
        """
        return str(self._snippet_end.item())


    def __str__(self):
        return f"SnippetBounds({self._snippet_start}, {self._snippet_end})"
    

    def __repr__(self):
        return f"SnippetBounds(snippet_start: {self._snippet_start}, snippet_end: {self._snippet_end})"

