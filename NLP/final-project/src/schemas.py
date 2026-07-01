from pydantic import BaseModel, Field
from typing import Optional


class Paper(BaseModel):
    paper_id: str
    title: str
    authors: str
    year: int
    venue: str = ""
    arxiv_id: str = ""
    url: str = ""
    topic_tag: str = ""
    pdf_filename: str
    needs_review: bool = False


class Chunk(BaseModel):
    chunk_id: str                    # "{paper_id}__{section_idx}__{n}"
    paper_id: str
    title: str
    authors: str
    year: int
    section: str
    topic_tag: str
    text: str                        # prepended with "Title — Section\n\n"

    def to_chroma_metadata(self) -> dict:
        """Return a flat dict with only str/int/float/bool values."""
        return {
            "chunk_id":  self.chunk_id,
            "paper_id":  self.paper_id,
            "title":     self.title,
            "authors":   self.authors,
            "year":      self.year,
            "section":   self.section,
            "topic_tag": self.topic_tag,
        }
