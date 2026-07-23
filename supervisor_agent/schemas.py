from typing import Literal

from pydantic import BaseModel, Field

# Shared Pydantic schemas, kept in their own module (rather than inside
# researcher.py) so both researcher.py and copywriter.py can import them
# without importing each other.


class ResearchReport(BaseModel):
    topic: str = Field(description="The specific research angle/subtopic this report covers")
    report: str = Field(description="Full markdown report body, ending with a Citations section")


class GeneratedContent(BaseModel):
    content_type: Literal["blog", "linkedin"] = Field(description="Which kind of content this is")
    title: str | None = Field(default=None, description="Title, if applicable (e.g. blog posts)")
    content: str = Field(description="The full generated content body")
