import pytest

from ingest import save_uploaded_pdfs


class _FakeUploadedFile:
    """Minimal stand-in for streamlit's UploadedFile -- just .name and .getvalue()."""

    def __init__(self, name: str, content: bytes = b"%PDF-1.4 fake content"):
        self.name = name
        self._content = content

    def getvalue(self) -> bytes:
        return self._content


def test_save_uploaded_pdfs_writes_files(tmp_path):
    papers_dir = tmp_path / "papers"
    files = [_FakeUploadedFile("a.pdf", b"content-a"), _FakeUploadedFile("b.pdf", b"content-b")]

    saved = save_uploaded_pdfs(papers_dir, files)

    assert {p.name for p in saved} == {"a.pdf", "b.pdf"}
    assert (papers_dir / "a.pdf").read_bytes() == b"content-a"
    assert (papers_dir / "b.pdf").read_bytes() == b"content-b"


def test_save_uploaded_pdfs_creates_papers_dir_if_missing(tmp_path):
    papers_dir = tmp_path / "does" / "not" / "exist"
    save_uploaded_pdfs(papers_dir, [_FakeUploadedFile("a.pdf")])
    assert (papers_dir / "a.pdf").exists()


def test_save_uploaded_pdfs_rejects_non_pdf(tmp_path):
    with pytest.raises(ValueError, match="Not a PDF"):
        save_uploaded_pdfs(tmp_path / "papers", [_FakeUploadedFile("malware.exe")])


def test_save_uploaded_pdfs_is_case_insensitive_about_extension(tmp_path):
    papers_dir = tmp_path / "papers"
    save_uploaded_pdfs(papers_dir, [_FakeUploadedFile("UPPER.PDF")])
    assert (papers_dir / "UPPER.PDF").exists()


def test_save_uploaded_pdfs_strips_directory_components_from_filename(tmp_path):
    # A crafted filename shouldn't be able to write outside papers_dir.
    papers_dir = tmp_path / "papers"
    outside = tmp_path / "escaped.pdf"
    save_uploaded_pdfs(papers_dir, [_FakeUploadedFile("../escaped.pdf")])

    assert not outside.exists()
    assert (papers_dir / "escaped.pdf").exists()


def test_save_uploaded_pdfs_overwrites_existing_file_of_the_same_name(tmp_path):
    papers_dir = tmp_path / "papers"
    save_uploaded_pdfs(papers_dir, [_FakeUploadedFile("a.pdf", b"old")])
    save_uploaded_pdfs(papers_dir, [_FakeUploadedFile("a.pdf", b"new")])
    assert (papers_dir / "a.pdf").read_bytes() == b"new"
