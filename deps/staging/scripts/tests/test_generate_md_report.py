#!/usr/bin/env python3
# Copyright (c) 2026 Ember Contributors. MIT License.
"""Regression check for MD report PDF merging."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPTS_ROOT))

import generate_md_report as report  # noqa: E402


def _write_single_page_pdf(path: Path, title: str) -> None:
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path))
    c.setFont('Helvetica', 18)
    c.drawString(72, 720, title)
    c.showPage()
    c.save()


def _resolve_pdfinfo() -> str | None:
    for candidate in (
        shutil.which('pdfinfo'),
        '/opt/homebrew/bin/pdfinfo',
        '/usr/local/bin/pdfinfo',
        '/usr/bin/pdfinfo',
    ):
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def _get_page_count(pdf_path: Path) -> int:
    pdfinfo = _resolve_pdfinfo()
    if pdfinfo is None:
        raise RuntimeError('pdfinfo not available for PDF page-count validation')

    result = subprocess.run(
        [pdfinfo, str(pdf_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith('Pages:'):
            return int(line.split(':', 1)[1].strip())
    raise RuntimeError(f'Could not parse page count from pdfinfo output for {pdf_path}')


def main() -> int:
    with tempfile.TemporaryDirectory(prefix='md_report_merge_') as tmpdir:
        tmpdir_path = Path(tmpdir)
        section_a = tmpdir_path / 'section_a.pdf'
        section_b = tmpdir_path / 'section_b.pdf'
        _write_single_page_pdf(section_a, 'Section A')
        _write_single_page_pdf(section_b, 'Section B')

        report_path = report.compile_pdf_simple(
            str(tmpdir_path),
            [str(section_a), str(section_b)],
            {'jobName': 'merge-smoke-test'},
        )

        if report_path is None:
            raise AssertionError('compile_pdf_simple returned None')

        merged_pdf = Path(report_path)
        if not merged_pdf.exists():
            raise AssertionError(f'Merged report not created: {merged_pdf}')

        page_count = _get_page_count(merged_pdf)
        if page_count != 3:
            raise AssertionError(f'Expected 3 pages (title + 2 sections), got {page_count}')

        print(f'PASS merged report page count: {page_count}')
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
