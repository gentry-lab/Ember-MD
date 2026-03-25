#!/usr/bin/env python3
# Copyright (c) 2026 Ember Contributors. MIT License.
"""
Generate comprehensive MD analysis report.

Runs all analysis scripts sequentially and compiles results into a combined PDF report.
Pipeline: contacts → rmsd → rmsf → sse → hbonds → ligand_props → torsions → clustering → compile PDF
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional

warnings.filterwarnings('ignore')


def run_analysis(python_exe: str, script_dir: str, script_name: str, args_list: List[str], step_name: str, event_sender: Any = None) -> Optional[str]:
    """Run a single analysis script and return success status."""
    script_path = os.path.join(script_dir, script_name)
    if not os.path.exists(script_path):
        print(f"Warning: {script_name} not found at {script_path}, skipping", file=sys.stderr)
        return False

    cmd = [python_exe, script_path] + args_list
    print(f"\n=== Running {step_name} ===")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        # Forward stdout
        if result.stdout:
            print(result.stdout, end='')
        if result.returncode != 0:
            print(f"Warning: {step_name} failed (exit code {result.returncode})", file=sys.stderr)
            if result.stderr:
                print(result.stderr[:500], file=sys.stderr)
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"Warning: {step_name} timed out after 600s", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Warning: {step_name} error: {e}", file=sys.stderr)
        return False


def _write_title_page(title_pdf_path: str, section_pdfs: List[str], sim_info: Optional[Dict[str, str]] = None) -> None:
    """Write the report title page to a standalone PDF."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        raise RuntimeError('matplotlib not available, cannot build title page')

    with PdfPages(title_pdf_path) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor('white')

        # Title
        fig.text(0.5, 0.85, 'MD Analysis Report', fontsize=28, fontweight='bold',
                 ha='center', va='top', color='#1f2937')
        fig.text(0.5, 0.80, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                 fontsize=11, ha='center', va='top', color='#6b7280')

        # Simulation details table
        if sim_info:
            table_data = []
            labels = {
                'jobName': 'Job Name',
                'atoms': 'Total Atoms',
                'waters': 'Water Molecules',
                'temperature': 'Temperature',
                'duration': 'Duration',
                'forceField': 'Force Field',
                'platform': 'GPU Platform',
                'performance': 'Performance',
            }
            for key, label in labels.items():
                if key in sim_info and sim_info[key]:
                    table_data.append([label, str(sim_info[key])])

            if table_data:
                ax = fig.add_axes([0.2, 0.35, 0.6, 0.35])
                ax.axis('off')
                table = ax.table(cellText=table_data, colLabels=['Parameter', 'Value'],
                                 cellLoc='left', loc='center', colWidths=[0.4, 0.6])
                table.auto_set_font_size(False)
                table.set_fontsize(11)
                table.scale(1, 1.5)
                for (row, col), cell in table.get_celld().items():
                    if row == 0:
                        cell.set_facecolor('#f3f4f6')
                        cell.set_text_props(fontweight='bold')
                    cell.set_edgecolor('#e5e7eb')

        # Sections summary
        section_count = len([p for p in section_pdfs if os.path.exists(p)])
        fig.text(0.5, 0.28, f'{section_count} analysis sections included',
                 fontsize=11, ha='center', va='top', color='#6b7280')

        fig.text(0.5, 0.08, 'Ember MD — Molecular Dynamics on Apple Silicon',
                 fontsize=10, ha='center', va='bottom', color='#9ca3af')

        pdf.savefig(fig)
        plt.close(fig)

def _load_python_pdf_backend() -> Optional[Any]:
    """Return a PDF reader/writer backend when one is available."""
    try:
        from pypdf import PdfReader, PdfWriter  # type: ignore
        return PdfReader, PdfWriter, 'pypdf'
    except ImportError:
        pass

    try:
        from PyPDF2 import PdfReader, PdfWriter  # type: ignore
        return PdfReader, PdfWriter, 'PyPDF2'
    except ImportError:
        return None


def _find_executable(command_name: str, env_var: Optional[str] = None) -> Optional[str]:
    """Resolve an executable even when the app is launched from Finder with a minimal PATH."""
    candidates: List[str] = []

    if env_var:
        configured = os.environ.get(env_var)
        if configured:
            candidates.append(configured)

    resolved = shutil.which(command_name)
    if resolved:
        return resolved

    candidates.extend([
        os.path.join('/opt/homebrew/bin', command_name),
        os.path.join('/usr/local/bin', command_name),
        os.path.join('/usr/bin', command_name),
    ])

    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def _merge_pdfs_with_python(report_path: str, input_pdfs: List[str]) -> bool:
    backend = _load_python_pdf_backend()
    if backend is None:
        return False

    PdfReader, PdfWriter, backend_name = backend
    writer = PdfWriter()

    try:
        for input_pdf in input_pdfs:
            reader = PdfReader(input_pdf)
            for page in reader.pages:
                writer.add_page(page)
        with open(report_path, 'wb') as handle:
            writer.write(handle)
        print(f"Merged report with {backend_name}")
        return True
    except Exception as e:
        print(f"Warning: Python PDF merge failed: {e}", file=sys.stderr)
        return False


def _merge_pdfs_with_pdfunite(report_path: str, input_pdfs: List[str]) -> bool:
    pdfunite = _find_executable('pdfunite', env_var='EMBER_PDFUNITE')
    if pdfunite is None:
        return False

    try:
        subprocess.run(
            [pdfunite, *input_pdfs, report_path],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"Merged report with {pdfunite}")
        return True
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or '').strip()
        print(f"Warning: pdfunite merge failed: {stderr or e}", file=sys.stderr)
        return False


def _merge_pdfs_with_ghostscript(report_path: str, input_pdfs: List[str]) -> bool:
    ghostscript = _find_executable('gs', env_var='EMBER_GS')
    if ghostscript is None:
        return False

    try:
        subprocess.run(
            [
                ghostscript,
                '-q',
                '-dBATCH',
                '-dNOPAUSE',
                '-sDEVICE=pdfwrite',
                '-dAutoRotatePages=/None',
                f'-sOutputFile={report_path}',
                *input_pdfs,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"Merged report with {ghostscript}")
        return True
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or '').strip()
        print(f"Warning: Ghostscript merge failed: {stderr or e}", file=sys.stderr)
        return False


def compile_pdf_simple(output_dir: str, section_pdfs: List[str], sim_info: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Compile report by prepending a title page and merging all section PDFs."""

    report_path = os.path.join(output_dir, 'full_report.pdf')
    existing_sections = [pdf for pdf in section_pdfs if os.path.exists(pdf)]

    title_pdf = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
    title_pdf.close()
    try:
        _write_title_page(title_pdf.name, existing_sections, sim_info)
        input_pdfs = [title_pdf.name, *existing_sections]

        if len(input_pdfs) == 1:
            shutil.copyfile(title_pdf.name, report_path)
            return report_path

        if os.path.exists(report_path):
            os.unlink(report_path)

        if _merge_pdfs_with_python(report_path, input_pdfs):
            return report_path
        if _merge_pdfs_with_pdfunite(report_path, input_pdfs):
            return report_path
        if _merge_pdfs_with_ghostscript(report_path, input_pdfs):
            return report_path

        print(
            'Warning: No PDF merge backend available. Install pypdf/PyPDF2, or make pdfunite/gs available.',
            file=sys.stderr,
        )
        return None
    finally:
        if os.path.exists(title_pdf.name):
            os.unlink(title_pdf.name)


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate comprehensive MD analysis report')
    parser.add_argument('--topology', required=True, help='Topology file (PDB)')
    parser.add_argument('--trajectory', required=True, help='Trajectory file (DCD)')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--ligand_selection', default=None, help='Ligand selection string')
    parser.add_argument('--ligand_sdf', default=None, help='Canonical ligand template SDF')
    parser.add_argument('--sim_info', default=None, help='JSON string with simulation metadata')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    python_exe = sys.executable

    # Parse simulation info
    sim_info = {}
    if args.sim_info:
        try:
            sim_info = json.loads(args.sim_info)
        except (json.JSONDecodeError, TypeError):
            pass

    section_pdfs = []
    contact_residues = []

    # Common args for topology/trajectory
    common_args = ['--topology', args.topology, '--trajectory', args.trajectory]
    lig_args = ['--ligand_selection', args.ligand_selection] if args.ligand_selection else []

    # ── 1. Contacts (provides data for RMSF green bars) ──
    print("PROGRESS:analyze_contacts:0")
    contacts_dir = os.path.join(args.output_dir, 'contacts')
    os.makedirs(contacts_dir, exist_ok=True)
    if run_analysis(python_exe, script_dir, 'analyze_contacts.py',
                    common_args + ['--output_dir', contacts_dir] + lig_args,
                    'Contact Analysis'):
        results_file = os.path.join(contacts_dir, 'contacts_results.json')
        if os.path.exists(results_file):
            with open(results_file) as f:
                contacts_data = json.load(f)
                contact_residues = contacts_data.get('contactResidues', [])
        for pdf_name in ['contacts_summary.pdf', 'contacts_timeline.pdf']:
            pdf_path = os.path.join(contacts_dir, pdf_name)
            if os.path.exists(pdf_path):
                section_pdfs.append(pdf_path)

    # ── 2. RMSD ──
    print("PROGRESS:analyze_rmsd:12")
    rmsd_dir = os.path.join(args.output_dir, 'rmsd')
    os.makedirs(rmsd_dir, exist_ok=True)
    if run_analysis(python_exe, script_dir, 'analyze_rmsd.py',
                    common_args + ['--output_dir', rmsd_dir, '--output_format', 'pdf'] + lig_args,
                    'RMSD Analysis'):
        pdf_path = os.path.join(rmsd_dir, 'rmsd.pdf')
        if os.path.exists(pdf_path):
            section_pdfs.append(pdf_path)

    # ── 3. RMSF (with contact residues) ──
    print("PROGRESS:analyze_rmsf:24")
    rmsf_dir = os.path.join(args.output_dir, 'rmsf')
    os.makedirs(rmsf_dir, exist_ok=True)
    rmsf_extra = ['--output_format', 'pdf']
    if contact_residues:
        rmsf_extra += ['--contact_residues', json.dumps(contact_residues)]
    if run_analysis(python_exe, script_dir, 'analyze_rmsf.py',
                    common_args + ['--output_dir', rmsf_dir] + rmsf_extra + lig_args,
                    'RMSF Analysis'):
        pdf_path = os.path.join(rmsf_dir, 'rmsf.pdf')
        if os.path.exists(pdf_path):
            section_pdfs.append(pdf_path)

    # ── 4. SSE ──
    print("PROGRESS:analyze_sse:36")
    sse_dir = os.path.join(args.output_dir, 'sse')
    os.makedirs(sse_dir, exist_ok=True)
    if run_analysis(python_exe, script_dir, 'analyze_sse.py',
                    common_args + ['--output_dir', sse_dir],
                    'Secondary Structure Analysis'):
        for pdf_name in ['sse_per_residue.pdf', 'sse_timeline.pdf']:
            pdf_path = os.path.join(sse_dir, pdf_name)
            if os.path.exists(pdf_path):
                section_pdfs.append(pdf_path)

    # ── 5. H-bonds ──
    print("PROGRESS:analyze_hbonds:48")
    hbonds_dir = os.path.join(args.output_dir, 'hbonds')
    os.makedirs(hbonds_dir, exist_ok=True)
    hbonds_args = common_args + ['--output_dir', hbonds_dir] + lig_args
    if run_analysis(python_exe, script_dir, 'analyze_hbonds.py',
                    hbonds_args,
                    'H-bond Analysis'):
        # analyze_hbonds.py outputs PNG — generate PDF version
        plot_png = os.path.join(hbonds_dir, 'hbonds_plot.png')
        if os.path.exists(plot_png):
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                import matplotlib.image as mpimg
                img = mpimg.imread(plot_png)
                fig = plt.figure(figsize=(10, max(6, img.shape[0] / img.shape[1] * 10)))
                ax = fig.add_axes([0, 0, 1, 1])
                ax.imshow(img)
                ax.axis('off')
                pdf_path = os.path.join(hbonds_dir, 'hbonds.pdf')
                plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
                plt.close()
                section_pdfs.append(pdf_path)
            except Exception as e:
                print(f"  Warning: Could not convert hbonds plot to PDF: {e}", file=sys.stderr)

    # ── 6. Ligand Properties ──
    print("PROGRESS:analyze_ligand_props:60")
    ligand_props_dir = os.path.join(args.output_dir, 'ligand_props')
    os.makedirs(ligand_props_dir, exist_ok=True)
    lp_args = common_args + ['--output_dir', ligand_props_dir] + lig_args
    rmsd_csv = os.path.join(rmsd_dir, 'rmsd_data.csv')
    if os.path.exists(rmsd_csv):
        lp_args += ['--rmsd_csv', rmsd_csv]
    if run_analysis(python_exe, script_dir, 'analyze_ligand_props.py',
                    lp_args,
                    'Ligand Properties Analysis'):
        pdf_path = os.path.join(ligand_props_dir, 'ligand_props.pdf')
        if os.path.exists(pdf_path):
            section_pdfs.append(pdf_path)

    # ── 7. Torsions ──
    print("PROGRESS:analyze_torsions:72")
    torsions_dir = os.path.join(args.output_dir, 'torsions')
    os.makedirs(torsions_dir, exist_ok=True)
    torsion_args = common_args + ['--output_dir', torsions_dir] + lig_args
    if args.ligand_sdf:
        torsion_args += ['--ligand_sdf', args.ligand_sdf]
    if run_analysis(python_exe, script_dir, 'analyze_torsions.py',
                    torsion_args,
                    'Torsion Analysis'):
        pdf_path = os.path.join(torsions_dir, 'torsions.pdf')
        if os.path.exists(pdf_path):
            section_pdfs.append(pdf_path)

    # ── 8. Clustering ──
    print("PROGRESS:clustering:84")
    clustering_dir = os.path.join(args.output_dir, 'clustering')
    os.makedirs(clustering_dir, exist_ok=True)
    clustering_results_path = os.path.join(clustering_dir, 'clustering_results.json')
    if os.path.exists(clustering_results_path):
        print(f"Reusing existing clustering: {clustering_results_path}")
    else:
        cluster_args = [
            '--topology', args.topology,
            '--trajectory', args.trajectory,
            '--output_dir', clustering_dir,
            '--n_clusters', '10',
            '--method', 'kmeans',
            '--selection', 'ligand',
            '--strip_waters',
        ]
        run_analysis(python_exe, script_dir, 'cluster_trajectory.py',
                     cluster_args, 'Clustering')

    if args.ligand_sdf and os.path.exists(clustering_results_path):
        cluster_torsion_args = [
            '--torsions_json', os.path.join(torsions_dir, 'torsions_results.json'),
            '--clustering_dir', clustering_dir,
            '--ligand_sdf', args.ligand_sdf,
        ]
        scored_clusters_dir = os.path.join(args.output_dir, 'scored_clusters')
        if os.path.exists(scored_clusters_dir):
            cluster_torsion_args += ['--scored_clusters_dir', scored_clusters_dir]
        run_analysis(
            python_exe,
            script_dir,
            'analyze_cluster_torsions.py',
            cluster_torsion_args,
            'Cluster Torsion Analysis',
        )

    # ── 9. Compile PDF ──
    print("PROGRESS:compile_pdf:92")
    print("\n=== Compiling Full Report ===")
    report_path = compile_pdf_simple(args.output_dir, section_pdfs, sim_info)

    if report_path and os.path.exists(report_path):
        print(f"\nFull report saved to: {report_path}")
        print(f"Section PDFs: {len(section_pdfs)}")
        for sp in section_pdfs:
            print(f"  {os.path.relpath(sp, args.output_dir)}")
    else:
        print("Warning: Could not compile full report PDF", file=sys.stderr)
        raise SystemExit(1)

    print("PROGRESS:done:100")
    print("\nDone!")


if __name__ == '__main__':
    main()
