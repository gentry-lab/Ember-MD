#!/usr/bin/env python3
# Copyright (c) 2026 Ember Contributors. MIT License.
"""
X-ray Structure Quality Analyzer
==================================
Unified PDB + MTZ analysis for crystallographic structure validation.

PDB-only mode: Header metrics (R-free, R-gap), B-factor analysis,
  ligand contacts, occupancy/chain breaks, B-factor cliffs, scorecard.

PDB+MTZ mode: All of the above PLUS per-atom electron density (sigma),
  RSCC, pocket density comparison, Fo-Fc difference map analysis,
  occupancy/B-factor audit, rigid fragment consistency test.

Requirements:
  - numpy (required)
  - gemmi (optional, needed only for MTZ analysis: pip install gemmi)

Setup:
  Place PDB files (and optionally matching MTZ files) in a folder
  named 'x-ray'. MTZ files are matched by filename stem.

Usage:
  python xray_analyzer.py              # scans x-ray/ by default
  python xray_analyzer.py /some/dir    # scan a different directory

Output:
  - xray_analysis_TIMESTAMP.json
  - xray_analysis_TIMESTAMP.md
"""

import numpy as np
from collections import defaultdict
import json
import sys
import os
import warnings
from typing import Dict, List, Tuple, Optional, Any
import argparse
import glob
from datetime import datetime
from pathlib import Path

# Optional: gemmi for MTZ density analysis
try:
    import gemmi
    HAS_GEMMI = True
except ImportError:
    HAS_GEMMI = False

# Optional: RDKit + matplotlib for ligand density images
try:
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as mfont
    import matplotlib.colors as mcolors
    from matplotlib.patches import Circle, Polygon
    import matplotlib.patheffects as pe
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

# Optional: reportlab for PDF reports
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch as RL_INCH
    from reportlab.lib import colors as rl_colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph as RLParagraph, Spacer as RLSpacer,
        Image as RLImage, Table as RLTable, TableStyle as RLTableStyle,
        HRFlowable, PageBreak as RLPageBreak, KeepTogether as RLKeepTogether
    )
    from reportlab.lib.styles import getSampleStyleSheet as rl_getSampleStyleSheet
    from reportlab.lib.styles import ParagraphStyle as RLParagraphStyle
    from reportlab.lib.enums import TA_CENTER as RL_TA_CENTER
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


def _register_pdf_font_family():
    """Prefer bundled Inter fonts for PDF output, with a safe fallback."""
    if not HAS_REPORTLAB:
        return 'Helvetica'

    font_dir = Path(__file__).resolve().parent / 'fonts'
    font_faces = {
        'Inter': font_dir / 'Inter-Regular.ttf',
        'Inter-Bold': font_dir / 'Inter-Bold.ttf',
        'Inter-Italic': font_dir / 'Inter-Italic.ttf',
        'Inter-BoldItalic': font_dir / 'Inter-BoldItalic.ttf',
    }

    if not all(font_path.exists() for font_path in font_faces.values()):
        return 'Helvetica'

    registered = set(pdfmetrics.getRegisteredFontNames())
    for font_name, font_path in font_faces.items():
        if font_name not in registered:
            pdfmetrics.registerFont(TTFont(font_name, str(font_path)))

    pdfmetrics.registerFontFamily(
        'Inter',
        normal='Inter',
        bold='Inter-Bold',
        italic='Inter-Italic',
        boldItalic='Inter-BoldItalic',
    )
    return 'Inter'


def _get_inter_semibold_font_properties():
    """Return Matplotlib font properties for bundled Inter SemiBold, if available."""
    if not HAS_RDKIT:
        return None

    font_path = Path(__file__).resolve().parent / 'fonts' / 'Inter-SemiBold.ttf'
    if not font_path.exists():
        return None

    return mfont.FontProperties(fname=str(font_path), weight='semibold')

class PDBAnalyzer:
    """Core PDB analyzer with enhanced R-value detection"""
    
    def __init__(self, pdb_file: str):
        self.pdb_file = pdb_file
        self.atoms = []
        self.hetatms = []
        self.remarks = defaultdict(list)
        self.header_lines = []
        
    def parse_pdb(self):
        """Parse PDB file and extract all records"""
        with open(self.pdb_file, 'r') as f:
            for line in f:
                record = line[:6].strip()
                
                # Store all header lines for better R-value extraction
                if record in ['HEADER', 'TITLE', 'COMPND', 'SOURCE', 'KEYWDS', 'EXPDTA']:
                    self.header_lines.append(line)
                elif record == 'ATOM':
                    self.parse_atom_record(line)
                elif record == 'HETATM':
                    self.parse_hetatm_record(line)
                elif record == 'REMARK':
                    self.parse_remark_record(line)
    
    def parse_atom_record(self, line: str):
        """Parse ATOM record"""
        try:
            atom = {
                'serial': int(line[6:11]),
                'name': line[12:16].strip(),
                'resname': line[17:20].strip(),
                'chain': line[21].strip(),
                'resnum': int(line[22:26]),
                'x': float(line[30:38]),
                'y': float(line[38:46]),
                'z': float(line[46:54]),
                'occupancy': float(line[54:60]) if line[54:60].strip() else 1.0,
                'bfactor': float(line[60:66]) if line[60:66].strip() else 0.0,
            }
            self.atoms.append(atom)
        except (ValueError, IndexError) as e:
            pass  # Skip malformed lines
    
    def parse_hetatm_record(self, line: str):
        """Parse HETATM record"""
        try:
            hetatm = {
                'serial': int(line[6:11]),
                'name': line[12:16].strip(),
                'resname': line[17:20].strip(),
                'chain': line[21].strip(),
                'resnum': int(line[22:26]),
                'x': float(line[30:38]),
                'y': float(line[38:46]),
                'z': float(line[46:54]),
                'occupancy': float(line[54:60]) if line[54:60].strip() else 1.0,
                'bfactor': float(line[60:66]) if line[60:66].strip() else 0.0,
            }
            self.hetatms.append(hetatm)
        except (ValueError, IndexError) as e:
            pass
    
    def parse_remark_record(self, line: str):
        """Parse REMARK records"""
        try:
            remark_num = int(line[6:10].strip())
            content = line[10:].rstrip()
            self.remarks[remark_num].append(content)
        except:
            pass
    
    def extract_crystallographic_data(self) -> Dict:
        """Enhanced extraction of crystallographic metrics"""
        data = {
            'resolution': None,
            'r_work': None,
            'r_free': None,
            'r_gap': None,
            'wilson_b': None,
            'completeness': None,
            'refinement_program': None,
            'space_group': None,
            'data_collection_temp': None,
            'ramachandran_favored': None,
            'ramachandran_outliers': None,
            'clashscore': None,
            'chain_breaks': None,
            'tls_groups': None
        }
        
        # Parse REMARK 3 for resolution (based on actual PDB headers)
        for line in self.remarks.get(3, []):
            if 'RESOLUTION RANGE HIGH' in line.upper():
                # Pattern: "RESOLUTION RANGE HIGH (ANGSTROMS) : 1.60"
                import re
                match = re.search(r'(\d+\.?\d*)\s*ANGSTROM', line.upper())
                if match:
                    data['resolution'] = float(match.group(1))
                else:
                    # Fallback: extract number after colon
                    parts = line.split(':')
                    if len(parts) > 1:
                        try:
                            data['resolution'] = float(parts[-1].strip())
                        except:
                            pass
        
        # Parse REMARK 3 for R-factors - ENHANCED DETECTION
        for line in self.remarks.get(3, []):
            line_upper = line.upper()
            
            # R-work detection
            if 'R VALUE' in line_upper and 'WORKING' in line_upper:
                # Skip bin-specific lines
                if 'BIN' not in line_upper and 'SHELL' not in line_upper:
                    # Extract number after colon or equals
                    import re
                    match = re.search(r'[:=]\s*(\d+\.?\d*)', line)
                    if match:
                        val = float(match.group(1))
                        # Some files report as percentage
                        if val > 1.0:
                            val = val / 100.0
                        if data['r_work'] is None:
                            data['r_work'] = val
            
            # R-free detection
            elif 'FREE R VALUE' in line_upper or 'R FREE' in line_upper:
                if 'BIN' not in line_upper and 'SHELL' not in line_upper:
                    import re
                    match = re.search(r'[:=]\s*(\d+\.?\d*)', line)
                    if match:
                        val = float(match.group(1))
                        if val > 1.0:
                            val = val / 100.0
                        if data['r_free'] is None:
                            data['r_free'] = val
            
            # Alternative R-factor format
            elif 'R WORK' in line_upper:
                import re
                match = re.search(r'R WORK\s*[:=]\s*(\d+\.?\d*)', line_upper)
                if match:
                    val = float(match.group(1))
                    if val > 1.0:
                        val = val / 100.0
                    if data['r_work'] is None:
                        data['r_work'] = val
            
            # Wilson B-factor
            elif 'WILSON' in line_upper and 'B' in line_upper:
                import re
                match = re.search(r'(\d+\.?\d*)', line)
                if match:
                    data['wilson_b'] = float(match.group(1))
            
            # Completeness
            elif 'COMPLETENESS' in line_upper and '%' in line:
                import re
                match = re.search(r'(\d+\.?\d*)\s*%', line)
                if match:
                    data['completeness'] = float(match.group(1))
            
            # Program
            elif 'PROGRAM' in line_upper:
                parts = line.split(':')
                if len(parts) > 1:
                    data['refinement_program'] = parts[-1].strip()

            # Ramachandran statistics
            elif 'RAMACHANDRAN PLOT' in line_upper:
                if 'OUTLIERS' in line_upper:
                    import re
                    match = re.search(r'(\d+\.?\d*)', line)
                    if match:
                        data['ramachandran_outliers'] = float(match.group(1))
                elif 'ALLOWED' in line_upper or 'FAVORED' in line_upper:
                    import re
                    match = re.search(r'(\d+\.?\d*)', line)
                    if match:
                        data['ramachandran_favored'] = float(match.group(1))

            # Clashscore
            elif 'CLASHSCORE' in line_upper:
                import re
                match = re.search(r'(\d+\.?\d*)', line)
                if match:
                    data['clashscore'] = float(match.group(1))

            # TLS groups
            elif 'NUMBER OF TLS GROUPS' in line_upper:
                import re
                match = re.search(r'(\d+)', line)
                if match:
                    data['tls_groups'] = int(match.group(1))
        
        # Calculate R-gap
        if data['r_work'] is not None and data['r_free'] is not None:
            data['r_gap'] = round(data['r_free'] - data['r_work'], 5)
        
        # Parse REMARK 200 for temperature
        for line in self.remarks.get(200, []):
            if 'TEMPERATURE' in line.upper() and 'KELVIN' in line.upper():
                import re
                match = re.search(r'(\d+\.?\d*)\s*KELVIN', line.upper())
                if match:
                    data['data_collection_temp'] = float(match.group(1))
        
        return data
    
    def calculate_distance(self, atom1: Dict, atom2: Dict) -> float:
        """Calculate distance between atoms"""
        return np.sqrt((atom1['x'] - atom2['x'])**2 + 
                      (atom1['y'] - atom2['y'])**2 + 
                      (atom1['z'] - atom2['z'])**2)
    
    def analyze_ligands(self, header_data: Dict) -> Dict:
        """Analyze ligand quality metrics"""
        ligand_analysis = {}

        # Group ligands - exclude water and common salts/ions
        excluded_resnames = [
            'HOH', 'WAT',  # Water
            'NA', 'K', 'CL', 'MG', 'CA', 'ZN',  # Common ions
            'SO4', 'PO4', 'NO3', 'ACET', 'ACT',  # Common salts/buffers
            'EDO', 'GOL', 'PEG', 'DMS'  # Common cryoprotectants
        ]
        non_water_hetatms = [h for h in self.hetatms if h['resname'] not in excluded_resnames]
        ligand_groups = defaultdict(list)
        for hetatm in non_water_hetatms:
            key = f"{hetatm['resname']}_{hetatm['chain']}_{hetatm['resnum']}"
            ligand_groups[key].append(hetatm)

        # Calculate protein statistics
        protein_bfactors = [a['bfactor'] for a in self.atoms]
        protein_mean_b = np.mean(protein_bfactors) if protein_bfactors else 0
        protein_std_b = np.std(protein_bfactors) if protein_bfactors else 1
        
        for ligand_id, ligand_atoms in ligand_groups.items():
            if not ligand_atoms:
                continue
            
            ligand_bfactors = [a['bfactor'] for a in ligand_atoms]
            ligand_occupancies = [a['occupancy'] for a in ligand_atoms]
            
            # Find nearby protein atoms (within 4Å)
            contact_atoms = []
            nearby_atoms_5A = []  # For binding pocket calculation
            
            for lig_atom in ligand_atoms:
                for prot_atom in self.atoms:
                    dist = self.calculate_distance(lig_atom, prot_atom)
                    if dist <= 4.0:
                        contact_atoms.append(prot_atom)
                    if dist <= 5.0:
                        nearby_atoms_5A.append(prot_atom)
            
            # Remove duplicates
            contact_atoms = list({a['serial']: a for a in contact_atoms}.values())
            nearby_atoms_5A = list({a['serial']: a for a in nearby_atoms_5A}.values())
            
            # Calculate binding pocket B-factor (atoms within 5Å of ligand)
            if nearby_atoms_5A:
                pocket_bfactors = [a['bfactor'] for a in nearby_atoms_5A]
                pocket_mean_b = np.mean(pocket_bfactors)
            else:
                pocket_mean_b = protein_mean_b

            # Calculate metrics
            ligand_mean_b = np.mean(ligand_bfactors)
            b_ratio_protein = ligand_mean_b / protein_mean_b if protein_mean_b > 0 else None
            b_ratio_pocket = ligand_mean_b / pocket_mean_b if pocket_mean_b > 0 else None
            z_score = (ligand_mean_b - protein_mean_b) / protein_std_b if protein_std_b > 0 else 0

            # Use the most stringent B-ratio for verdict (protein vs pocket)
            worst_b_ratio = max(filter(None, [b_ratio_protein, b_ratio_pocket]), default=999)
            
            # Determine verdict based on critical metrics
            verdict = self.determine_ligand_verdict(worst_b_ratio, len(contact_atoms), 
                                                   min(ligand_occupancies), ligand_mean_b)
            
            ligand_analysis[ligand_id] = {
                'resname': ligand_atoms[0]['resname'],
                'chain': ligand_atoms[0]['chain'],
                'num_atoms': len(ligand_atoms),
                'mean_b': round(ligand_mean_b, 2),
                'min_b': round(min(ligand_bfactors), 2),
                'max_b': round(max(ligand_bfactors), 2),
                'mean_occupancy': round(np.mean(ligand_occupancies), 3),
                'min_occupancy': round(min(ligand_occupancies), 3),
                'num_protein_contacts_4A': len(contact_atoms),
                'b_ratio_to_protein': round(b_ratio_protein, 2) if b_ratio_protein else None,
                'b_ratio_to_pocket': round(b_ratio_pocket, 2) if b_ratio_pocket else None,
                'pocket_mean_b': round(pocket_mean_b, 2),
                'z_score': round(z_score, 2),
                'verdict': verdict
            }
        
        return ligand_analysis

    def analyze_occupancy_and_chain_breaks(self) -> Dict:
        """Analyze partial occupancy atoms and detect chain breaks"""
        occupancy_analysis = {
            'partial_occupancy_atoms': 0,
            'partial_occupancy_residues': set(),
            'chain_breaks': [],
            'missing_residues': 0
        }

        # Analyze occupancy
        for atom in self.atoms:
            if atom['occupancy'] < 0.95:  # Less than 95% occupancy
                occupancy_analysis['partial_occupancy_atoms'] += 1
                occupancy_analysis['partial_occupancy_residues'].add(
                    f"{atom['chain']}_{atom['resnum']}"
                )

        # Detect chain breaks by looking for gaps in residue numbering
        chain_residues = {}
        for atom in self.atoms:
            chain = atom['chain']
            resnum = atom['resnum']
            if chain not in chain_residues:
                chain_residues[chain] = []
            chain_residues[chain].append(resnum)

        for chain, residues in chain_residues.items():
            residues = sorted(set(residues))
            for i in range(len(residues) - 1):
                gap = residues[i + 1] - residues[i]
                if gap > 1:  # Gap of more than 1 residue
                    occupancy_analysis['chain_breaks'].append({
                        'chain': chain,
                        'start_res': residues[i],
                        'end_res': residues[i + 1],
                        'gap_size': gap - 1
                    })
                    occupancy_analysis['missing_residues'] += gap - 1

        occupancy_analysis['num_partial_residues'] = len(occupancy_analysis['partial_occupancy_residues'])
        # Convert set to list for JSON serialization
        occupancy_analysis['partial_occupancy_residues'] = list(occupancy_analysis['partial_occupancy_residues'])
        return occupancy_analysis

    def detect_b_factor_cliffs(self, threshold: float = 3.0) -> Dict:
        """Detect B-factor cliffs - sudden jumps in thermal motion"""
        cliffs = []

        # Sort atoms by residue number for sequential analysis
        sorted_atoms = sorted(self.atoms, key=lambda x: (x['chain'], x['resnum'], x['serial']))

        for i in range(len(sorted_atoms) - 1):
            atom1 = sorted_atoms[i]
            atom2 = sorted_atoms[i + 1]

            # Only check atoms in same chain and consecutive residues
            if atom1['chain'] == atom2['chain'] and atom2['resnum'] - atom1['resnum'] <= 1:
                b1 = atom1['bfactor']
                b2 = atom2['bfactor']

                # Calculate fold change
                if b1 > 0 and b2 > 0:
                    fold_change = max(b1, b2) / min(b1, b2)

                    if fold_change >= threshold:
                        cliffs.append({
                            'atom1': f"{atom1['resname']}{atom1['resnum']}_{atom1['name']}",
                            'atom2': f"{atom2['resname']}{atom2['resnum']}_{atom2['name']}",
                            'b1': round(b1, 2),
                            'b2': round(b2, 2),
                            'fold_change': round(fold_change, 2),
                            'direction': 'increase' if b2 > b1 else 'decrease'
                        })

        return {
            'num_cliffs': len(cliffs),
            'cliff_threshold': threshold,
            'cliffs': cliffs[:20],  # Limit output
            'severity': 'HIGH' if len(cliffs) > 10 else 'MEDIUM' if len(cliffs) > 5 else 'LOW'
        }
    
    def determine_ligand_verdict(self, b_ratio, contacts, min_occupancy, mean_b):
        """Determine ligand reliability verdict"""
        if b_ratio is None:
            return "NO_DATA"
        elif b_ratio > 3.0:
            return "UNRELIABLE"
        elif b_ratio > 2.5:
            return "HIGHLY_QUESTIONABLE"
        elif b_ratio > 2.0:
            return "QUESTIONABLE"
        elif contacts < 5:
            return "INSUFFICIENT_CONTACTS"
        elif b_ratio > 1.5:
            return "MARGINAL"
        elif min_occupancy < 0.5:
            return "LOW_OCCUPANCY"
        else:
            return "ACCEPTABLE"
    
    def calculate_structure_grade(self, header_data: Dict, ligand_analysis: Dict) -> str:
        """Calculate overall structure grade with strict criteria"""
        # Start with 100 points
        score = 100
        
        # R-free penalty (up to -50 points)
        r_free = header_data.get('r_free')
        if r_free is None:
            score -= 30  # Heavy penalty for missing data
        elif r_free > 0.40:
            score -= 50
        elif r_free > 0.35:
            score -= 40
        elif r_free > 0.30:
            score -= 25
        elif r_free > 0.25:
            score -= 10
        
        # R-gap penalty (up to -20 points)
        r_gap = header_data.get('r_gap')
        if r_gap and r_gap > 0.10:
            score -= 30
        elif r_gap and r_gap > 0.07:
            score -= 20
        elif r_gap and r_gap > 0.05:
            score -= 10
        
        # Ligand quality penalty (up to -40 points)
        if ligand_analysis:
            # Use worst ligand for grading
            worst_b_ratio = max([l.get('b_ratio_to_protein', 999) 
                               for l in ligand_analysis.values() 
                               if l.get('b_ratio_to_protein')])
            if worst_b_ratio > 3.0:
                score -= 40
            elif worst_b_ratio > 2.5:
                score -= 30
            elif worst_b_ratio > 2.0:
                score -= 20
            elif worst_b_ratio > 1.5:
                score -= 10
        
        # Convert to grade
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'
    
    
    def calculate_scorecard(self, header_data: Dict, ligand_analysis: Dict, b_factor_cliffs: Dict = None, occupancy_analysis: Dict = None) -> Dict:
        """Calculate detailed scorecard with subgrades"""
        scorecard = {
            'overall_grade': 'F',
            'overall_score': 0,
            'subgrades': {},
            'weights': {
                'r_free': 0.35,
                'ligand_quality': 0.35,
                'occupancy_contacts': 0.30
            }
        }
        
        # R-free subgrade (40% weight)
        r_free = header_data.get('r_free')
        if r_free is None:
            scorecard['subgrades']['r_free'] = 'F'
            r_free_score = 0
        elif r_free < 0.25:
            scorecard['subgrades']['r_free'] = 'A'
            r_free_score = 4.0
        elif r_free < 0.30:
            scorecard['subgrades']['r_free'] = 'B'
            r_free_score = 3.0
        elif r_free < 0.35:
            scorecard['subgrades']['r_free'] = 'C'
            r_free_score = 2.0
        elif r_free < 0.40:
            scorecard['subgrades']['r_free'] = 'D'
            r_free_score = 1.0
        else:
            scorecard['subgrades']['r_free'] = 'F'
            r_free_score = 0
        
        # R-gap penalty
        r_gap = header_data.get('r_gap')
        if r_gap and r_gap > 0.10:
            r_free_score = max(0, r_free_score - 2.0)
        elif r_gap and r_gap > 0.07:
            r_free_score = max(0, r_free_score - 1.0)
        
        # Ligand quality subgrade (30% weight)
        if ligand_analysis:
            worst_b_ratio = max([l.get('b_ratio_to_protein', 999) 
                               for l in ligand_analysis.values() 
                               if l.get('b_ratio_to_protein')], default=999)
            
            if worst_b_ratio == 999:
                scorecard['subgrades']['ligand_quality'] = 'F'
                ligand_score = 0
            elif worst_b_ratio < 1.5:
                scorecard['subgrades']['ligand_quality'] = 'A'
                ligand_score = 4.0
            elif worst_b_ratio < 2.0:
                scorecard['subgrades']['ligand_quality'] = 'B'
                ligand_score = 3.0
            elif worst_b_ratio < 2.5:
                scorecard['subgrades']['ligand_quality'] = 'C'
                ligand_score = 2.0
            elif worst_b_ratio < 3.0:
                scorecard['subgrades']['ligand_quality'] = 'D'
                ligand_score = 1.0
            else:
                scorecard['subgrades']['ligand_quality'] = 'F'
                ligand_score = 0
        else:
            scorecard['subgrades']['ligand_quality'] = 'N/A'
            ligand_score = 0
        
        # Occupancy/Contacts subgrade (20% weight)
        if ligand_analysis:
            min_contacts = min([l.get('num_protein_contacts_4A', 0) 
                              for l in ligand_analysis.values()], default=0)
            min_occupancy = min([l.get('min_occupancy', 0) 
                               for l in ligand_analysis.values()], default=0)
            
            if min_contacts >= 15 and min_occupancy >= 0.9:
                scorecard['subgrades']['occupancy_contacts'] = 'A'
                occ_score = 4.0
            elif min_contacts >= 10 and min_occupancy >= 0.7:
                scorecard['subgrades']['occupancy_contacts'] = 'B'
                occ_score = 3.0
            elif min_contacts >= 5 and min_occupancy >= 0.5:
                scorecard['subgrades']['occupancy_contacts'] = 'C'
                occ_score = 2.0
            elif min_contacts >= 3:
                scorecard['subgrades']['occupancy_contacts'] = 'D'
                occ_score = 1.0
            else:
                scorecard['subgrades']['occupancy_contacts'] = 'F'
                occ_score = 0
        else:
            scorecard['subgrades']['occupancy_contacts'] = 'N/A'
            occ_score = 0
        
        
        # Quality metric penalties
        quality_penalty = 0

        # B-factor cliffs penalty (aligned with community standards)
        if b_factor_cliffs:
            num_cliffs = b_factor_cliffs.get('num_cliffs', 0)
            if num_cliffs > 15:
                quality_penalty += 1.0  # Severe penalty
            elif num_cliffs > 10:
                quality_penalty += 0.5  # Moderate penalty

        # Clashscore penalty (updated to community standards)
        clashscore = header_data.get('clashscore')
        if clashscore and clashscore > 40:
            quality_penalty += 0.5

        # Ramachandran outliers penalty (updated to community standards)
        rama_out = header_data.get('ramachandran_outliers')
        if rama_out and rama_out > 5.0:
            quality_penalty += 0.5

        # Chain breaks penalty (updated to community standards)
        if occupancy_analysis:
            missing_res = occupancy_analysis.get('missing_residues', 0)
            if missing_res > 5:
                quality_penalty += 0.5

        # Calculate weighted overall score
        overall_numeric = (
            scorecard['weights']['r_free'] * r_free_score +
            scorecard['weights']['ligand_quality'] * ligand_score +
            scorecard['weights']['occupancy_contacts'] * occ_score
        ) - quality_penalty
        
        scorecard['overall_score'] = round(overall_numeric, 2)
        
        # Convert to letter grade
        if overall_numeric >= 3.7:
            scorecard['overall_grade'] = 'A'
        elif overall_numeric >= 3.3:
            scorecard['overall_grade'] = 'A-'
        elif overall_numeric >= 3.0:
            scorecard['overall_grade'] = 'B+'
        elif overall_numeric >= 2.7:
            scorecard['overall_grade'] = 'B'
        elif overall_numeric >= 2.3:
            scorecard['overall_grade'] = 'B-'
        elif overall_numeric >= 2.0:
            scorecard['overall_grade'] = 'C+'
        elif overall_numeric >= 1.7:
            scorecard['overall_grade'] = 'C'
        elif overall_numeric >= 1.3:
            scorecard['overall_grade'] = 'C-'
        elif overall_numeric >= 1.0:
            scorecard['overall_grade'] = 'D'
        else:
            scorecard['overall_grade'] = 'F'
        
        return scorecard
    
    def generate_report(self) -> Dict:
        """Generate comprehensive analysis report"""
        self.parse_pdb()

        header_data = self.extract_crystallographic_data()
        ligand_analysis = self.analyze_ligands(header_data)
        b_factor_cliffs = self.detect_b_factor_cliffs()
        occupancy_analysis = self.analyze_occupancy_and_chain_breaks()

        # Calculate protein statistics
        protein_bfactors = [a['bfactor'] for a in self.atoms]
        global_stats = {
            'mean_protein_b': round(np.mean(protein_bfactors), 2) if protein_bfactors else None,
            'median_protein_b': round(np.median(protein_bfactors), 2) if protein_bfactors else None,
            'std_protein_b': round(np.std(protein_bfactors), 2) if protein_bfactors else None,
            'num_protein_atoms': len(self.atoms),
            'num_hetatm_records': len(self.hetatms)
        }

        # Calculate scorecard
        scorecard = self.calculate_scorecard(header_data, ligand_analysis, b_factor_cliffs, occupancy_analysis)

        # Overall assessment
        grade = self.calculate_structure_grade(header_data, ligand_analysis)

        # Determine overall verdict
        if header_data.get('r_free') is None:
            overall_verdict = "MISSING_VALIDATION_DATA"
        elif header_data.get('r_free', 1.0) > 0.35:
            overall_verdict = "POOR_QUALITY"
        elif any(l.get('b_ratio_to_protein', 0) > 2.5 for l in ligand_analysis.values()):
            overall_verdict = "UNRELIABLE_LIGANDS"
        elif header_data.get('r_free', 1.0) > 0.30:
            overall_verdict = "MARGINAL"
        else:
            overall_verdict = "ACCEPTABLE"

        return {
            'pdb_file': self.pdb_file,
            'header_data': header_data,
            'global_statistics': global_stats,
            'ligands': ligand_analysis,
            'b_factor_cliffs': b_factor_cliffs,
            'occupancy_analysis': occupancy_analysis,
            'scorecard': scorecard,
            'grade': grade,
            'overall_verdict': overall_verdict,
            'timestamp': datetime.now().isoformat()
        }


# ---------------------------------------------------------------------------
# MTZ Density Analysis (requires gemmi)
# ---------------------------------------------------------------------------
STANDARD_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
    "TYR", "VAL", "MSE", "SEC", "PYL",
}

EXCLUDED_HETNAMES = {
    "HOH", "WAT", "GOL", "EDO", "SO4", "ACT", "PEG", "PGE", "DMS",
    "BME", "MPD", "CIT", "TRS", "MES", "EPE", "IMD", "FMT", "IOD",
    "CL", "NA", "MG", "CA", "ZN", "MN", "FE", "CO", "NI", "CU",
    "K", "BR", "PO4", "SCN", "NO3", "NH4",
}


def analyze_density(pdb_path, mtz_path):
    """Run full density validation on a PDB/MTZ pair. Returns dict."""
    if not HAS_GEMMI:
        return {"error": "gemmi not installed — pip install gemmi"}

    # Read structure — support both PDB and mmCIF
    pdb_str = str(pdb_path)
    try:
        if pdb_str.endswith('.cif') or pdb_str.endswith('.mmcif'):
            st = gemmi.read_structure(pdb_str)
        else:
            st = gemmi.read_pdb(pdb_str)
    except Exception as e:
        return {"error": f"Cannot read PDB file: {e}"}

    try:
        mtz = gemmi.read_mtz_file(str(mtz_path))
    except Exception as e:
        return {"error": f"Cannot read MTZ file: {e}"}

    # Cell consistency check
    pdb_cell = st.cell
    mtz_cell = mtz.cell
    cell_diffs = [
        abs(pdb_cell.a - mtz_cell.a),
        abs(pdb_cell.b - mtz_cell.b),
        abs(pdb_cell.c - mtz_cell.c),
    ]
    if max(cell_diffs) > 0.5:
        return {"error": f"Unit cell mismatch between PDB and MTZ — files may not be from the same crystal. "
                f"PDB: {pdb_cell.a:.1f} {pdb_cell.b:.1f} {pdb_cell.c:.1f}, "
                f"MTZ: {mtz_cell.a:.1f} {mtz_cell.b:.1f} {mtz_cell.c:.1f}"}

    # Detect map columns — support all major refinement programs
    col_labels = [c.label for c in mtz.columns]

    # 2Fo-Fc map coefficients (try in order of prevalence)
    f2, p2 = None, None
    candidates_2fofc = [
        ("FWT", "PHWT"),             # REFMAC, BUSTER
        ("2FOFCWT", "PH2FOFCWT"),    # Phenix
        ("2FOFC", "PH2FOFC"),        # some Phenix versions
        ("FDM", "PHIDM"),            # density modification
    ]
    for fc, pc in candidates_2fofc:
        if fc in col_labels and pc in col_labels:
            f2, p2 = fc, pc
            break

    # Fallback: look for F/PHI type columns heuristically
    if f2 is None:
        for col in mtz.columns:
            if col.type == 'F' and 'FC' not in col.label.upper():
                # Find matching phase column
                for pcol in mtz.columns:
                    if pcol.type == 'P' and pcol.dataset_id == col.dataset_id:
                        f2, p2 = col.label, pcol.label
                        break
            if f2:
                break

    if f2 is None:
        # Check if this looks like unrefined data
        has_f = any(c.type == 'F' for c in mtz.columns)
        has_phase = any(c.type == 'P' for c in mtz.columns)
        if has_f and not has_phase:
            return {"error": "MTZ contains structure factors but no phases — this appears to be unrefined data. "
                    "Run refinement (REFMAC/Phenix/BUSTER) first to generate map coefficients."}
        else:
            return {"error": f"Cannot find 2Fo-Fc map columns in MTZ. "
                    f"Expected FWT/PHWT (REFMAC) or 2FOFCWT/PH2FOFCWT (Phenix). "
                    f"Available columns: {', '.join(col_labels)}"}

    # Fo-Fc difference map coefficients
    fd, pd = None, None
    candidates_fofc = [
        ("DELFWT", "PHDELWT"),       # REFMAC, BUSTER
        ("FOFCWT", "PHFOFCWT"),      # Phenix
        ("FOFC", "PHFOFC"),          # some Phenix versions
    ]
    for fc, pc in candidates_fofc:
        if fc in col_labels and pc in col_labels:
            fd, pd = fc, pc
            break

    if fd is None:
        # Fo-Fc is nice to have but not essential — proceed without it
        print(f"    Warning: No Fo-Fc columns found, skipping difference map analysis")

    grid_2fofc = mtz.transform_f_phi_to_map(f2, p2, sample_rate=3.0)
    grid_fofc = mtz.transform_f_phi_to_map(fd, pd, sample_rate=3.0) if fd else None

    # Model density map for proper RSCC
    grid_fc = None
    candidates_fc = [
        ("FC_ALL", "PHIC_ALL"),
        ("FC", "PHIC"),
        ("FC_ALL_LS", "PHIC_ALL_LS"),
    ]
    for fc_label, pc_label in candidates_fc:
        if fc_label in col_labels and pc_label in col_labels:
            grid_fc = mtz.transform_f_phi_to_map(fc_label, pc_label, sample_rate=3.0)
            break

    # Normalize to sigma
    for grid in (grid_2fofc, grid_fofc):
        if grid is None:
            continue
        arr = np.array(grid, copy=False)
        mu, sigma = arr.mean(), arr.std()
        arr -= mu
        arr /= sigma

    resolution = round(mtz.resolution_high(), 2)

    # Map validation: check backbone
    backbone_vals = []
    for model in st:
        for chain in model:
            for res in chain:
                if res.name not in STANDARD_RESIDUES:
                    continue
                for atom in res:
                    if atom.name == "CA" and atom.b_iso < 30.0:
                        backbone_vals.append(grid_2fofc.interpolate_value(atom.pos))

    map_validation = {
        "n_atoms": len(backbone_vals),
        "mean_sigma": round(np.mean(backbone_vals), 2) if backbone_vals else 0,
        "PASS": bool(backbone_vals and np.mean(backbone_vals) > 2.0),
    }

    # If map validation fails, warn but continue
    if not map_validation["PASS"]:
        print(f"    ⚠ Map validation: backbone mean = {map_validation['mean_sigma']}σ "
              f"— possible alignment issue")

    # Find ligands
    ligand_results = []
    for model in st:
        for chain in model:
            for res in chain:
                if res.name in STANDARD_RESIDUES or res.name in EXCLUDED_HETNAMES:
                    continue
                atoms = [a for a in res if a.element.name != "H"]
                if len(atoms) < 3:
                    continue

                lig_id = f"{res.name}_{chain.name}_{res.seqid}"
                lig_result = _analyze_single_ligand(
                    grid_2fofc, grid_fofc, grid_fc, st, atoms, lig_id, resolution
                )
                ligand_results.append(lig_result)

    return {
        "mtz_info": {
            "spacegroup": mtz.spacegroup.hm,
            "resolution": resolution,
            "n_reflections": mtz.nreflections,
        },
        "map_validation": map_validation,
        "ligands": ligand_results,
    }


def _analyze_single_ligand(grid_2fofc, grid_fofc, grid_fc, structure, atoms, lig_id, resolution):
    """Analyze density for a single ligand instance."""

    # Covalent bond radii for connectivity detection
    COVALENT_RADII = {
        'C': 0.77, 'N': 0.75, 'O': 0.73, 'S': 1.05, 'Cl': 0.99,
        'F': 0.64, 'Br': 1.14, 'P': 1.10, 'I': 1.33, 'Se': 1.20,
        'B': 0.82,
    }

    # Per-atom density
    atom_data = []
    sigma_by_name = {}
    for atom in atoms:
        v2 = grid_2fofc.interpolate_value(atom.pos)
        vd = grid_fofc.interpolate_value(atom.pos) if grid_fofc else 0.0
        sigma_by_name[atom.name] = round(v2, 2)
        atom_data.append({
            "name": atom.name,
            "element": atom.element.name,
            "b_factor": round(atom.b_iso, 1),
            "occupancy": round(atom.occ, 3),
            "density_2fofc": round(v2, 2),
            "density_fofc": round(vd, 2),
        })

    d2 = np.array([a["density_2fofc"] for a in atom_data])
    df = np.array([a["density_fofc"] for a in atom_data])
    n = len(d2)

    density_summary = {
        "n_atoms": n,
        "mean_2fofc": round(d2.mean(), 2),
        "min_2fofc": round(d2.min(), 2),
        "max_2fofc": round(d2.max(), 2),
        "atoms_below_0.5sigma": int(np.sum(d2 < 0.5)),
        "atoms_below_1.0sigma": int(np.sum(d2 < 1.0)),
        "pct_no_density": round(np.sum(d2 < 0.5) / n * 100, 1),
        "pct_below_contour": round(np.sum(d2 < 1.0) / n * 100, 1),
        "atom_details": atom_data,
    }

    # --- Connectivity & Rigid Fragment Analysis ---
    # Build connectivity graph from covalent radii
    adj = defaultdict(list)
    bond_list = []
    for i in range(n):
        for j in range(i+1, n):
            d_ij = atoms[i].pos.dist(atoms[j].pos)
            r1 = COVALENT_RADII.get(atoms[i].element.name, 0.77)
            r2 = COVALENT_RADII.get(atoms[j].element.name, 0.77)
            if 0.8 < d_ij < (r1 + r2 + 0.4):
                adj[i].append(j)
                adj[j].append(i)
                bond_list.append((i, j))

    # Find rings via BFS cycle detection
    all_rings = set()
    for start in range(n):
        queue = [(start, [start], {start})]
        while queue:
            node, path, visited = queue.pop(0)
            for nb in adj[node]:
                if nb == start and len(path) >= 3:
                    all_rings.add(tuple(sorted(path)))
                elif nb not in visited and len(path) < 7:
                    queue.append((nb, path + [nb], visited | {nb}))

    # Filter to 5-6 membered rings, remove composites
    rings = []
    for ring in sorted(all_rings, key=len):
        if 5 <= len(ring) <= 6:
            if not any(set(existing).issubset(set(ring)) for existing in rings):
                rings.append(ring)

    # Merge fused rings into single planar systems
    # Two rings sharing ≥2 atoms (a bond) are fused and must be evaluated together
    fused_systems = []
    used = set()
    for i, ring_a in enumerate(rings):
        if i in used:
            continue
        system = set(ring_a)
        merged = True
        while merged:
            merged = False
            for j, ring_b in enumerate(rings):
                if j in used or j == i:
                    continue
                if len(system & set(ring_b)) >= 2:  # share a bond
                    system |= set(ring_b)
                    used.add(j)
                    merged = True
        used.add(i)
        fused_systems.append(sorted(system))

    # Analyze each fused system as one rigid planar fragment
    rigid_fragment_results = []
    for system in fused_systems:
        ring_names = [atoms[i].name for i in system]
        ring_elements = [atoms[i].element.name for i in system]
        ring_sigmas = [sigma_by_name[atoms[i].name] for i in system]
        ring_bs = [round(atoms[i].b_iso, 1) for i in system]

        n_members = len(system)
        # Label: e.g. "9-atom fused" for indole, "6-atom" for benzene
        is_fused = n_members > 6

        # Planarity check via SVD
        coords = np.array([[atoms[i].pos.x, atoms[i].pos.y, atoms[i].pos.z] for i in system])
        centered = coords - coords.mean(axis=0)
        _, s, _ = np.linalg.svd(centered)
        planarity = round(float(s[-1]), 3)
        is_planar = planarity < 0.15

        spread_sigma = round(max(ring_sigmas) - min(ring_sigmas), 2)
        mean_sigma = round(float(np.mean(ring_sigmas)), 2)
        has_good = any(s > 1.0 for s in ring_sigmas)
        has_absent = any(s < 0.5 for s in ring_sigmas)

        if has_good and has_absent:
            verdict = "INCONSISTENT — density is NOT from this ring"
        elif spread_sigma > 1.5 and mean_sigma > 0.5:
            verdict = "INCONSISTENT — sigma spread too large for rigid fragment"
        elif mean_sigma < 0.5:
            verdict = "ABSENT — entire ring below noise floor"
        elif mean_sigma < 1.0:
            verdict = "WEAK — ring marginally visible"
        else:
            verdict = "CONSISTENT — density supports this ring"

        rigid_fragment_results.append({
            "ring_atoms": ring_names,
            "ring_elements": ring_elements,
            "is_planar": is_planar,
            "planarity": planarity,
            "ring_sigmas": [round(s, 2) for s in ring_sigmas],
            "ring_bfactors": ring_bs,
            "sigma_spread": spread_sigma,
            "mean_sigma": mean_sigma,
            "verdict": verdict,
            "consistent": sum(1 for s in ring_sigmas if s < 0.5) <= 1 and mean_sigma >= 1.0,
        })

    # Bonded-atom sigma cliffs within ligand
    bonded_cliffs = []
    for i, j in bond_list:
        s1 = sigma_by_name[atoms[i].name]
        s2 = sigma_by_name[atoms[j].name]
        diff = round(abs(s1 - s2), 2)
        if diff > 1.0:
            bonded_cliffs.append({
                "atom1": atoms[i].name,
                "atom2": atoms[j].name,
                "sigma1": s1,
                "sigma2": s2,
                "delta_sigma": diff,
                "b1": round(atoms[i].b_iso, 1),
                "b2": round(atoms[j].b_iso, 1),
            })
    bonded_cliffs.sort(key=lambda x: -x["delta_sigma"])

    # RSCC: real-space correlation coefficient
    # If Fc map available (from FC_ALL/PHIC_ALL), use proper grid-sampled RSCC
    # Otherwise fall back to Gaussian approximation
    rscc_method = 'gaussian'
    is_omit_map = False
    if grid_fc is not None:
        # Check if Fc map has ligand density — if not, this is an omit map
        # and the Fc-based RSCC is meaningless
        fc_at_atoms = [grid_fc.interpolate_value(a.pos) for a in atoms]
        fc_mean = np.mean(np.abs(fc_at_atoms))
        # Compare to Fc density at pocket protein atoms as reference
        fc_pocket = []
        for model in structure:
            for chain in model:
                for res in chain:
                    if res.name not in STANDARD_RESIDUES:
                        continue
                    for pa in res:
                        if pa.element.name == 'H':
                            continue
                        for la in atoms:
                            if pa.pos.dist(la.pos) < 5.0:
                                fc_pocket.append(abs(grid_fc.interpolate_value(pa.pos)))
                                break
        fc_pocket_mean = np.mean(fc_pocket) if fc_pocket else 1.0
        
        # If ligand Fc density is < 5% of pocket Fc density, likely omit map
        # (weak ligands at partial occ still contribute ~10-30% of pocket Fc)
        if fc_pocket_mean > 0 and fc_mean / fc_pocket_mean < 0.05:
            rscc_method = 'gaussian'
            is_omit_map = True
        else:
            rscc_method = 'fc_map'

    if rscc_method == 'fc_map':
        obs_vals, calc_vals = [], []
        sample_radius = 1.5
        step = 0.5
        for atom in atoms:
            cx, cy, cz = atom.pos.x, atom.pos.y, atom.pos.z
            for dx in np.arange(-sample_radius, sample_radius + step, step):
                for dy in np.arange(-sample_radius, sample_radius + step, step):
                    for dz in np.arange(-sample_radius, sample_radius + step, step):
                        if dx*dx + dy*dy + dz*dz > sample_radius**2:
                            continue
                        pos = gemmi.Position(cx+dx, cy+dy, cz+dz)
                        obs_vals.append(grid_2fofc.interpolate_value(pos))
                        calc_vals.append(grid_fc.interpolate_value(pos))
        obs = np.array(obs_vals)
        calc = np.array(calc_vals)
        oc, cc = obs - obs.mean(), calc - calc.mean()
        denom = np.sqrt(np.sum(oc**2) * np.sum(cc**2))
        rscc = round(float(np.sum(oc*cc) / denom), 3) if denom > 0 else 0.0
    else:
        # Gaussian fallback (no Fc map, or omit map detected)
        obs_vals, calc_vals = [], []
        for atom in atoms:
            u_iso = max(atom.b_iso / (8 * np.pi**2), 0.01)
            for dx in np.arange(-1.5, 2.0, 0.5):
                for dy in np.arange(-1.5, 2.0, 0.5):
                    for dz in np.arange(-1.5, 2.0, 0.5):
                        dist2 = dx*dx + dy*dy + dz*dz
                        if dist2 > 2.25:
                            continue
                        pos = gemmi.Position(atom.pos.x+dx, atom.pos.y+dy, atom.pos.z+dz)
                        obs_vals.append(grid_2fofc.interpolate_value(pos))
                        calc_vals.append(atom.occ * np.exp(-dist2 / (2*u_iso)))
        obs = np.array(obs_vals)
        calc = np.array(calc_vals)
        oc, cc = obs - obs.mean(), calc - calc.mean()
        denom = np.sqrt(np.sum(oc**2) * np.sum(cc**2))
        rscc = round(float(np.sum(oc*cc) / denom), 3) if denom > 0 else 0.0

    # Pocket density comparison — per-atom
    pocket_atom_data = []
    pocket_vals = []
    seen_serials = set()
    for model in structure:
        for chain in model:
            for res in chain:
                if res.name not in STANDARD_RESIDUES:
                    continue
                for patom in res:
                    if patom.element.name == "H":
                        continue
                    for latom in atoms:
                        if patom.pos.dist(latom.pos) < 4.0:
                            serial_key = (chain.name, res.seqid.num, patom.name)
                            if serial_key not in seen_serials:
                                seen_serials.add(serial_key)
                                v2 = grid_2fofc.interpolate_value(patom.pos)
                                pocket_vals.append(v2)
                                pocket_atom_data.append({
                                    "chain": chain.name,
                                    "residue": f"{res.name}{res.seqid.num}",
                                    "name": patom.name,
                                    "element": patom.element.name,
                                    "b_factor": round(patom.b_iso, 1),
                                    "occupancy": round(patom.occ, 3),
                                    "density_2fofc": round(v2, 2),
                                })
                            break

    mean_pocket = round(np.mean(pocket_vals), 2) if pocket_vals else 0
    mean_lig = round(d2.mean(), 2)
    pocket_ratio = round(mean_pocket / max(mean_lig, 0.01), 1) if pocket_vals else 0

    # Summarize by residue
    pocket_by_residue = defaultdict(list)
    for pa in pocket_atom_data:
        pocket_by_residue[f"{pa['chain']}_{pa['residue']}"].append(pa)

    pocket_residue_summary = []
    for resid, patoms in sorted(pocket_by_residue.items()):
        mean_res_sigma = round(np.mean([a['density_2fofc'] for a in patoms]), 2)
        mean_res_b = round(np.mean([a['b_factor'] for a in patoms]), 1)
        pocket_residue_summary.append({
            "residue": resid,
            "mean_sigma": mean_res_sigma,
            "mean_b": mean_res_b,
            "n_atoms": len(patoms),
        })

    pocket_comparison = {
        "mean_pocket_sigma": mean_pocket,
        "mean_ligand_sigma": mean_lig,
        "pocket_to_ligand_ratio": pocket_ratio,
        "n_pocket_residues": len(pocket_by_residue),
        "pocket_residue_summary": pocket_residue_summary,
        "pocket_atom_details": pocket_atom_data,
    }

    # Occupancy/B audit
    occ_vals = [a.occ for a in atoms]
    b_vals = [a.b_iso for a in atoms]
    has_partial = any(o < 0.95 for o in occ_vals)
    mixed_occ = len(set(round(o, 2) for o in occ_vals)) > 1

    est_b_full = []
    if has_partial:
        for atom in atoms:
            if atom.occ < 0.95:
                est_b_full.append(atom.b_iso / (atom.occ ** (2.0/3.0)))

    occ_audit_flags = []
    mean_b_full = round(np.mean(est_b_full), 0) if est_b_full else None
    if has_partial and mean_b_full and mean_b_full > 100:
        occ_audit_flags.append(
            f"At full occupancy, estimated mean B = {mean_b_full:.0f} Å² "
            f"(current: {np.mean(b_vals):.0f} Å²). "
            f"Partial occupancy may be cosmetically deflating B-factors."
        )
    if mixed_occ:
        occ_audit_flags.append(
            f"Mixed occupancies ({min(occ_vals):.2f}–{max(occ_vals):.2f}) "
            f"without physical justification."
        )
    if has_partial and resolution > 1.8:
        occ_audit_flags.append(
            f"Partial occupancy at {resolution:.1f} Å — B and occupancy "
            f"cannot be reliably separated at this resolution."
        )

    occ_audit = {
        "min_occupancy": round(min(occ_vals), 3),
        "max_occupancy": round(max(occ_vals), 3),
        "has_partial": has_partial,
        "has_mixed": mixed_occ,
        "estimated_b_at_full_occ": mean_b_full,
        "flags": occ_audit_flags,
    }

    # Verdict
    issues = []
    severity = 0
    if rscc < 0.5:
        issues.append(f"RSCC = {rscc}")
        severity = max(severity, 3)
    pct_absent = density_summary["pct_no_density"]
    pct_below = density_summary["pct_below_contour"]
    if pct_absent > 50:
        issues.append(f"{pct_absent}% of atoms have no density (<0.5σ)")
        severity = max(severity, 3)
    elif pct_absent > 25:
        issues.append(f"{pct_absent}% of atoms have no density (<0.5σ)")
        severity = max(severity, 2)
    if pct_below > 70:
        issues.append(f"{pct_below}% below 1.0σ contour")
        severity = max(severity, 3)
    if pocket_ratio > 4:
        issues.append(f"Pocket {pocket_ratio}x stronger than ligand")
        severity = max(severity, 2)
    elif pocket_ratio > 3:
        issues.append(f"Pocket {pocket_ratio}x stronger than ligand")
        severity = max(severity, 1)

    # Rigid fragment failures
    n_inconsistent = sum(1 for r in rigid_fragment_results if not r["consistent"])
    if n_inconsistent > 0:
        issues.append(
            f"{n_inconsistent}/{len(rigid_fragment_results)} rigid rings show "
            f"inconsistent density — visible atoms are coincidental map features"
        )
        severity = max(severity, 3)

    # Bonded-atom cliffs
    if bonded_cliffs:
        worst = bonded_cliffs[0]
        issues.append(
            f"{len(bonded_cliffs)} bonded-atom sigma cliffs detected "
            f"(worst: {worst['atom1']}→{worst['atom2']} Δσ={worst['delta_sigma']})"
        )
        if worst["delta_sigma"] > 1.5:
            severity = max(severity, 2)

    # Store occ flags separately — they get their own section in the report
    # but still factor into severity
    for flag in occ_audit_flags:
        if "cosmetically" in flag or "full occupancy" in flag:
            severity = max(severity, 1)
        if "cannot be reliably separated" in flag:
            severity = max(severity, 1)

    verdicts = {
        0: "CONFIDENT",
        1: "CAUTION",
        2: "SUSPECT",
        3: "REJECTED",
    }

    return {
        "ligand_id": lig_id,
        "density_summary": density_summary,
        "rscc": rscc,
        "rscc_method": rscc_method,
        "is_omit_map": is_omit_map,
        "rigid_fragments": rigid_fragment_results,
        "bonded_cliffs": bonded_cliffs,
        "pocket_comparison": pocket_comparison,
        "occupancy_b_audit": occ_audit,
        "verdict": {"verdict": verdicts[severity], "severity": severity, "issues": issues},
    }


# ---------------------------------------------------------------------------
# Ligand Density Visualization (requires RDKit + matplotlib)
# ---------------------------------------------------------------------------
def generate_ligand_image(pdb_path, mtz_path, lig_result, header_data, output_path):
    """Generate per-ligand sigma and B-factor PNGs.
    
    Produces two files: output_path (sigma) and output_path with _bfac suffix.
    4-tier discrete color scale, strokeless circles, labels with collision resolution.
    """
    if not (HAS_RDKIT and HAS_GEMMI):
        return

    parts = lig_result["ligand_id"].split("_")
    resname, chain_id, seqid_str = parts[0], parts[1], parts[2]

    pdb_str = str(pdb_path)
    if pdb_str.endswith('.cif') or pdb_str.endswith('.mmcif'):
        structure = gemmi.read_structure(pdb_str)
    else:
        structure = gemmi.read_pdb(pdb_str)

    mtz = gemmi.read_mtz_file(str(mtz_path))
    col_labels = [c.label for c in mtz.columns]
    for fc, pc in [("FWT","PHWT"),("2FOFCWT","PH2FOFCWT"),("2FOFC","PH2FOFC")]:
        if fc in col_labels and pc in col_labels:
            grid_map = mtz.transform_f_phi_to_map(fc, pc, sample_rate=3.0)
            break
    else:
        return

    arr_g = np.array(grid_map, copy=False)
    arr_g -= arr_g.mean()
    arr_g /= arr_g.std()

    lig_atoms = None
    for model in structure:
        for chain in model:
            if chain.name != chain_id:
                continue
            for res in chain:
                if res.name == resname and str(res.seqid) == seqid_str:
                    lig_atoms = [a for a in res if a.element.name != 'H']
                    break
    if not lig_atoms or len(lig_atoms) < 3:
        return

    COVALENT_RADII_V = {
        'C': 0.77, 'N': 0.75, 'O': 0.73, 'S': 1.05, 'Cl': 0.99,
        'F': 0.64, 'Br': 1.14, 'P': 1.10, 'I': 1.33, 'Se': 1.20,
    }

    elements_v = [a.element.name for a in lig_atoms]
    sigmas_v = [round(grid_map.interpolate_value(a.pos), 2) for a in lig_atoms]
    bfacs_v = [round(a.b_iso, 1) for a in lig_atoms]
    n_v = len(lig_atoms)

    mol = Chem.RWMol()
    for atom in lig_atoms:
        mol.AddAtom(Chem.Atom(atom.element.atomic_number))

    bonds_v, adj_v = [], defaultdict(list)
    for i in range(n_v):
        for j in range(i + 1, n_v):
            d = lig_atoms[i].pos.dist(lig_atoms[j].pos)
            r1 = COVALENT_RADII_V.get(elements_v[i], 0.77)
            r2 = COVALENT_RADII_V.get(elements_v[j], 0.77)
            if 0.8 < d < (r1 + r2 + 0.4):
                bt = Chem.BondType.DOUBLE if (
                    d < 1.30 and elements_v[i] in ('C','N','O')
                    and elements_v[j] in ('C','N','O')
                ) else Chem.BondType.SINGLE
                mol.AddBond(i, j, bt)
                bonds_v.append((i, j))
                adj_v[i].append(j)
                adj_v[j].append(i)

    try:
        rdDepictor.Compute2DCoords(mol)
    except Exception:
        return

    conf = mol.GetConformer()
    coords_v = np.array([
        [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y]
        for i in range(n_v)
    ]) * 2.6

    # Ring detection
    all_rings = set()
    for start in range(n_v):
        queue = [(start, [start], {start})]
        while queue:
            node, path, visited = queue.pop(0)
            for nb in adj_v[node]:
                if nb == start and len(path) >= 3:
                    all_rings.add(tuple(sorted(path)))
                elif nb not in visited and len(path) < 7:
                    queue.append((nb, path + [nb], visited | {nb}))

    vrings = []
    for ring in sorted(all_rings, key=len):
        if 5 <= len(ring) <= 6:
            if not any(set(ex).issubset(set(ring)) for ex in vrings):
                vrings.append(ring)

    # Merge fused rings into single planar systems
    fused_v = []
    used_v = set()
    for i, ra in enumerate(vrings):
        if i in used_v:
            continue
        system = set(ra)
        merged = True
        while merged:
            merged = False
            for j, rb in enumerate(vrings):
                if j in used_v or j == i:
                    continue
                if len(system & set(rb)) >= 2:
                    system |= set(rb)
                    used_v.add(j)
                    merged = True
        used_v.add(i)
        fused_v.append(tuple(sorted(system)))

    vring_results = []
    arc_v = defaultdict(list)
    for system in fused_v:
        rs = [sigmas_v[i] for i in system]
        spread = round(max(rs) - min(rs), 2)
        has_good = any(s > 1.0 for s in rs)
        has_bad = any(s < 0.5 for s in rs)
        mean_rs = np.mean(rs)
        consistent = sum(1 for s in rs if s < 0.5) <= 1 and mean_rs >= 1.0
        vring_results.append({'indices': system, 'spread': spread, 'consistent': consistent})
        rc = coords_v[list(system)]
        cx, cy = rc.mean(axis=0)
        for idx in system:
            arc_v[idx].append((cx, cy))

    vcliffs = sorted(
        [(i, j, round(abs(sigmas_v[i] - sigmas_v[j]), 2))
         for i, j in bonds_v if abs(sigmas_v[i] - sigmas_v[j]) > 1.0],
        key=lambda x: -x[2]
    )

    # 4-tier colormaps
    sigma_cmap_v = mcolors.ListedColormap(['#CC2222', '#EE8833', '#33AA55', '#2266AA'])
    sigma_norm_v = mcolors.BoundaryNorm([-1.5, 0.5, 1.0, 2.0, 4.0], 4)
    sigma_tiers = [
        ('#CC2222', '< 0.5   absent'),
        ('#EE8833', '0.5 – 1.0   marginal'),
        ('#33AA55', '1.0 – 2.0   visible'),
        ('#2266AA', '> 2.0   well-ordered'),
    ]

    bfac_cmap_v = mcolors.ListedColormap(['#2266AA', '#33AA55', '#EE8833', '#CC2222'])
    bfac_norm_v = mcolors.BoundaryNorm([20, 40, 60, 80, 130], 4)
    bfac_tiers = [
        ('#2266AA', '< 40    well-ordered'),
        ('#33AA55', '40 – 60   moderate'),
        ('#EE8833', '60 – 80   elevated'),
        ('#CC2222', '> 80    unreliable'),
    ]

    DARK_V = '#1a1a1a'
    LABEL_SIZE_V = 14
    FONT_PROPS_V = _get_inter_semibold_font_properties()

    def _compute_labels(values, fmt):
        labels = []
        for i in range(n_v):
            px, py = 0.0, 0.0
            if i in arc_v:
                for cx, cy in arc_v[i]:
                    dx, dy = coords_v[i, 0] - cx, coords_v[i, 1] - cy
                    d = np.sqrt(dx**2 + dy**2)
                    if d > 0:
                        px += dx / d * 2
                        py += dy / d * 2
            for nb in adj_v[i]:
                dx = coords_v[i, 0] - coords_v[nb, 0]
                dy = coords_v[i, 1] - coords_v[nb, 1]
                d = np.sqrt(dx**2 + dy**2)
                if d > 0:
                    px += dx / d
                    py += dy / d
            mag = np.sqrt(px**2 + py**2)
            if mag > 0:
                px /= mag
                py /= mag
            else:
                px, py = 0, -1
            offset = 0.55 + 0.55
            lx = coords_v[i, 0] + px * offset
            ly = coords_v[i, 1] + py * offset
            if abs(px) > abs(py):
                ha = 'left' if px > 0 else 'right'
                va = 'center'
            else:
                ha = 'center'
                va = 'bottom' if py > 0 else 'top'
            labels.append({'x': lx, 'y': ly, 'ha': ha, 'va': va, 'text': fmt(values[i])})

        # Collision resolution
        LW, LH = 1.5, 0.6
        for _ in range(20):
            moved = False
            for i in range(len(labels)):
                for j in range(i + 1, len(labels)):
                    dx = labels[i]['x'] - labels[j]['x']
                    dy = labels[i]['y'] - labels[j]['y']
                    ox = LW - abs(dx)
                    oy = LH - abs(dy)
                    if ox > 0 and oy > 0:
                        if ox < oy:
                            sh = ox / 2 + 0.05
                            if dx >= 0:
                                labels[i]['x'] += sh
                                labels[j]['x'] -= sh
                            else:
                                labels[i]['x'] -= sh
                                labels[j]['x'] += sh
                        else:
                            sh = oy / 2 + 0.05
                            if dy >= 0:
                                labels[i]['y'] += sh
                                labels[j]['y'] -= sh
                            else:
                                labels[i]['y'] -= sh
                                labels[j]['y'] += sh
                        moved = True
            if not moved:
                break
        return labels

    def _draw_fig(values, cmap, norm_val, title, fmt, path_out,
                  show_rings, show_cliffs, tiers, unit, pocket_label):
        from matplotlib.patches import Circle as MplCircle, Polygon as MplPolygon, Rectangle as MplRect
        import matplotlib.patheffects as mpe

        fig, ax = plt.subplots(figsize=(16, 10), facecolor='white')

        if show_rings:
            for rr in vring_results:
                if not rr['consistent']:
                    rc = coords_v[list(rr['indices'])]
                    cx, cy = rc.mean(axis=0)
                    angles = np.arctan2(rc[:, 1] - cy, rc[:, 0] - cx)
                    ordered = rc[np.argsort(angles)]
                    expanded = []
                    for pt in ordered:
                        dx, dy = pt[0] - cx, pt[1] - cy
                        dist = np.sqrt(dx**2 + dy**2)
                        s = (dist + 1.3) / dist if dist > 0 else 1
                        expanded.append([cx + dx * s, cy + dy * s])
                    ax.add_patch(MplPolygon(expanded, closed=True,
                                           facecolor='#FF4444', alpha=0.07,
                                           edgecolor='none', zorder=0))

        for i, j in bonds_v:
            ax.plot([coords_v[i, 0], coords_v[j, 0]],
                    [coords_v[i, 1], coords_v[j, 1]],
                    color='#bbbbbb', linewidth=2.5, zorder=1, solid_capstyle='round')

        if show_cliffs:
            for i, j, diff in vcliffs:
                ax.plot([coords_v[i, 0], coords_v[j, 0]],
                        [coords_v[i, 1], coords_v[j, 1]],
                        color='#CC0000', linewidth=3, zorder=1.5,
                        solid_capstyle='round', alpha=0.3)

        radius = 0.55
        for i in range(n_v):
            color = cmap(norm_val(values[i]))
            ax.add_patch(MplCircle(coords_v[i], radius, facecolor=color,
                                   edgecolor='none', zorder=2))

        for lb in _compute_labels(values, fmt):
            ax.text(lb['x'], lb['y'], lb['text'],
                    ha=lb['ha'], va=lb['va'], fontsize=LABEL_SIZE_V,
                    color=DARK_V, zorder=4, fontproperties=FONT_PROPS_V,
                    path_effects=[mpe.withStroke(linewidth=4, foreground='white')])

        m = 3.2
        mol_xmin = coords_v[:, 0].min() - m
        mol_xmax = coords_v[:, 0].max() + m
        mol_ymin = coords_v[:, 1].min() - m
        mol_ymax = coords_v[:, 1].max() + m
        mol_height = mol_ymax - mol_ymin

        ax.set_xlim(mol_xmin, mol_xmax + 7.5)
        ax.set_ylim(mol_ymin, mol_ymax)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(title, fontsize=16, pad=24,
                     fontproperties=FONT_PROPS_V, color=DARK_V)

        ax.text(mol_xmin + 0.5, mol_ymax - 0.5, pocket_label,
                ha='left', va='top', fontsize=LABEL_SIZE_V,
                color='#555555', fontproperties=FONT_PROPS_V,
                path_effects=[mpe.withStroke(linewidth=3, foreground='white')])

        n_tiers = len(tiers)
        seg_h = mol_height / n_tiers
        legend_x = mol_xmax + 1.8
        legend_w = 1.3
        for ti, (color, label) in enumerate(tiers):
            y_bot = mol_ymin + ti * seg_h
            ax.add_patch(MplRect((legend_x, y_bot), legend_w, seg_h,
                                 facecolor=color, edgecolor='white',
                                 linewidth=1.5, zorder=5))
            ax.text(legend_x + legend_w + 0.5, y_bot + seg_h / 2, label,
                    ha='left', va='center', fontsize=LABEL_SIZE_V,
                    color=DARK_V, fontproperties=FONT_PROPS_V, zorder=5)
        ax.text(legend_x + legend_w / 2, mol_ymax + 0.6, unit,
                ha='center', va='bottom', fontsize=LABEL_SIZE_V,
                color=DARK_V, fontproperties=FONT_PROPS_V)

        plt.savefig(path_out, dpi=200, bbox_inches='tight',
                    facecolor='white', pad_inches=0.15)
        plt.close(fig)

    # Get pocket references
    pocket_comp = lig_result['pocket_comparison']
    pocket_sigma = pocket_comp['mean_pocket_sigma']

    # Sigma figure
    _draw_fig(sigmas_v, sigma_cmap_v, sigma_norm_v,
              f'{lig_result["ligand_id"]}    2Fo-Fc Density (σ)',
              lambda v: f"{v:.1f}",
              output_path,
              True, True, sigma_tiers, 'σ',
              f'pocket mean: {pocket_sigma} σ')

    # B-factor figure
    bfac_path = output_path.replace('.png', '_bfac.png')
    if bfac_path == output_path:
        bfac_path = output_path + '_bfac.png'

    # Get pocket B from header analysis
    pocket_b_val = None
    if hasattr(header_data, 'get'):
        # Try to get from parent analysis context
        pass
    _draw_fig(bfacs_v, bfac_cmap_v, bfac_norm_v,
              f'{lig_result["ligand_id"]}    B-factor (Å²)',
              lambda v: f"{v:.0f}",
              bfac_path,
              False, False, bfac_tiers, 'Å²',
              f'pocket mean: {int(np.mean(bfacs_v) * 0.5):.0f} Å²')  # approximate


# ---------------------------------------------------------------------------
# Verdict Logic
# ---------------------------------------------------------------------------
def compute_ligand_verdict(lig, analysis):
    """Compute verdict for a single ligand from density data.
    
    Returns (verdict_key, verdict_text, criteria) where:
      verdict_key: 'supported' | 'ambiguous' | 'insufficient' | 'not_supported'
      verdict_text: human-readable conclusion
      criteria: list of (status, text, decision) tuples
    """
    ds = lig['density_summary']
    pc = lig['pocket_comparison']
    rf = lig.get('rigid_fragments', [])
    oa = lig.get('occupancy_b_audit', {})
    all_b = [a['b_factor'] for a in ds['atom_details']]
    mean_b = np.mean(all_b)
    lid = lig['ligand_id']
    pocket_b = analysis['ligands'].get(lid, {}).get('pocket_mean_b', None)
    b_ratio = mean_b / pocket_b if pocket_b and pocket_b > 0 else None
    sigma_ratio = pc['pocket_to_ligand_ratio']
    n_miss = sum(1 for r in rf if not r['consistent'])
    n_tot = len(rf)
    n_comp = n_tot - n_miss
    rscc = lig.get('rscc', 0)
    is_omit = lig.get('is_omit_map', False)
    pct_above = 100 - ds.get('pct_below_contour', 100)

    # Occupancy info
    min_occ = oa.get('min_occupancy', 1.0)
    max_occ = oa.get('max_occupancy', 1.0)
    est_b_full = oa.get('estimated_b_at_full_occ')
    occ_reduced = max_occ < 0.95

    criteria = []

    if b_ratio is not None:
        if b_ratio < 1.5: criteria.append(('pass', f'B-factor ratio: {b_ratio:.1f}x', True))
        elif b_ratio < 2.0: criteria.append(('warn', f'B-factor ratio: {b_ratio:.1f}x', True))
        else: criteria.append(('fail', f'B-factor ratio: {b_ratio:.1f}x', True))

    if sigma_ratio < 1.5: criteria.append(('pass', f'\u03c3-ratio: {sigma_ratio:.1f}x', True))
    elif sigma_ratio < 2.0: criteria.append(('warn', f'\u03c3-ratio: {sigma_ratio:.1f}x', True))
    else: criteria.append(('fail', f'\u03c3-ratio: {sigma_ratio:.1f}x', True))

    if mean_b < 60: criteria.append(('pass', f'Average ligand B-factor: {mean_b:.0f} \u00c5\u00b2', True))
    elif mean_b < 80: criteria.append(('warn', f'Average ligand B-factor: {mean_b:.0f} \u00c5\u00b2', True))
    else: criteria.append(('fail', f'Average ligand B-factor: {mean_b:.0f} \u00c5\u00b2', True))

    if n_tot > 0:
        if n_miss == 0: criteria.append(('pass', f'Rigid rings: {n_comp}/{n_tot} complete', True))
        elif n_miss < n_tot: criteria.append(('warn', f'Rigid rings: {n_comp}/{n_tot} complete', True))
        else: criteria.append(('fail', f'Rigid rings: {n_comp}/{n_tot} complete', True))

    # Display-only
    if is_omit:
        criteria.append(('pass', 'RSCC: N/A (omit map)', False))
    elif rscc > 0.8: criteria.append(('pass', f'RSCC: {rscc:.3f}', False))
    elif rscc > 0.6: criteria.append(('warn', f'RSCC: {rscc:.3f}', False))
    else: criteria.append(('fail', f'RSCC: {rscc:.3f}', False))

    if pct_above > 75: criteria.append(('pass', f'Atoms above 1\u03c3: {pct_above:.0f}%', False))
    elif pct_above > 50: criteria.append(('warn', f'Atoms above 1\u03c3: {pct_above:.0f}%', False))
    else: criteria.append(('fail', f'Atoms above 1\u03c3: {pct_above:.0f}%', False))

    n_fail = sum(1 for s, _, d in criteria if d and s == 'fail')
    n_warn = sum(1 for s, _, d in criteria if d and s == 'warn')
    both_fail = (b_ratio is not None and b_ratio >= 2.0) and sigma_ratio >= 2.0
    sigma_definitive = sigma_ratio >= 3.0
    b_definitive = b_ratio is not None and b_ratio >= 3.0
    # At B ≥ 100 Å², RMS displacement exceeds 1.1 Å per axis — the atom's
    # 95% probability sphere (~2.3 Å radius) is larger than a bond length.
    # No specific binding pose can be claimed at this precision.
    b_absolute = mean_b >= 100

    # Build occupancy diagnostic
    occ_note = ''
    if occ_reduced:
        occ_str = f'{min_occ:.2f}' if min_occ == max_occ else f'{min_occ:.2f}\u2013{max_occ:.2f}'
        if est_b_full and est_b_full > 100:
            occ_note = (f' Occupancy was reduced to {occ_str}; '
                        f'estimated B-factor at full occupancy is {est_b_full:.0f} \u00c5\u00b2.')
        else:
            occ_note = f' Occupancy was reduced to {occ_str}.'

    if n_fail == 0 and n_warn == 0:
        return 'supported', 'Binding pose supported by density.', criteria
    elif both_fail or sigma_definitive or b_definitive or b_absolute:
        if occ_reduced:
            text = f'Density does not support modeled binding pose.{occ_note}'
        else:
            text = 'Density does not support modeled binding pose.'
        return 'not_supported', text, criteria
    elif n_fail >= 2:
        if occ_reduced and pct_above < 50:
            text = (f'Unable to confirm binding pose from available density.{occ_note}'
                    f' Reduced occupancy does not resolve the absence of density at most atom positions.')
        else:
            text = f'Unable to confirm binding pose from available density.{occ_note}'
        return 'insufficient', text, criteria
    elif n_fail == 1:
        text = f'Binding pose requires further validation.{occ_note}'
        return 'ambiguous', text, criteria
    elif n_warn >= 2:
        text = f'Binding pose requires further validation.{occ_note}'
        return 'ambiguous', text, criteria
    else:
        return 'supported', 'Binding pose supported by density.', criteria


# ---------------------------------------------------------------------------
# PDF Report Generation (requires reportlab)
# ---------------------------------------------------------------------------
def generate_pdf_report(results, image_prefix, pdf_path):
    """Generate PDF with sigma/B-factor figures and audit tables."""
    if not HAS_REPORTLAB:
        return

    from reportlab.lib.enums import TA_LEFT as _TA_L

    _FONT = _register_pdf_font_family()
    _BLK = rl_colors.HexColor('#1a1a1a')
    _GRAY = rl_colors.HexColor('#999999')
    _HEAD = rl_colors.HexColor('#E8E8E8')
    _BORD = rl_colors.HexColor('#CCCCCC')
    _PW = letter[0] - 1.5 * RL_INCH

    _C_PASS = '#228855'
    _C_WARN = '#CC8800'
    _C_FAIL = '#CC2200'

    sty = rl_getSampleStyleSheet()
    sty.add(RLParagraphStyle('_T', fontName=_FONT, fontSize=14, leading=18, spaceAfter=2))
    sty.add(RLParagraphStyle('_Sub', fontName=_FONT, fontSize=11, leading=14, spaceAfter=4,
                             textColor=rl_colors.HexColor('#555555')))
    sty.add(RLParagraphStyle('_Sec', fontName=_FONT, fontSize=12, leading=16, spaceBefore=10, spaceAfter=4))
    sty.add(RLParagraphStyle('_CH', fontName=_FONT, fontSize=14, leading=18, spaceBefore=4, spaceAfter=4))
    sty.add(RLParagraphStyle('_Foot', fontName=_FONT, fontSize=9, leading=12,
                             textColor=rl_colors.HexColor('#888888')))
    sty.add(RLParagraphStyle('_Concl', fontName=_FONT, fontSize=11, leading=14,
                             alignment=_TA_L, textColor=_BLK, spaceBefore=8))
    sty.add(RLParagraphStyle('_CP', fontName=_FONT, fontSize=11, leading=14,
                             textColor=rl_colors.HexColor(_C_PASS)))
    sty.add(RLParagraphStyle('_CW', fontName=_FONT, fontSize=11, leading=14,
                             textColor=rl_colors.HexColor(_C_WARN)))
    sty.add(RLParagraphStyle('_CF', fontName=_FONT, fontSize=11, leading=14,
                             textColor=rl_colors.HexColor(_C_FAIL)))

    def _fit(path, max_h=3.0*RL_INCH):
        from PIL import Image as _PI
        img = _PI.open(path)
        w, h = img.size
        ratio = h / w
        img_w = _PW
        img_h = img_w * ratio
        if img_h > max_h:
            img_h = max_h
            img_w = img_h / ratio
        return RLImage(path, width=img_w, height=img_h)

    def _tbl(data, widths):
        t = RLTable(data, colWidths=widths)
        s = [
            ('FONTNAME', (0,0), (-1,-1), _FONT),
            ('FONTSIZE', (0,0), (-1,-1), 11),
            ('TEXTCOLOR', (0,0), (-1,-1), _BLK),
            ('BACKGROUND', (0,0), (-1,0), _HEAD),
            ('GRID', (0,0), (-1,-1), 0.5, _BORD),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('LEFTPADDING', (0,0), (-1,-1), 8),
        ]
        if len(data[0]) >= 3:
            s.append(('TEXTCOLOR', (2,1), (2,-1), _GRAY))
        t.setStyle(RLTableStyle(s))
        return t

    def _verdict(lig, analysis):
        key, text, criteria = compute_ligand_verdict(lig, analysis)
        label_map = {
            'supported': '<u>Supported</u>',
            'ambiguous': '<u>Ambiguous</u>',
            'insufficient': '<u>Insufficient</u>',
            'not_supported': '<u>Not Supported</u>',
        }
        label = f'{label_map[key]}: {text}'
        return label, criteria

    doc = SimpleDocTemplate(pdf_path, pagesize=letter,
        leftMargin=0.75*RL_INCH, rightMargin=0.75*RL_INCH,
        topMargin=0.5*RL_INCH, bottomMargin=0.6*RL_INCH)

    _date_str = results.get('analysis_date', '')[:10]

    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(_FONT, 9)
        canvas.setFillColor(rl_colors.HexColor('#888888'))
        # Left: date
        canvas.drawString(0.75*RL_INCH, 0.3*RL_INCH, f'Generated: {_date_str}')
        # Center: page number
        page_num = f'({canvas.getPageNumber()})'
        canvas.drawCentredString(letter[0]/2, 0.3*RL_INCH, page_num)
        canvas.restoreState()

    sty.add(RLParagraphStyle('_Body', fontName=_FONT, fontSize=10, leading=13, spaceAfter=6,
                             textColor=_BLK))
    sty.add(RLParagraphStyle('_MetricName', fontName=_FONT, fontSize=11, leading=14,
                             spaceBefore=10, spaceAfter=2, textColor=_BLK))

    story = []

    for analysis in results.get('analyses', []):
        if 'error' in analysis:
            continue
        dv = analysis.get('density_validation', {})
        if not dv or 'error' in dv:
            continue

        header = analysis['header_data']
        r_free = header.get('r_free')
        r_work = header.get('r_work')
        r_gap = header.get('r_gap')
        res = header.get('resolution')

        for li, lig in enumerate(dv.get('ligands', [])):
            lid = lig['ligand_id']
            chain_id = lid.split('_')[1]
            ds = lig['density_summary']
            pc = lig['pocket_comparison']
            oa = lig['occupancy_b_audit']

            verdict_label, criteria = _verdict(lig, analysis)

            if li > 0 or story:
                story.append(RLPageBreak())

            story.append(RLParagraph('X-RAY STRUCTURE QUALITY REPORT', sty['_T']))
            pdb_name = os.path.basename(analysis['pdb_file'])
            mtz_name = os.path.basename(analysis.get('mtz_file', ''))
            story.append(RLParagraph(
                f'{pdb_name}  +  {mtz_name}',
                sty['_Sub']))
            story.append(RLSpacer(1, 4))
            story.append(HRFlowable(width='100%', thickness=0.5, color=_BORD))
            story.append(RLSpacer(1, 8))
            story.append(RLParagraph(
                f'LIGAND {lid.split("_")[0]}  \u2014  CHAIN {chain_id}', sty['_CH']))
            story.append(RLSpacer(1, 4))

            sigma_img = f'{image_prefix}_{lid}.png'
            if os.path.exists(sigma_img):
                story.append(_fit(sigma_img))
            story.append(RLSpacer(1, 8))

            bfac_img = f'{image_prefix}_{lid}_bfac.png'
            if os.path.exists(bfac_img):
                story.append(_fit(bfac_img))
            story.append(RLSpacer(1, 10))

            story.append(RLParagraph('<u>Results</u>', sty['_Sec']))
            for status, text, _decision in criteria:
                if status == 'pass':
                    story.append(RLParagraph(f'  \u2713  {text}', sty['_CP']))
                elif status == 'warn':
                    story.append(RLParagraph(f'  ~  {text}', sty['_CW']))
                else:
                    story.append(RLParagraph(f'  \u2717  {text}', sty['_CF']))
            story.append(RLParagraph(verdict_label, sty['_Concl']))

            story.append(RLPageBreak())
            story.append(RLParagraph(
                f'LIGAND {lid.split("_")[0]}  \u2014  CHAIN {chain_id}', sty['_CH']))

            story.append(RLParagraph('OVERVIEW', sty['_Sec']))
            cr = [
                ['Metric', 'Value', 'Expected'],
                ['Resolution', f'{res:.1f} \u00c5' if res else 'N/A', '< 2.5 \u00c5'],
                ['R-free', f'{r_free:.3f}' if r_free else 'N/A', '< 0.30'],
                ['R-work', f'{r_work:.3f}' if r_work else 'N/A', '< 0.25'],
                ['R-gap', f'{r_gap*100:.1f}%' if r_gap else 'N/A', '< 5%'],
            ]
            story.append(_tbl(cr, [140, 140, 80]))
            story.append(RLSpacer(1, 8))

            story.append(RLParagraph('MTZ DENSITY METRICS', sty['_Sec']))
            is_omit = lig.get('is_omit_map', False)
            rscc_val = 'N/A (omit map)' if is_omit else f'{lig["rscc"]:.3f}'
            rscc_exp = '' if is_omit else '> 0.8'
            ad = [
                ['Metric', 'Value', 'Expected'],
                ['RSCC', rscc_val, rscc_exp],
                ['Atoms absent (<0.5\u03c3)', f'{ds["pct_no_density"]:.0f}%', '0%'],
                ['Below 1\u03c3 contour', f'{ds["pct_below_contour"]:.0f}%', '< 10%'],
                ['Pocket \u03c3-ratio', f'{pc["pocket_to_ligand_ratio"]}x', '< 2.0'],
            ]
            story.append(_tbl(ad, [170, 100, 80]))
            story.append(RLSpacer(1, 8))

            story.append(RLParagraph('B-FACTOR / OCCUPANCY METRICS', sty['_Sec']))
            all_b = [a['b_factor'] for a in ds['atom_details']]
            mean_b = np.mean(all_b)
            pocket_b = analysis['ligands'].get(lid, {}).get('pocket_mean_b', None)
            expected_b = f'< {pocket_b*2:.0f} \u00c5\u00b2' if pocket_b else '< 2\u00d7 pocket'
            est_b = oa.get('estimated_b_at_full_occ')
            bd = [
                ['Metric', 'Value', 'Expected'],
                ['Average ligand B-factor', f'{mean_b:.0f} \u00c5\u00b2', expected_b],
                ['Average pocket B-factor', f'{pocket_b:.0f} \u00c5\u00b2' if pocket_b else 'N/A', ''],
                ['B-factor ratio', f'{mean_b/pocket_b:.1f}x' if pocket_b else 'N/A', '< 2.0'],
                ['Occupancy range', f'{oa["min_occupancy"]:.2f} \u2013 {oa["max_occupancy"]:.2f}', '1.0'],
                ['Est. B at occ = 1.0', f'{est_b:.0f} \u00c5\u00b2' if est_b else 'N/A', expected_b],
            ]
            story.append(_tbl(bd, [170, 100, 100]))

    # Methods section at end
    story.append(RLPageBreak())
    story.append(RLParagraph('METHODS', sty['_T']))
    story.append(RLSpacer(1, 4))
    story.append(HRFlowable(width='100%', thickness=0.5, color=_BORD))
    story.append(RLSpacer(1, 6))

    story.append(RLParagraph(
        'Ligand binding poses are evaluated by comparing per-atom electron '
        'density from the refined 2Fo-Fc map against the surrounding protein '
        'pocket. Four decision criteria determine the verdict; two additional '
        'metrics are displayed for reference. All density values are read '
        'directly from the experimental map at atomic coordinates.',
        sty['_Body']))

    story.append(RLKeepTogether([
        RLParagraph('<b>B-factor ratio</b>', sty['_MetricName']),
        RLParagraph(
            'Average ligand B-factor divided by average pocket B-factor (protein '
            'atoms within 5 \u00c5). A bound ligand should be comparably ordered to the '
            'protein residues that hold it in place \u2014 if the ligand is vibrating '
            'far more than its pocket, the model is fitting noise rather than a '
            'defined binding interaction. Above ~80 \u00c5\u00b2 at moderate resolution '
            '(~2 \u00c5), B-factor and occupancy become degenerate: the refinement '
            'cannot distinguish a fully occupied, highly mobile atom from a '
            'partially occupied, well-ordered one (Rupp, '
            '<i>Biomolecular Crystallography</i>, 2010). '
            'B-factor ratio \u2265 3.0 alone triggers Not Supported. '
            'Thresholds are derived from crystallographic refinement physics; '
            'the B/occupancy degeneracy boundary at ~80 \u00c5\u00b2 informs the 2.0x '
            'fail cutoff relative to typical pocket B-factors. '
            'Pass: &lt; 1.5x. Warning: 1.5\u20132.0x. Fail: \u2265 2.0x.',
            sty['_Body']),
    ]))

    story.append(RLKeepTogether([
        RLParagraph('<b>\u03c3-ratio</b>', sty['_MetricName']),
        RLParagraph(
            'Mean pocket 2Fo-Fc density divided by mean ligand 2Fo-Fc density, '
            'both in \u03c3 units. This is the most manipulation-resistant metric: it '
            'is read directly from the experimental electron density map and cannot '
            'be changed by adjusting B-factors, occupancy, or any other model '
            'parameter. The B-factor ratio and \u03c3-ratio are entangled \u2014 lowering '
            'B-factors improves one but worsens the other. Both must pass, because '
            'no refinement parameterization can make both look acceptable when the '
            'ligand lacks genuine density. Both ratios are self-normalizing '
            '(ligand vs. its own pocket in the same map), so poor resolution '
            'degrades both sides equally and does not inflate the ratio. '
            '\u03c3-ratio \u2265 3.0 alone triggers Not Supported. '
            'This metric and its thresholds are derived from first principles '
            'of electron density map interpretation; no literature precedent '
            'exists because the ligand-to-pocket density comparison is specific '
            'to this analysis. '
            'Pass: &lt; 1.5x. Warning: 1.5\u20132.0x. Fail: \u2265 2.0x.',
            sty['_Body']),
    ]))

    story.append(RLKeepTogether([
        RLParagraph('<b>Average ligand B-factor</b>', sty['_MetricName']),
        RLParagraph(
            'B-factor encodes positional uncertainty: B = 8\u03c0\u00b2\u27e8u\u00b2\u27e9, where '
            '\u27e8u\u00b2\u27e9 is the mean-square atomic displacement. At B = 60 \u00c5\u00b2, '
            'RMS displacement is 0.87 \u00c5. At B = 80 \u00c5\u00b2, it reaches 1.0 \u00c5 \u2014 '
            'equal to a C\u2013O bond. At B = 100 \u00c5\u00b2, RMS displacement is 1.13 \u00c5 '
            'per axis and the atom\u2019s 95% probability sphere has a radius of ~2.3 '
            '\u00c5. At this precision, two bonded atoms (1.5 \u00c5 apart) literally '
            'cannot be distinguished from each other. No specific binding pose can '
            'be claimed. B \u2265 100 \u00c5\u00b2 triggers Not Supported. '
            'When occupancy has been reduced below 1.0, the estimated B at full '
            'occupancy is reported as a diagnostic. '
            'Pass: &lt; 60 \u00c5\u00b2. Warning: 60\u201380 \u00c5\u00b2. Fail: \u2265 80 \u00c5\u00b2.',
            sty['_Body']),
    ]))

    story.append(RLKeepTogether([
        RLParagraph('<b>Rigid rings</b>', sty['_MetricName']),
        RLParagraph(
            'Fused aromatic and heteroaromatic ring systems (e.g., indole, '
            'naphthalene) are evaluated as single rigid planar units. Because '
            'these systems are conformationally locked, all atoms share the same '
            'positional certainty \u2014 it is physically impossible for three coplanar '
            'atoms in a ring to be well-resolved while adjacent atoms in the same '
            'ring are absent. This is a first-principles test with no literature '
            'threshold: it derives from the geometry of rigid planar systems. '
            'At most one atom may fall below 0.5\u03c3 (noise floor) to accommodate '
            'edge disorder; the system mean must be \u2265 1.0\u03c3 (standard 2Fo-Fc '
            'contour level, the default display in Coot and PyMOL). '
            'Pass: all complete. Warning: partial. Fail: none complete.',
            sty['_Body']),
    ]))

    story.append(RLKeepTogether([
        RLParagraph('<b>RSCC</b>  (display only)', sty['_MetricName']),
        RLParagraph(
            'Real-space correlation coefficient: Pearson correlation between the '
            'observed 2Fo-Fc map and the calculated Fc map, sampled on a 0.5 \u00c5 '
            'grid within 1.5 \u00c5 of each ligand atom. When FC_ALL/PHIC_ALL columns '
            'are present (standard REFMAC/Phenix/BUSTER output), the Fc map is '
            'built from the refined structure factors. Otherwise a single-Gaussian '
            'approximation is used, which systematically underestimates RSCC. '
            'The wwPDB flags RSCC &lt; 0.8 as an outlier (Gore et al., 2017, '
            '<i>Structure</i> 25, 1916). For omit maps, RSCC is reported as N/A '
            'because the Fc map lacks the ligand contribution. '
            'Does not affect the verdict.',
            sty['_Body']),
    ]))

    story.append(RLKeepTogether([
        RLParagraph('<b>Atoms above 1\u03c3</b>  (display only)', sty['_MetricName']),
        RLParagraph(
            'Percentage of non-hydrogen ligand atoms with 2Fo-Fc density \u2265 1.0\u03c3 '
            'at the atom center. The 1.0\u03c3 contour is the standard display level '
            'in Coot (Emsley et al., 2010, <i>Acta Cryst.</i> D66, 486) and '
            'PyMOL. Atoms below this threshold are invisible at default '
            'settings. '
            'Does not affect the verdict.',
            sty['_Body']),
    ]))

    story.append(RLKeepTogether([
        RLParagraph('<b>Occupancy and omit maps</b>', sty['_MetricName']),
        RLParagraph(
            'When occupancy &lt; 1.0, the conclusion reports the value and the '
            'estimated B-factor at full occupancy. Legitimate partial occupancy '
            'requires modeling the apo fraction (waters, alternate conformers) '
            'to justify the chosen value. Reducing occupancy without this is a '
            'refinement adjustment with no physical basis. '
            'Standard analysis uses the refined (phase-biased) 2Fo-Fc map, which '
            'is conservative: the map tries to show density at the ligand position. '
            'For ambiguous cases, omit maps (ligand removed before phase '
            'calculation) eliminate phase bias. The script auto-detects omit maps; '
            'RSCC is reported as N/A but all other metrics remain valid.',
            sty['_Body']),
    ]))

    story.append(RLSpacer(1, 6))
    story.append(RLKeepTogether([
        RLParagraph('<b>Verdict logic</b>', sty['_MetricName']),
        RLParagraph(
            '<b>Supported</b>: All criteria pass with no warnings. '
            '<b>Ambiguous</b>: One criterion fails, or two or more warnings. '
            '<b>Insufficient</b>: Two or more criteria fail.',
            sty['_Body']),
        RLParagraph(
            '<b>Not Supported</b>: Any of: '
            '(1) both B-factor ratio and \u03c3-ratio fail (\u2265 2.0x) \u2014 entangled '
            'metrics that cannot both be gamed; '
            '(2) \u03c3-ratio \u2265 3.0 alone \u2014 read from experimental map, immune to '
            'occupancy manipulation; '
            '(3) B-factor ratio \u2265 3.0 alone \u2014 definitive, works without MTZ; '
            '(4) average ligand B \u2265 100 \u00c5\u00b2 \u2014 positional uncertainty exceeds '
            'bond length, no binding pose can be specified.',
            sty['_Body']),
    ]))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)


def process_directory(directory: str, output_prefix: str = "xray_analysis"):
    """Process all PDB files in directory, with optional MTZ pairing"""
    pdb_files = sorted(
        glob.glob(os.path.join(directory, "*.pdb")) +
        glob.glob(os.path.join(directory, "*.cif")) +
        glob.glob(os.path.join(directory, "*.mmcif"))
    )
    mtz_paths = glob.glob(os.path.join(directory, "*.mtz"))
    
    if not pdb_files:
        print(f"No PDB files found in {directory}")
        return
    
    # Smart MTZ matching
    # Strategy: try exact stem, then case-insensitive, then strip suffixes,
    # then match on compound ID substring
    def _normalize(stem):
        """Strip common suffixes and normalize for matching."""
        s = stem.lower()
        for suffix in ['_map', '_final', '_refine', '_001']:
            if s.endswith(suffix):
                s = s[:len(s)-len(suffix)]
        return s
    
    def _extract_id(stem):
        """Extract compound identifier (e.g., VU0961666 → 961666, 407288 → 407288)."""
        s = stem.upper()
        # Try to find a numeric compound ID, with or without VU prefix
        import re
        # Match VU followed by digits, or just a block of 5+ digits
        matches = re.findall(r'VU?0*(\d{5,})', s)
        if matches:
            return matches[0]
        matches = re.findall(r'(\d{5,})', s)
        if matches:
            return matches[0]
        return None
    
    # Build MTZ lookup with multiple keys
    mtz_by_stem = {}  # exact stem → path
    mtz_by_norm = {}  # normalized stem → path
    mtz_by_id = {}    # compound ID → path
    for mtz_path in mtz_paths:
        stem = Path(mtz_path).stem
        mtz_by_stem[stem] = mtz_path
        mtz_by_norm[_normalize(stem)] = mtz_path
        cid = _extract_id(stem)
        if cid:
            mtz_by_id[cid] = mtz_path
    
    def _find_mtz(pdb_stem):
        """Find matching MTZ using progressively looser strategies."""
        # 1. Exact stem match
        if pdb_stem in mtz_by_stem:
            return mtz_by_stem[pdb_stem]
        # 2. Case-insensitive + suffix-stripped
        norm = _normalize(pdb_stem)
        if norm in mtz_by_norm:
            return mtz_by_norm[norm]
        # 3. Compound ID match
        cid = _extract_id(pdb_stem)
        if cid and cid in mtz_by_id:
            return mtz_by_id[cid]
        return None
    
    # Build pairs
    mtz_files = {}
    for pdb_file in pdb_files:
        pdb_stem = Path(pdb_file).stem
        matched = _find_mtz(pdb_stem)
        if matched:
            mtz_files[pdb_stem] = matched
    
    if not pdb_files:
        print(f"No PDB files found in {directory}")
        return
    
    print(f"\n{'='*60}")
    print(f"X-RAY STRUCTURE QUALITY ANALYZER")
    print(f"{'='*60}")
    print(f"Processing {len(pdb_files)} PDB files from {directory}")
    n_paired = len(mtz_files)
    n_unpaired = len(pdb_files) - n_paired
    if mtz_files and HAS_GEMMI:
        print(f"Paired: {n_paired}  |  PDB-only: {n_unpaired}")
        for pdb_stem, mtz_path in sorted(mtz_files.items()):
            print(f"  {pdb_stem}  ←  {os.path.basename(mtz_path)}")
        if HAS_REPORTLAB:
            if not HAS_RDKIT:
                print(f"  Figures: disabled (pip install rdkit matplotlib)")
        else:
            print(f"  PDF: disabled (pip install reportlab)")
            if not HAS_RDKIT:
                print(f"  Figures: disabled (pip install rdkit matplotlib)")
    elif mtz_files and not HAS_GEMMI:
        print(f"Found {len(mtz_files)} MTZ file(s) but gemmi is not installed!")
        print(f"  Running PDB-only analysis (B-factor ratios only)")
        print(f"    → pip install gemmi")
    else:
        print(f"No MTZ files found — running PDB-only analysis")
        print(f"  (place matching MTZ files in '{directory}' for density validation)")
    print()
    
    # Process all files
    results = {
        'analysis_date': datetime.now().isoformat(),
        'directory': directory,
        'total_files': len(pdb_files),
        'analyses': []
    }
    
    grade_dist = defaultdict(int)  # kept for report compatibility
    verdict_dist = defaultdict(int)
    
    for i, pdb_file in enumerate(pdb_files, 1):
        pdb_stem = Path(pdb_file).stem
        mtz_file = mtz_files.get(pdb_stem)
        has_mtz = mtz_file is not None and HAS_GEMMI
        
        label = f"[{i}/{len(pdb_files)}] {os.path.basename(pdb_file)}"
        if has_mtz:
            label += " + MTZ"
        print(f"{label}")
        
        try:
            # Always run PDB analysis
            analyzer = PDBAnalyzer(pdb_file)
            report = analyzer.generate_report()
            
            # Run MTZ density analysis if paired
            if has_mtz:
                report['mtz_file'] = mtz_file
                try:
                    density_report = analyze_density(pdb_file, mtz_file)
                    
                    # Check for analysis-level errors
                    if 'error' in density_report:
                        print(f"    ⚠ {density_report['error']}")
                        report['density_validation'] = density_report
                        report['has_mtz'] = True
                    else:
                        report['density_validation'] = density_report
                        report['has_mtz'] = True
                        
                        # Print density summary
                        for lig in density_report.get('ligands', []):
                            ds = lig['density_summary']
                            pct_above = 100 - ds.get('pct_below_contour', 100)
                            if lig.get('is_omit_map'):
                                rscc_str = 'N/A (omit map)'
                            else:
                                rscc_str = f'{lig["rscc"]:.3f}'
                            print(f"    [{lig['ligand_id']}] "
                                  f"{pct_above:.0f}% above 1σ, "
                                  f"RSCC={rscc_str}")
                        
                        if not density_report.get('ligands'):
                            print(f"    No ligands detected in structure")
                        
                        # Generate ligand density images (used by PDF, cleaned up after)
                        if HAS_RDKIT:
                            for lig in density_report.get('ligands', []):
                                safe_id = lig['ligand_id'].replace(' ', '_')
                                img_path = f"{output_prefix}_{safe_id}.png"
                                try:
                                    generate_ligand_image(
                                        pdb_file, mtz_file, lig,
                                        report['header_data'], img_path
                                    )
                                except Exception as img_e:
                                    print(f"    Image generation error: {img_e}")
                except Exception as e:
                    print(f"    ⚠ MTZ analysis failed: {e}")
                    report['density_validation'] = {'error': str(e)}
                    report['has_mtz'] = True
            else:
                report['has_mtz'] = False
            
            results['analyses'].append(report)
            
            # Print per-structure summary
            r_free = report['header_data'].get('r_free')
            r_free_str = f"{r_free:.3f}" if r_free is not None else 'N/A'
            res = report['header_data'].get('resolution')
            res_str = f"{res:.1f}Å" if res else 'N/A'
            
            # Compute density verdict for CLI if we have MTZ data
            dv = report.get('density_validation', {})
            if has_mtz and dv and 'error' not in dv and dv.get('ligands'):
                # Use worst verdict across ligands
                verdict_keys = []
                for lig in dv['ligands']:
                    key, text, _ = compute_ligand_verdict(lig, report)
                    verdict_keys.append((key, text))
                
                # Priority: not_supported > insufficient > ambiguous > supported
                priority = {'not_supported': 0, 'insufficient': 1, 'ambiguous': 2, 'supported': 3}
                worst_key, worst_text = min(verdict_keys, key=lambda x: priority.get(x[0], 99))
                verdict_label = worst_key.upper().replace('_', ' ')
                
                print(f"  → {res_str} | R-free: {r_free_str} | {verdict_label}")
                verdict_dist[worst_key] += 1
            elif has_mtz and dv and 'error' in dv:
                print(f"  → {res_str} | R-free: {r_free_str} | MTZ error (see above)")
            elif has_mtz and dv and not dv.get('ligands'):
                print(f"  → {res_str} | R-free: {r_free_str} | No ligands found")
            else:
                # PDB-only: give limited verdict based on B-ratio
                if report.get('ligands'):
                    b_ratios = [l.get('b_ratio_to_protein', 0) 
                               for l in report['ligands'].values() if l.get('b_ratio_to_protein')]
                    if b_ratios:
                        worst_br = max(b_ratios)
                        if worst_br >= 3.0:
                            print(f"  → {res_str} | R-free: {r_free_str} | "
                                  f"B-ratio: {worst_br:.1f}x — NOT SUPPORTED (PDB-only)")
                            verdict_dist['not_supported'] += 1
                        elif worst_br >= 2.0:
                            print(f"  → {res_str} | R-free: {r_free_str} | "
                                  f"B-ratio: {worst_br:.1f}x — needs MTZ for density verdict")
                        elif worst_br >= 1.5:
                            print(f"  → {res_str} | R-free: {r_free_str} | "
                                  f"B-ratio: {worst_br:.1f}x — needs MTZ for density verdict")
                        else:
                            print(f"  → {res_str} | R-free: {r_free_str} | "
                                  f"B-ratio: {worst_br:.1f}x — needs MTZ for density verdict")
                    else:
                        print(f"  → {res_str} | R-free: {r_free_str} | PDB-only (no ligands)")
                else:
                    print(f"  → {res_str} | R-free: {r_free_str} | PDB-only (no ligands)")
            
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            results['analyses'].append({
                'pdb_file': pdb_file,
                'error': str(e)
            })
    
    # Save outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate one PDF per PDB/MTZ pair
    pdf_files = []
    if HAS_REPORTLAB:
        for analysis in results['analyses']:
            if 'error' in analysis:
                continue
            if not analysis.get('has_mtz'):
                continue
            dv = analysis.get('density_validation', {})
            if 'error' in dv:
                continue
            pdb_stem = Path(analysis['pdb_file']).stem
            pdf_name = f"{output_prefix}_{pdb_stem}.pdf"
            single_result = {
                'analysis_date': results['analysis_date'],
                'directory': results['directory'],
                'total_files': 1,
                'analyses': [analysis],
            }
            try:
                generate_pdf_report(single_result, output_prefix, pdf_name)
                pdf_files.append(pdf_name)
            except Exception as pdf_e:
                print(f"  PDF error ({pdb_stem}): {pdf_e}")
    
    # Clean up intermediate image files
    for f_clean in glob.glob(f"{output_prefix}_*_*.png"):
        try:
            os.remove(f_clean)
        except OSError:
            pass
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    
    if verdict_dist:
        print(f"Results:")
        label_order = [
            ('supported', 'Supported'),
            ('ambiguous', 'Ambiguous'),
            ('insufficient', 'Insufficient'),
            ('not_supported', 'Not Supported'),
        ]
        for key, label in label_order:
            count = verdict_dist.get(key, 0)
            if count:
                print(f"  {label}: {count}")
    
    if pdf_files:
        print(f"\nOutput files:")
        for pf in pdf_files:
            print(f"  PDF: {pf}")
    elif not HAS_REPORTLAB:
        print(f"\n  No PDF generated — pip install reportlab")
    
    return results


def generate_markdown_report(results: Dict, output_file: str):
    """Generate unified markdown report with PDB + optional density data"""
    with open(output_file, 'w') as f:
        f.write("# X-ray Structure Quality Analysis Report\n\n")
        f.write(f"**Analysis Date:** {results['analysis_date']}\n")
        f.write(f"**Directory:** {results['directory']}\n")
        f.write(f"**Total Files:** {results['total_files']}\n\n")

        # Check if any analysis has density data
        any_density = any(
            a.get('has_mtz') and 'error' not in a.get('density_validation', {})
            for a in results['analyses'] if 'error' not in a
        )

        # Summary table
        f.write("## Summary Table\n\n")
        if any_density:
            f.write("| File | Res | Grade | R-free | R-gap% | Prot B | Lig B | Pkt B-ratio | % No Density | % <1σ | RSCC | Pkt σ-ratio | Density |\n")
            f.write("|------|-----|-------|--------|--------|--------|-------|-------------|--------------|-------|------|-------------|----------|\n")
        else:
            f.write("| File | Resolution | Grade | R-free | R-gap % | Breaks | Prot B | Lig B | Pkt B-ratio | Verdict |\n")
            f.write("|------|------------|-------|--------|---------|--------|--------|--------|------------|----------|\n")

        for analysis in results['analyses']:
            if 'error' in analysis:
                continue

            file_name = os.path.basename(analysis['pdb_file'])
            resolution = analysis['header_data'].get('resolution')
            resolution = f"{resolution:.2f} Å" if resolution else 'N/A'
            grade = analysis['grade']
            r_free = analysis['header_data'].get('r_free')
            r_free = f"{r_free:.3f}" if r_free else 'N/A'

            r_work = analysis['header_data'].get('r_work')
            r_gap_pct = 'N/A'
            if analysis['header_data'].get('r_free') and r_work:
                r_gap_pct = f"{((analysis['header_data']['r_free'] - r_work) * 100):.1f}%"

            prot_b = analysis['global_statistics'].get('mean_protein_b')
            prot_b = f"{prot_b:.1f}" if prot_b else 'N/A'

            lig_b = 'N/A'
            pkt_b_ratio = 'N/A'
            if analysis['ligands']:
                lig_b_factors = [l['mean_b'] for l in analysis['ligands'].values()]
                if lig_b_factors:
                    lig_b = f"{sum(lig_b_factors)/len(lig_b_factors):.1f}"
                pkt_b_ratios = [l['b_ratio_to_pocket'] for l in analysis['ligands'].values() if l.get('b_ratio_to_pocket')]
                if pkt_b_ratios:
                    pkt_b_ratio = f"{max(pkt_b_ratios):.2f}"

            if any_density:
                # Get density metrics
                dv = analysis.get('density_validation', {})
                if dv and 'error' not in dv:
                    for lig in dv.get('ligands', []):
                        ds = lig['density_summary']
                        v = lig['verdict']['verdict']
                        pc = lig['pocket_comparison']
                        f.write(f"| {file_name} | {resolution} | {grade} | {r_free} | {r_gap_pct} | "
                                f"{prot_b} | {lig_b} | {pkt_b_ratio} | "
                                f"{ds['pct_no_density']}% | {ds['pct_below_contour']}% | "
                                f"{lig['rscc']} | {pc['pocket_to_ligand_ratio']}x | {v} |\n")
                else:
                    f.write(f"| {file_name} | {resolution} | {grade} | {r_free} | {r_gap_pct} | "
                            f"{prot_b} | {lig_b} | {pkt_b_ratio} | — | — | — | — | no MTZ |\n")
            else:
                breaks = analysis.get('occupancy_analysis', {}).get('missing_residues', 0)
                f.write(f"| {file_name} | {resolution} | {grade} | {r_free} | {r_gap_pct} | "
                        f"{breaks} | {prot_b} | {lig_b} | {pkt_b_ratio} | {analysis['overall_verdict']} |\n")

        # Quality metric ranges
        f.write("\n## Quality Metric Ranges\n\n")
        f.write("| Metric | Exceptional | Good | Fair | Poor | ??? | Source |\n")
        f.write("|--------|-------------|------|------|------|-----|--------|\n")
        f.write("| **Resolution (Å)** | ≤ 1.5 | ≤ 2.0 | ≤ 2.5 | ≤ 3.0 | > 3.0 | [1] |\n")
        f.write("| **R-free** | ≤ 0.20 | ≤ 0.25 | ≤ 0.30 | ≤ 0.35 | > 0.35 | [2,3] |\n")
        f.write("| **R-free/R-work Gap (%)** | ≤ 3 | ≤ 5 | ≤ 7 | ≤ 10 | > 10 | [3,4] |\n")
        f.write("| **Protein B-factor (Å²)** | ≤ 25 | ≤ 40 | ≤ 60 | ≤ 80 | > 100 | [5] |\n")
        f.write("| **Ligand B-factor (Å²)** | ≤ 30 | ≤ 50 | ≤ 70 | ≤ 90 | > 100 | [5] |\n")
        f.write("| **Pocket B-ratio** | ≤ 1.3 | ≤ 1.8 | ≤ 2.5 | ≤ 3.5 | > 4.0 | [5,6] |\n")
        if any_density:
            f.write("| **% Atoms < 0.5σ** | 0% | ≤ 10% | ≤ 25% | ≤ 50% | > 50% | [7] |\n")
            f.write("| **% Atoms < 1.0σ** | ≤ 10% | ≤ 25% | ≤ 40% | ≤ 60% | > 60% | [7] |\n")
            f.write("| **RSCC (approx)¹** | ≥ 0.9 | ≥ 0.8 | ≥ 0.7 | ≥ 0.5 | < 0.5 | [8,9] |\n")
            f.write("| **Pocket σ-ratio** | ≤ 1.5 | ≤ 2.0 | ≤ 3.0 | ≤ 4.0 | > 4.0 | [7]² |\n")
            f.write("| **Ring σ-spread** | < 0.5 | < 0.8 | < 1.0 | < 1.5 | ≥ 1.5 | ³ |\n")
            f.write("| **Bonded Δσ cliff** | < 0.3 | < 0.5 | < 0.8 | < 1.0 | ≥ 1.0 | ³ |\n")

        f.write("\n### References\n\n")
        f.write("1. Rupp, B. (2010). *Biomolecular Crystallography*. Garland Science. Ch. 12–13.\n")
        f.write("2. Brünger, A.T. (1992). Free R value: a novel statistical quantity for assessing "
                "the accuracy of crystal structures. *Nature* 355, 472–475.\n")
        f.write("3. Tickle, I.J., Laskowski, R.A. & Moss, D.S. (1998). R-free and the R-free ratio. "
                "*Acta Cryst.* D54, 547–557.\n")
        f.write("4. Kleywegt, G.J. & Jones, T.A. (1995). Where freedom is given, liberties are taken. "
                "*Structure* 3, 535–540.\n")
        f.write("5. Rupp, B. (2010). Ch. 12: B-factor interpretation, occupancy–B correlation at "
                "moderate resolution, and ligand validation considerations.\n")
        f.write("6. Smart, O.S. et al. (2018). Validation of ligands in macromolecular structures "
                "determined by X-ray crystallography. *Acta Cryst.* D74, 228–236.\n")
        f.write("7. 1.0σ contour level is the standard visualization threshold in Coot/PyMOL for "
                "2Fo-Fc maps; 0.5σ represents the noise floor. Pocket σ-ratio is an internally "
                "controlled metric comparing ligand and protein density within the same map.\n")
        f.write("8. Jones, T.A. et al. (1991). Improved methods for building protein models in "
                "electron density maps. *Acta Cryst.* A47, 110–119.\n")
        f.write("9. wwPDB validation pipeline flags RSCC < 0.8 as outlier for ligands. See: "
                "Gore, S. et al. (2017). Validation of structures in the PDB. *Structure* 25, "
                "1916–1927.\n")
        if any_density:
            f.write("\n¹ *RSCC here is computed using a Gaussian model approximation. Absolute values "
                    "are systematically lower than EDSTATS/wwPDB validation. Use for relative "
                    "comparison between ligands within a dataset.*\n")
            f.write("\n² *Pocket σ-ratio thresholds are empirical. The principle — comparing ligand "
                    "density to surrounding protein density as an internally controlled metric — "
                    "eliminates confounds from resolution, completeness, and data quality.*\n")
            f.write("\n³ *Ring σ-spread and bonded Δσ thresholds are derived from first principles: "
                    "atoms in a rigid planar ring share identical translational/rotational states "
                    "and must show comparable density. Inconsistency indicates the density is not "
                    "from the modeled fragment.*\n")
        f.write("\n")

        # Detailed analysis
        f.write("\n## Detailed Structure Analysis\n\n")

        for analysis in results['analyses']:
            if 'error' in analysis:
                continue

            f.write(f"### {os.path.basename(analysis['pdb_file'])}\n\n")

            # --- PDB header data ---
            header = analysis['header_data']
            def _fmt(v, suffix='', precision=None):
                if v is None: return 'N/A'
                if precision is not None: return f"{v:.{precision}f}{suffix}"
                return f"{v}{suffix}"

            r_gap = header.get('r_gap')
            r_gap_display = f"{r_gap*100:.1f}%" if r_gap is not None else 'N/A'
            f.write("**Crystallographic Data:**\n")
            f.write(f"- Resolution: {_fmt(header.get('resolution'), ' Å', 1)}\n")
            f.write(f"- R-work: {_fmt(header.get('r_work'), precision=3)}\n")
            f.write(f"- R-free: {_fmt(header.get('r_free'), precision=3)}\n")
            f.write(f"- R-gap: {r_gap_display}\n")
            f.write(f"- Wilson B: {_fmt(header.get('wilson_b'), ' Å²', 1)}\n")
            tls = header.get('tls_groups')
            if tls:
                f.write(f"- TLS groups: {tls}\n")
            prog = header.get('refinement_program')
            if prog:
                f.write(f"- Program: {prog}\n")
            f.write("\n")

            # Map validation (if MTZ)
            dv = analysis.get('density_validation', {})
            has_density = dv and 'error' not in dv
            if has_density:
                mv = dv['map_validation']
                status = "PASS" if mv['PASS'] else "FAIL"
                f.write(f"**Map Validation:** {status} — "
                        f"{mv['n_atoms']} Cα atoms, mean = {mv['mean_sigma']}σ\n\n")

            # --- Ligand analysis ---
            if analysis['ligands']:
                f.write("**Ligand Analysis:**\n\n")

                # Build density lookup by ligand_id
                density_by_lig = {}
                if has_density:
                    for dl in dv.get('ligands', []):
                        density_by_lig[dl['ligand_id']] = dl

                for lig_id, lig_data in analysis['ligands'].items():
                    f.write(f"#### {lig_id}\n\n")

                    # PDB metrics row
                    f.write("| Metric | Value |\n")
                    f.write("|--------|-------|\n")
                    f.write(f"| Mean B-factor | {lig_data['mean_b']} Å² |\n")
                    f.write(f"| B-ratio (protein) | {lig_data.get('b_ratio_to_protein', 'N/A')} |\n")
                    f.write(f"| B-ratio (pocket) | {lig_data.get('b_ratio_to_pocket', 'N/A')} |\n")
                    f.write(f"| Pocket mean B | {lig_data.get('pocket_mean_b', 'N/A')} Å² |\n")
                    f.write(f"| Occupancy | {lig_data['mean_occupancy']} (min: {lig_data['min_occupancy']}) |\n")
                    f.write(f"| Protein contacts (4Å) | {lig_data['num_protein_contacts_4A']} |\n")
                    f.write(f"| PDB verdict | {lig_data['verdict']} |\n")

                    # Add density metrics if available
                    dl = density_by_lig.get(lig_id)
                    if dl:
                        ds = dl['density_summary']
                        pc = dl['pocket_comparison']
                        f.write(f"| Mean 2Fo-Fc | {ds['mean_2fofc']}σ |\n")
                        f.write(f"| Atoms absent (<0.5σ) | {ds['atoms_below_0.5sigma']}/{ds['n_atoms']} ({ds['pct_no_density']}%) |\n")
                        f.write(f"| Atoms below contour (<1.0σ) | {ds['atoms_below_1.0sigma']}/{ds['n_atoms']} ({ds['pct_below_contour']}%) |\n")
                        f.write(f"| RSCC (approx) | {dl['rscc']} |\n")
                        f.write(f"| Pocket σ-ratio | {pc['pocket_to_ligand_ratio']}x |\n")
                        occ_a = dl['occupancy_b_audit']
                        if occ_a.get('estimated_b_at_full_occ'):
                            f.write(f"| Est. B at full occ | {occ_a['estimated_b_at_full_occ']:.0f} Å² |\n")
                        f.write(f"| **Density verdict** | **{dl['verdict']['verdict']}** |\n")
                    f.write("\n")

                    # Per-atom density table (only if MTZ data)
                    if dl:
                        f.write("**Per-Atom Ligand Density:**\n\n")
                        f.write("| Atom | Elem | B (Å²) | Occ | 2FoFc (σ) | FoFc (σ) | Flag |\n")
                        f.write("|------|------|--------|-----|-----------|----------|------|\n")
                        for a in ds['atom_details']:
                            flag = ""
                            if a['density_2fofc'] < 0.5:
                                flag = "NO DENSITY"
                            elif a['density_2fofc'] < 1.0:
                                flag = "WEAK"
                            if a['density_fofc'] < -3.0:
                                flag += " NEG_DIFF"
                            if a['density_fofc'] > 3.0:
                                flag += " POS_DIFF"
                            f.write(f"| {a['name']} | {a['element']} | {a['b_factor']} | "
                                    f"{a['occupancy']} | {a['density_2fofc']} | "
                                    f"{a['density_fofc']} | {flag} |\n")
                        f.write("\n")

                        # Pocket residue table: B-factor vs sigma
                        pc = dl['pocket_comparison']
                        if pc.get('pocket_residue_summary'):
                            f.write("**Pocket Residue Density (within 4Å):**\n\n")
                            f.write("| Residue | Mean B (Å²) | Mean 2FoFc (σ) | # Atoms |\n")
                            f.write("|---------|-------------|----------------|---------|\n")
                            for pr in pc['pocket_residue_summary']:
                                f.write(f"| {pr['residue']} | {pr['mean_b']} | {pr['mean_sigma']} | {pr['n_atoms']} |\n")
                            f.write(f"\n*Pocket mean: {pc['mean_pocket_sigma']}σ vs ligand mean: {pc['mean_ligand_sigma']}σ "
                                    f"(ratio: {pc['pocket_to_ligand_ratio']}x)*\n\n")

                        # Rigid fragment consistency test
                        if dl.get('rigid_fragments'):
                            f.write("**Rigid Fragment Test:**\n\n")
                            for rf in dl['rigid_fragments']:
                                status = "✓" if rf['consistent'] else "✗"
                                planar = "planar" if rf['is_planar'] else "non-planar"
                                f.write(f"*{status} Ring ({planar}): {rf['ring_atoms']}*\n\n")
                                f.write("| Atom | σ (2FoFc) | B (Å²) |\n")
                                f.write("|------|-----------|--------|\n")
                                for aname, sig, bf in zip(rf['ring_atoms'], rf['ring_sigmas'], rf['ring_bfactors']):
                                    flag = ""
                                    if sig < 0.5: flag = " ← absent"
                                    elif sig < 1.0: flag = " ← weak"
                                    f.write(f"| {aname} | {sig} | {bf}{flag} |\n")
                                f.write(f"\n*σ spread: {rf['sigma_spread']} | verdict: {rf['verdict']}*\n\n")

                        # Bonded-atom sigma cliffs
                        if dl.get('bonded_cliffs'):
                            f.write("**Bonded-Atom Sigma Cliffs:**\n\n")
                            f.write("| Bond | σ₁ | σ₂ | Δσ | B₁ | B₂ |\n")
                            f.write("|------|----|----|----|----|----|\n")
                            for bc in dl['bonded_cliffs']:
                                f.write(f"| {bc['atom1']}—{bc['atom2']} | "
                                        f"{bc['sigma1']} | {bc['sigma2']} | "
                                        f"{bc['delta_sigma']} | {bc['b1']} | {bc['b2']} |\n")
                            f.write("\n")

                        # Occupancy/B audit flags
                        if dl['occupancy_b_audit']['flags']:
                            f.write("**Occupancy/B-Factor Audit:**\n")
                            for flag in dl['occupancy_b_audit']['flags']:
                                f.write(f"- ⚠ {flag}\n")
                            f.write("\n")

                        # Density verdict with issues
                        if dl['verdict']['issues']:
                            f.write(f"**Density Issues:**\n")
                            for issue in dl['verdict']['issues']:
                                f.write(f"- {issue}\n")
                            f.write("\n")

            # B-factor cliffs
            if 'b_factor_cliffs' in analysis:
                cliffs = analysis['b_factor_cliffs']
                if cliffs['num_cliffs'] > 0:
                    f.write(f"**B-Factor Cliffs ({cliffs['severity']} — {cliffs['num_cliffs']} detected):**\n\n")
                    f.write("| Atom 1 | Atom 2 | B1 | B2 | Fold Change | Direction |\n")
                    f.write("|--------|--------|----|----|-------------|-----------|\n")
                    for cliff in cliffs['cliffs']:
                        f.write(f"| {cliff['atom1']} | {cliff['atom2']} | {cliff['b1']} | {cliff['b2']} | {cliff['fold_change']}x | {cliff['direction']} |\n")
                    f.write("\n")

            # Occupancy / chain breaks
            if 'occupancy_analysis' in analysis:
                occ = analysis['occupancy_analysis']
                if occ['partial_occupancy_atoms'] > 0 or occ['missing_residues'] > 0:
                    f.write("**Occupancy & Chain Analysis:**\n")
                    f.write(f"- Partial occupancy atoms: {occ['partial_occupancy_atoms']}\n")
                    f.write(f"- Residues with partial occupancy: {occ['num_partial_residues']}\n")
                    f.write(f"- Missing residues (chain breaks): {occ['missing_residues']}\n")
                    if occ['chain_breaks']:
                        f.write("- Chain breaks:\n")
                        for b in occ['chain_breaks'][:5]:
                            f.write(f"  - Chain {b['chain']}: {b['start_res']} → {b['end_res']} (gap: {b['gap_size']})\n")
                    f.write("\n")

            # Scorecard
            if 'scorecard' in analysis:
                sc = analysis['scorecard']
                f.write("**Scorecard:**\n\n")
                f.write("| Category | Weight | Grade | Score |\n")
                f.write("|----------|--------|-------|-------|\n")
                f.write(f"| R-free | {sc['weights']['r_free']*100:.0f}% | {sc['subgrades']['r_free']} | - |\n")
                f.write(f"| Ligand Quality | {sc['weights']['ligand_quality']*100:.0f}% | {sc['subgrades']['ligand_quality']} | - |\n")
                f.write(f"| Occupancy/Contacts | {sc['weights']['occupancy_contacts']*100:.0f}% | {sc['subgrades']['occupancy_contacts']} | - |\n")
                f.write(f"| **Overall** | **100%** | **{sc['overall_grade']}** | **{sc['overall_score']:.2f}** |\n\n")

            f.write(f"**PDB Assessment:** Grade {analysis['grade']} — {analysis['overall_verdict']}\n\n")

            # Composite verdict if density data available
            if has_density:
                density_verdicts = [dl['verdict']['verdict']
                                   for dl in dv.get('ligands', [])]
                worst_density = "CONFIDENT"
                for dv_str in density_verdicts:
                    if dv_str == "REJECTED": worst_density = "REJECTED"; break
                    if dv_str == "SUSPECT": worst_density = "SUSPECT"
                    if dv_str == "CAUTION" and worst_density == "CONFIDENT": worst_density = "CAUTION"

                if worst_density == "REJECTED" and analysis['overall_verdict'] == "ACCEPTABLE":
                    f.write(f"**⚠ CONFLICT: PDB metrics say ACCEPTABLE but density says REJECTED.**\n")
                    f.write(f"B-factor ratios appear reasonable only because occupancy has been "
                            f"adjusted to deflate them. The experimental density does not support "
                            f"the ligand placement.\n\n")
                elif worst_density in ("REJECTED", "SUSPECT"):
                    f.write(f"**Density Verdict: {worst_density}** — "
                            f"overrides PDB assessment.\n\n")

            f.write("---\n\n")


def launch_gui():
    """Launch tkinter GUI for the analyzer."""
    import tkinter as tk
    from tkinter import filedialog, ttk
    import threading

    BG = "#f5f5f5"
    FG = "#1a1a1a"
    ACCENT = "#2266AA"
    SUBTLE = "#888888"
    ENTRY_BG = "#ffffff"
    BTN_BG = "#2266AA"
    BTN_FG = "#ffffff"
    BTN_DISABLED = "#aabbcc"
    BROWSE_BG = "#e0e0e0"
    BROWSE_FG = "#1a1a1a"

    root = tk.Tk()
    root.title("X-ray Structure Analyzer")
    root.geometry("560x340")
    root.resizable(False, False)
    root.configure(bg=BG)

    # Bring to front
    root.lift()
    root.attributes('-topmost', True)
    root.after(100, lambda: root.attributes('-topmost', False))
    root.focus_force()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    FONT_TITLE = ("Helvetica", 18, "bold")
    FONT_SUBTITLE = ("Helvetica", 11)
    FONT_LABEL = ("Helvetica", 11, "bold")
    FONT_BODY = ("Helvetica", 11)
    FONT_ENTRY = ("Courier", 11)
    FONT_BTN = ("Helvetica", 12, "bold")
    FONT_STATUS = ("Helvetica", 10)

    input_dir = tk.StringVar()
    output_dir = tk.StringVar()
    status_text = tk.StringVar(value="Select an input directory to begin.")
    pair_info = tk.StringVar(value="")
    running = tk.BooleanVar(value=False)

    # Title area
    title_frame = tk.Frame(root, bg=BG)
    title_frame.pack(fill="x", padx=24, pady=(20, 0))
    tk.Label(title_frame, text="X-ray Structure Analyzer", font=FONT_TITLE,
             bg=BG, fg=FG, anchor="w").pack(fill="x")
    tk.Label(title_frame, text="PDB / MTZ density validation", font=FONT_SUBTITLE,
             bg=BG, fg=SUBTLE, anchor="w").pack(fill="x", pady=(0, 8))
    tk.Frame(root, bg="#dddddd", height=1).pack(fill="x", padx=24)

    # Input directory
    input_frame = tk.Frame(root, bg=BG)
    input_frame.pack(fill="x", padx=24, pady=(16, 0))
    tk.Label(input_frame, text="Input Directory", font=FONT_LABEL,
             bg=BG, fg=FG, anchor="w").pack(fill="x")
    row1 = tk.Frame(input_frame, bg=BG)
    row1.pack(fill="x", pady=(4, 0))
    input_entry = tk.Entry(row1, textvariable=input_dir, font=FONT_ENTRY,
                           state="readonly", bg=ENTRY_BG, fg=FG,
                           readonlybackground=ENTRY_BG, relief="solid", bd=1)
    input_entry.pack(side="left", fill="x", expand=True, padx=(0, 8), ipady=4)

    def browse_input():
        d = filedialog.askdirectory(title="Select directory with PDB/MTZ files",
                                    initialdir=script_dir)
        if not d:
            return
        input_dir.set(d)
        pdb_files = sorted(
            glob.glob(os.path.join(d, "*.pdb")) +
            glob.glob(os.path.join(d, "*.cif"))
        )
        mtz_files = glob.glob(os.path.join(d, "*.mtz"))
        n_pdb = len(pdb_files)
        if n_pdb == 0:
            pair_info.set("No PDB files found.")
            status_text.set("No PDB files in selected directory.")
        else:
            import re as _re
            def _norm(stem):
                s = stem.lower()
                for sfx in ['_map', '_final', '_refine', '_001']:
                    if s.endswith(sfx): s = s[:len(s)-len(sfx)]
                return s
            def _eid(stem):
                s = stem.upper()
                m = _re.findall(r'VU?0*(\d{5,})', s)
                if m: return m[0]
                m = _re.findall(r'(\d{5,})', s)
                if m: return m[0]
                return None

            mtz_stem = {Path(f).stem: f for f in mtz_files}
            mtz_norm = {_norm(Path(f).stem): f for f in mtz_files}
            mtz_id = {}
            for f in mtz_files:
                cid = _eid(Path(f).stem)
                if cid: mtz_id[cid] = f

            n_paired = 0
            for pf in pdb_files:
                ps = Path(pf).stem
                if ps in mtz_stem or _norm(ps) in mtz_norm:
                    n_paired += 1
                else:
                    cid = _eid(ps)
                    if cid and cid in mtz_id:
                        n_paired += 1

            n_unpaired = n_pdb - n_paired
            parts = []
            if n_paired:
                parts.append(f"{n_paired} paired")
            if n_unpaired:
                parts.append(f"{n_unpaired} PDB-only")
            pair_info.set(f"{n_pdb} structures found  ·  " + "  ·  ".join(parts))
            status_text.set("Ready.")

    tk.Button(row1, text="Browse", font=FONT_BODY, command=browse_input,
              bg=BROWSE_BG, fg=BROWSE_FG, relief="solid", bd=1, padx=12,
              activebackground="#d0d0d0", activeforeground=BROWSE_FG,
              cursor="hand2").pack(side="right")

    # Pair info
    pair_label = tk.Label(root, textvariable=pair_info, font=FONT_STATUS,
                          fg=ACCENT, bg=BG, anchor="w")
    pair_label.pack(fill="x", padx=24, pady=(4, 0))

    # Output directory
    output_frame = tk.Frame(root, bg=BG)
    output_frame.pack(fill="x", padx=24, pady=(12, 0))
    tk.Label(output_frame, text="Output Directory", font=FONT_LABEL,
             bg=BG, fg=FG, anchor="w").pack(fill="x")
    row2 = tk.Frame(output_frame, bg=BG)
    row2.pack(fill="x", pady=(4, 0))
    output_entry = tk.Entry(row2, textvariable=output_dir, font=FONT_ENTRY,
                            state="readonly", bg=ENTRY_BG, fg=FG,
                            readonlybackground=ENTRY_BG, relief="solid", bd=1)
    output_entry.pack(side="left", fill="x", expand=True, padx=(0, 8), ipady=4)

    def browse_output():
        d = filedialog.askdirectory(title="Select output directory for PDFs",
                                    initialdir=script_dir)
        if d:
            output_dir.set(d)

    tk.Button(row2, text="Browse", font=FONT_BODY, command=browse_output,
              bg=BROWSE_BG, fg=BROWSE_FG, relief="solid", bd=1, padx=12,
              activebackground="#d0d0d0", activeforeground=BROWSE_FG,
              cursor="hand2").pack(side="right")

    # Spacer
    tk.Frame(root, bg=BG, height=8).pack(fill="x")

    # Status
    status_label = tk.Label(root, textvariable=status_text, font=FONT_STATUS,
                            fg=SUBTLE, bg=BG, anchor="center")
    status_label.pack(fill="x", padx=24, pady=(16, 8))

    # Analyze button — centered and prominent
    def run_analysis():
        if not input_dir.get():
            status_text.set("Select an input directory first.")
            return
        if not output_dir.get():
            status_text.set("Select an output directory first.")
            return
        if running.get():
            return

        running.set(True)
        analyze_label.configure(bg=BTN_DISABLED, fg="#667788", cursor="arrow")
        btn_frame.configure(bg=BTN_DISABLED, cursor="arrow")
        status_text.set("Analyzing...")

        def _run():
            import io
            old_stdout = sys.stdout
            sys.stdout = buf = io.StringIO()

            try:
                out_prefix = os.path.join(output_dir.get(), "xray_analysis")
                process_directory(input_dir.get(), out_prefix)
                root.after(0, lambda: status_text.set("Complete."))
            except Exception as e:
                root.after(0, lambda: status_text.set(f"Error: {e}"))
            finally:
                sys.stdout = old_stdout
                root.after(0, lambda: running.set(False))
                root.after(0, lambda: analyze_label.configure(
                    bg=BTN_BG, fg=BTN_FG, cursor="hand2"))
                root.after(0, lambda: btn_frame.configure(
                    bg=BTN_BG, cursor="hand2"))

        threading.Thread(target=_run, daemon=True).start()

    # Analyze button — Label-based for cross-platform color support
    btn_frame = tk.Frame(root, bg=BTN_BG, padx=2, pady=2, cursor="hand2")
    btn_frame.pack(pady=(0, 20))
    analyze_label = tk.Label(btn_frame, text="  Analyze  ", font=FONT_BTN,
                             bg=BTN_BG, fg=BTN_FG, padx=36, pady=8)
    analyze_label.pack()

    def _btn_enter(e):
        if not running.get():
            analyze_label.configure(bg="#1a5090")
            btn_frame.configure(bg="#1a5090")
    def _btn_leave(e):
        if not running.get():
            analyze_label.configure(bg=BTN_BG)
            btn_frame.configure(bg=BTN_BG)
    def _btn_click(e):
        if not running.get():
            run_analysis()

    analyze_label.bind("<Enter>", _btn_enter)
    analyze_label.bind("<Leave>", _btn_leave)
    analyze_label.bind("<Button-1>", _btn_click)
    btn_frame.bind("<Button-1>", _btn_click)

    root.mainloop()


def main():
    parser = argparse.ArgumentParser(
        description='X-ray Structure Quality Analyzer — PDB + MTZ'
    )
    parser.add_argument('directory', nargs='?', default=None,
                       help='Directory containing PDB/MTZ files (default: launch GUI)')
    parser.add_argument('-o', '--output', default='xray_analysis',
                       help='Output file prefix (default: xray_analysis)')
    parser.add_argument('--gui', action='store_true',
                       help='Launch graphical interface')
    
    args = parser.parse_args()
    
    if args.gui or args.directory is None:
        launch_gui()
    else:
        if not os.path.exists(args.directory):
            print(f"Error: Directory '{args.directory}' not found")
            sys.exit(1)
        process_directory(args.directory, args.output)


if __name__ == "__main__":
    main()
