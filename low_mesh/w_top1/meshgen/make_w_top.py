"""
w_top mesh in one go (OpenFOAM-6, run inside WSL):

  riser (z 0 .. 16.82)        blockMesh, gen_blockMeshDict.py --wtop
                              (same blocks as wo_intake; --preset selects the
                               resolution: full = wo_intake3, low = low_mesh)
  top part (dome, duct,       snappyHexMesh on a watertight STL built from
  vessel; z 15.82 .. 20.77)   w_top_fidelity.stp (duct roof = STEP B-spline)
  joint at z = 16.82          cyclicAMI  riser_top_AMI <-> top_interface_AMI

The w_top frame (riser at the origin, duct along -y) is turned -90 deg about z
and moved onto the riser axis, so the duct runs along -x, parallel to the
slots, as in Meshtest/all_geo/all_case1.

Usage (WSL):
  python3 make_w_top.py <caseDir> [--preset=full|low] [--step=<w_top_fidelity.stp>]
                        [--test-base=<dir>] [--reuse-top=<dir with constant/polyMesh>]

  <caseDir>     new folder -> copy of test_base; existing folder without a mesh
                is used as it is; a folder that already has constant/polyMesh is
                refused (nothing is overwritten)
  --reuse-top   take the top part from an earlier run (its meshgen/top) instead
                of running snappyHexMesh again (~5 min)

Output: <caseDir>/constant/polyMesh, log.<caseName> (checkMesh -allGeometry
-allTopology), 0/ boundary names set for this mesh, and <caseDir>/meshgen with
everything needed to rebuild (this script, gen_blockMeshDict.py, the STEP file,
the riser and snappy cases and their logs).
"""
import math
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FOAM_BASHRC = os.environ.get('FOAM_BASHRC', '/home/js/OpenFOAM/OpenFOAM-6/etc/bashrc')
TEST_BASE = '/home/js/Task/POSCO-furnace/Meshtest/test_base'

# ================================================================ geometry
CX, CY = -2.75528782867519, 1.47488765579627       # riser axis (blockMesh frame)
ZF = 16.82                      # riser top / duct floor
R1, R2 = 2.26, 3.95             # riser cap, vessel radius
YV = -12.738                    # vessel axis (w_top frame)
ZOUT = 15.82                    # vessel bottom (outlet)
NY, NZ = 0.992396519437701, -0.123081876033544     # normal of the duct section planes
ET = (0.0, -NZ, NY)             # in-plane "up" direction of the duct sections
BETA0 = math.atan2(ET[2], ET[1])    # section plane angle above the horizontal (82.93 deg)

# ================================================================ settings
NA = 120                        # STL points over a half circle (all seams)
NV = 120                        # STL duct sections
NB_CAP, NB_DOME = 40, 60
TOP_BASE_CELL = 0.1             # snappy background cell [m] (level 1 near walls / BOI: 0.05)
WALLS_TOP = 'center_top|right_top|left_top|center_bottom|wall3'
WALL_REGEX = '"(clearance|inlet_wall|gap1|wall1|wall2|wall3|center_top|right_top|left_top|center_bottom)"'


def lin(a, b, n):
    return [a + (b - a) * i / n for i in range(n + 1)]


ALPHA = lin(0.0, math.pi, NA)


# ================================================================ STEP B-spline
def step_entities(path):
    txt = open(path, encoding='latin-1').read().replace('\r', '').replace('\n', '')
    return {int(m.group(1)): m.group(2) for m in re.finditer(r'#(\d+)\s*=\s*(.*?);(?=#|ENDSEC)', txt)}


def step_point(E, i):
    return [float(v) for v in re.search(r'\(([^()]*)\)\s*\)$', E[i]).group(1).split(',')]


def bspline_surface(E):
    """The duct roof: the only B_SPLINE_SURFACE_WITH_KNOTS of the file."""
    for s in E.values():
        if s.startswith('B_SPLINE_SURFACE_WITH_KNOTS'):
            m = re.match(r"B_SPLINE_SURFACE_WITH_KNOTS\('[^']*',(\d+),(\d+),\((.*)\),\.(\w+)\.,\.(\w+)\.,"
                         r"\.(\w+)\.,\.(\w+)\.,\((.*?)\),\((.*?)\),\((.*?)\),\((.*?)\),", s)
            rows = re.findall(r'\(([^()]*)\)', m.group(3))
            ctrl = [[step_point(E, int(x.strip()[1:])) for x in r.split(',')] for r in rows]
            mult = lambda g: [int(x) for x in g.split(',')]
            knot = lambda g: [float(x) for x in g.split(',')]
            KU = [k for k, n in zip(knot(m.group(10)), mult(m.group(8))) for _ in range(n)]
            KV = [k for k, n in zip(knot(m.group(11)), mult(m.group(9))) for _ in range(n)]
            return int(m.group(1)), int(m.group(2)), ctrl, KU, KV
    raise ValueError('no B-spline surface in STEP file')


def deboor(k, x, t, c, p):
    d = [c[j + k - p] for j in range(p + 1)]
    for r in range(1, p + 1):
        for j in range(p, r - 1, -1):
            a = (x - t[j + k - p]) / (t[j + 1 + k - r] - t[j + k - p])
            d[j] = [(1 - a) * d[j - 1][q] + a * d[j][q] for q in range(len(d[j]))]
    return d[p]


def span(t, x, p, n):
    if x >= t[n]:
        return n - 1
    k = p
    while not (t[k] <= x < t[k + 1]):
        k += 1
    return k


# ================================================================ STL
def to_out(p):
    x, y, z = p
    return (CX + y, CY - x, z)


def area2(a, b, c):
    u = [b[k] - a[k] for k in range(3)]
    v = [c[k] - a[k] for k in range(3)]
    cr = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
    return math.sqrt(sum(x * x for x in cr))


def tri_grid(grid):
    out = []
    for i in range(len(grid) - 1):
        for j in range(len(grid[i]) - 1):
            a, b, c, d = grid[i][j], grid[i + 1][j], grid[i + 1][j + 1], grid[i][j + 1]
            for t in ((a, b, c), (a, c, d)):
                if area2(*t) > 1e-12:
                    out.append(t)
    return out


def fan(center, ring):
    return [(center, ring[i], ring[i + 1]) for i in range(len(ring) - 1)
            if area2(center, ring[i], ring[i + 1]) > 1e-12]


def ear_clip(poly):
    """Triangulate a simple planar polygon through all its boundary points."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    pts = list(poly)
    if sum(cross((0, 0), pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))) < 0:
        pts.reverse()
    idx = list(range(len(pts)))

    def inside(p, a, b, c):
        return cross(a, b, p) > 0 and cross(b, c, p) > 0 and cross(c, a, p) > 0
    tris = []
    while len(idx) > 3:
        n = len(idx)
        best = None
        for k in range(n):
            i0, i1, i2 = idx[(k - 1) % n], idx[k], idx[(k + 1) % n]
            a, b, c = pts[i0], pts[i1], pts[i2]
            if cross(a, b, c) <= 0:
                continue
            if any(inside(pts[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue
            best = k
            break
        if best is None:
            best = max(range(n), key=lambda k: cross(pts[idx[(k - 1) % n]], pts[idx[k]], pts[idx[(k + 1) % n]]))
        tris.append((pts[idx[(best - 1) % n]], pts[idx[best]], pts[idx[(best + 1) % n]]))
        idx.pop(best)
    tris.append(tuple(pts[i] for i in idx))
    return tris


def write_top_stl(step, out):
    """Watertight surfaces; all seams share their points."""
    du, dv, ctrl, KU, KV = bspline_surface(step_entities(step))

    def section(v):                 # roof points at the fixed section angles
        kv = span(KV, v, dv, len(ctrl[0]))
        col = [deboor(kv, v, KV, row, dv) for row in ctrl]
        c = (0.0, -12.738 * v, ZF)

        def at(u):
            return deboor(span(KU, u, du, len(col)), u, KU, col, du)

        def ang(u):
            p = at(u)
            q = [p[k] - c[k] for k in range(3)]
            return math.atan2(q[1] * ET[1] + q[2] * ET[2], q[0])
        pts = []
        for a in ALPHA:
            lo, hi = KU[0], KU[-1]
            for _ in range(60):
                m = 0.5 * (lo + hi)
                am = ang(m)
                if am < -math.pi / 2:
                    am += 2 * math.pi
                if am > a:
                    lo = m
                else:
                    hi = m
            pts.append(at(0.5 * (lo + hi)))
        return pts

    roof = []
    for j, v in enumerate(lin(0.0, 1.0, NV)):
        if j in (0, NV):            # duct ends: exact circles in the end planes
            r, c = (R1, (0.0, 0.0, ZF)) if j == 0 else (R2, (0.0, YV, ZF))
            row = [(c[0] + r * math.cos(a), c[1] + r * math.sin(a) * ET[1],
                    c[2] + r * math.sin(a) * ET[2]) for a in ALPHA]
        else:
            row = [tuple(p) for p in section(v)]
        row[0] = (row[0][0], row[0][1], ZF)
        row[-1] = (row[-1][0], row[-1][1], ZF)
        roof.append(row)
    edge_e = [row[0] for row in roof]
    edge_w = [row[-1] for row in roof]

    def circle(c, r, sgn):
        return [(c[0] + r * math.cos(a), c[1] + sgn * r * math.sin(a), ZF) for a in ALPHA]
    riser_p, riser_m = circle((0, 0), R1, 1), circle((0, 0), R1, -1)
    vessel_p, vessel_m = circle((0, YV), R2, 1), circle((0, YV), R2, -1)

    def lune(c, r, b0, b1, nb):
        return [[(c[0] + r * math.cos(a), c[1] + r * math.sin(a) * math.cos(b),
                  c[2] + r * math.sin(a) * math.sin(b)) for b in lin(b0, b1, nb)] for a in ALPHA]
    cap = lune((0, 0, ZF), R1, 0.0, BETA0, NB_CAP)
    dome = lune((0, YV, ZF), R2, BETA0, math.pi, NB_DOME)
    for i in range(NA + 1):
        cap[i][0], cap[i][-1] = riser_p[i], roof[0][i]
        dome[i][0], dome[i][-1] = roof[-1][i], vessel_m[i]

    riser_ring = riser_p + riser_m[::-1][1:]
    vessel_ring = vessel_p + vessel_m[::-1][1:]
    bottom_ring = [(p[0], p[1], ZOUT) for p in vessel_ring]
    surfaces = {
        'center_top': tri_grid(roof),
        'right_top': tri_grid(cap),
        'left_top': tri_grid(dome),
        'top_interface': fan((0.0, 0.0, ZF), riser_ring),
        'outlet': fan((0.0, YV, ZOUT), bottom_ring),
        'wall3': tri_grid([[vessel_ring[i], bottom_ring[i]] for i in range(len(vessel_ring))]),
        'center_bottom': ear_clip(edge_e + vessel_p[1:-1] + edge_w[::-1] + riser_m[::-1][1:-1]),
    }
    with open(out, 'w', newline='\n') as fh:
        for name, tris in surfaces.items():
            fh.write('solid %s\n' % name)
            for t in tris:
                fh.write('  facet normal 0 0 0\n    outer loop\n')
                for p in t:
                    fh.write('      vertex %.9f %.9f %.9f\n' % to_out(p))
                fh.write('    endloop\n  endfacet\n')
            fh.write('endsolid %s\n' % name)
    return {n: len(t) for n, t in surfaces.items()}


# ================================================================ dictionaries
HEAD = 'FoamFile\n{\n    version     2.0;\n    format      ascii;\n    class       dictionary;\n    object      %s;\n}\n'

CONTROL = HEAD % 'controlDict' + """application     snappyHexMesh;
startFrom       latestTime;
startTime       0;
stopAt          endTime;
endTime         1;
deltaT          1;
writeControl    timeStep;
writeInterval   1;
writeFormat     binary;
writePrecision  10;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""
SCHEMES = HEAD % 'fvSchemes' + """ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
"""
SOLUTION = HEAD % 'fvSolution'


def background_dict():
    lo, hi = (-19.75, -2.78, 15.62), (-0.25, 5.73, 21.02)
    n = [round((hi[k] - lo[k]) / TOP_BASE_CELL) for k in range(3)]
    v = [(x, y, z) for z in (lo[2], hi[2]) for (x, y) in ((lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1]))]
    return HEAD % 'blockMeshDict' + """// background mesh for snappyHexMesh (top part)
convertToMeters 1;
vertices ( %s );
blocks ( hex (0 1 2 3 4 5 6 7) (%d %d %d) simpleGrading (1 1 1) );
edges ();
boundary ( background { type patch; faces ((0 4 7 3) (1 2 6 5) (0 1 5 4) (3 7 6 2) (0 3 2 1) (4 5 6 7)); } );
""" % (' '.join('(%g %g %g)' % p for p in v), n[0], n[1], n[2])


FEATURES = HEAD % 'surfaceFeatureExtractDict' + """top.stl
{
    extractionMethod    extractFromSurface;
    extractFromSurfaceCoeffs
    {
        includedAngle   150;
        geometricTestOnly no;     // region boundaries (e.g. riser rim) are features too
    }
    subsetFeatures { nonManifoldEdges no; openEdges yes; }
    writeObj        no;
}
"""

SNAPPY = HEAD % 'snappyHexMeshDict' + """castellatedMesh true;
snap            true;
addLayers       true;

geometry
{
    top.stl
    {
        type triSurfaceMesh;
        name top;
        regions
        {
            center_top    { name center_top; }
            right_top     { name right_top; }
            left_top      { name left_top; }
            wall3         { name wall3; }
            outlet        { name outlet; }
            top_interface { name top_interface; }
            center_bottom { name center_bottom; }
        }
    }
    BOI      // body of influence from w_top.scdoc (around the floor plane z = 16.82)
    {
        type searchableBox;
        min (-20.0152878 -3.5251123 16.52);
        max ( -0.0152878  6.4748877 17.12);
    }
}

castellatedMeshControls
{
    maxLocalCells       5000000;
    maxGlobalCells      20000000;
    minRefinementCells  10;
    maxLoadUnbalance    0.10;
    nCellsBetweenLevels 3;
    features ( { file "top.eMesh"; level 1; } );
    refinementSurfaces
    {
        top
        {
            level (1 1);
            patchInfo { type wall; }
            regions
            {
                outlet        { level (1 1); patchInfo { type patch; } }
                top_interface { level (1 1); patchInfo { type patch; } }
            }
        }
    }
    resolveFeatureAngle 30;
    refinementRegions
    {
        BOI { mode inside; levels ((1e15 1)); }
    }
    locationInMesh (-8.755 1.475 17.5);
    allowFreeStandingZoneFaces true;
}

snapControls
{
    nSmoothPatch    3;
    tolerance       2.0;
    nSolveIter      50;
    nRelaxIter      5;
    nFeatureSnapIter 10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap true;
}

addLayersControls
{
    relativeSizes true;
    layers
    {
        "(%s)" { nSurfaceLayers 3; }
    }
    expansionRatio      1.2;
    finalLayerThickness 0.4;
    minThickness        0.1;
    nGrow               0;
    featureAngle        60;
    slipFeatureAngle    30;
    nRelaxIter          3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals      3;
    nSmoothThickness    10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedianAxisAngle  90;
    nBufferCellsNoExtrude 0;
    nLayerIter          50;
}

meshQualityControls
{
    maxNonOrtho         65;
    maxBoundarySkewness 4;
    maxInternalSkewness 3.5;
    maxConcave          80;
    minVol              1e-13;
    minTetQuality       1e-15;
    minArea             -1;
    minTwist            0.02;
    minDeterminant      0.001;
    minFaceWeight       0.05;
    minVolRatio         0.01;
    minTriangleTwist    -1;
    nSmoothScale        4;
    errorReduction      0.75;
    relaxed { maxNonOrtho 75; }
}

mergeTolerance 1e-6;
""" % WALLS_TOP

AMI = HEAD % 'createPatchDict' + """pointSync false;
patches
(
    { name riser_top_AMI; patchInfo { type cyclicAMI; matchTolerance 0.0001; neighbourPatch top_interface_AMI; transform noOrdering; }
      constructFrom patches; patches (riser_top); }
    { name top_interface_AMI; patchInfo { type cyclicAMI; matchTolerance 0.0001; neighbourPatch riser_top_AMI; transform noOrdering; }
      constructFrom patches; patches (top_interface); }
);
"""


# ================================================================ helpers
def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='\n') as fh:
        fh.write(text)


def foam(cmd, cwd, log):
    """Run an OpenFOAM command in cwd, output to cwd/log; stop on failure."""
    r = subprocess.call(['bash', '-c', 'source %s >/dev/null 2>&1; %s > %s 2>&1' % (FOAM_BASHRC, cmd, log)], cwd=cwd)
    if r != 0:
        sys.exit('FAILED: %s (see %s)' % (cmd, os.path.join(cwd, log)))


def grep(path, pattern):
    return [l.rstrip() for l in open(path, errors='replace') if re.search(pattern, l)]


def say(msg):
    print(msg, flush=True)


def set_bcs(case):
    """0/: wall group of this mesh and the AMI pair (idempotent)."""
    for f in sorted(os.listdir(os.path.join(case, '0'))):
        p = os.path.join(case, '0', f)
        if not os.path.isfile(p):
            continue
        s = open(p).read()
        if 'boundaryField' not in s:
            continue
        s2 = re.sub(r'"\([^"\n]*wall1[^"\n]*\)"', WALL_REGEX, s, count=1)
        if '_AMI' not in s2:
            end = s2.rindex('// ****') if '// ****' in s2 else len(s2)
            i = s2.rindex('}', 0, end)
            s2 = s2[:i] + '\n    ".*_AMI"\n    {\n        type            cyclicAMI;\n    }\n' + s2[i:]
        if s2 != s:
            open(p, 'w').write(s2)


# ================================================================ main
def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    opt = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--') and '=' in a)
    if len(args) != 1:
        sys.exit(__doc__)
    case = os.path.abspath(args[0])
    name = os.path.basename(case)
    preset = opt.get('preset', 'full')
    step = opt.get('step') or next((p for p in (os.path.join(HERE, 'w_top_fidelity.stp'),
                                                os.path.join(HERE, '..', 'w_top_fidelity.stp')) if os.path.isfile(p)), None)
    if not step or not os.path.isfile(step):
        sys.exit('STEP file not found, give --step=<w_top_fidelity.stp>')
    gen = os.path.join(HERE, 'gen_blockMeshDict.py')

    # ---- case folder
    if os.path.exists(os.path.join(case, 'constant', 'polyMesh')):
        sys.exit('%s already has constant/polyMesh - move it away first (nothing overwritten)' % case)
    if not os.path.isdir(case):
        shutil.copytree(opt.get('test-base', TEST_BASE), case)
        say('case        : new copy of %s' % opt.get('test-base', TEST_BASE))
    mg = os.path.join(case, 'meshgen')
    if os.path.exists(mg):
        sys.exit('%s exists - move it away first' % mg)
    os.makedirs(mg)
    for f in (os.path.abspath(__file__), gen, step):
        shutil.copy(f, mg)
    say('case        : %s  (preset %s)' % (case, preset))

    # ---- 1. riser (blockMesh)
    riser = os.path.join(mg, 'riser')
    out = subprocess.check_output([sys.executable, gen, riser, '--wtop', '--preset=' + preset]).decode()
    write(os.path.join(riser, 'system', 'controlDict'), CONTROL)
    write(os.path.join(riser, 'system', 'fvSchemes'), SCHEMES)
    write(os.path.join(riser, 'system', 'fvSolution'), SOLUTION)
    if os.path.exists(os.path.join(riser, 'blockInfo.txt')):
        os.remove(os.path.join(riser, 'blockInfo.txt'))
    foam('blockMesh', riser, 'log.blockMesh')
    say('1 riser     : %s' % [l for l in out.splitlines() if l.startswith('cells')][0])

    # ---- 2. top part (snappyHexMesh)
    top = os.path.join(mg, 'top')
    for f, text in (('controlDict', CONTROL), ('fvSchemes', SCHEMES), ('fvSolution', SOLUTION),
                    ('blockMeshDict', background_dict()), ('surfaceFeatureExtractDict', FEATURES),
                    ('snappyHexMeshDict', SNAPPY)):
        write(os.path.join(top, 'system', f), text)
    if 'reuse-top' in opt:
        src = os.path.join(os.path.abspath(opt['reuse-top']), 'constant', 'polyMesh')
        shutil.copytree(src, os.path.join(top, 'constant', 'polyMesh'))
        say('2 top       : reused %s' % src)
    else:
        stl = os.path.join(top, 'constant', 'triSurface', 'top.stl')
        os.makedirs(os.path.dirname(stl))
        n = write_top_stl(step, stl)
        say('2 top STL   : %d triangles (%s)' % (sum(n.values()), ', '.join('%s %d' % kv for kv in n.items())))
        foam('blockMesh', top, 'log.blockMesh')
        foam('surfaceFeatureExtract', top, 'log.surfaceFeatureExtract')
        say('2 top       : snappyHexMesh running (several minutes) ...')
        foam('snappyHexMesh -overwrite', top, 'log.snappyHexMesh')
        say('2 top       : %s' % grep(os.path.join(top, 'log.snappyHexMesh'), r'Layer mesh')[-1].strip())

    # ---- 3. join riser + top with cyclicAMI
    merged = os.path.join(mg, 'merged')
    shutil.copytree(os.path.join(riser, 'constant'), os.path.join(merged, 'constant'))
    for f, text in (('controlDict', CONTROL), ('fvSchemes', SCHEMES), ('fvSolution', SOLUTION), ('createPatchDict', AMI)):
        write(os.path.join(merged, 'system', f), text)
    foam('mergeMeshes -overwrite merged top', mg, 'log.mergeMeshes')
    foam('createPatch -overwrite', merged, 'log.createPatch')
    shutil.move(os.path.join(merged, 'constant', 'polyMesh'), os.path.join(case, 'constant', 'polyMesh'))
    shutil.rmtree(merged)
    say('3 joined    : riser_top_AMI <-> top_interface_AMI')

    # ---- 4. boundary names in 0/, checks
    set_bcs(case)
    subprocess.call(['bash', '-c', 'source %s >/dev/null 2>&1; checkMesh -allGeometry -allTopology > log.%s 2>&1' % (FOAM_BASHRC, name)], cwd=case)
    for l in grep(os.path.join(case, 'log.' + name),
                  r'cells:|hexahedra|polyhedra|prisms|Total volume|non-orthogonality Max|skewness|determinant|Concave|sum\(weights\)|Failed|Mesh OK'):
        say('4 checkMesh : ' + l.strip())
    ok = subprocess.call(['bash', '-c', 'source %s >/dev/null 2>&1; foamToVTK -time 0 -noInternal > log.fieldCheck 2>&1' % FOAM_BASHRC], cwd=case) == 0
    say('4 0/ fields : %s' % ('all fields read OK' if ok else 'ERROR, see log.fieldCheck'))
    for d in ('VTK', 'postProcessing'):
        shutil.rmtree(os.path.join(case, d), ignore_errors=True)
    if ok:
        os.remove(os.path.join(case, 'log.fieldCheck'))


if __name__ == '__main__':
    main()
