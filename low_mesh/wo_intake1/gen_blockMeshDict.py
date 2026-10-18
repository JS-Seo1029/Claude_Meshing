"""
blockMeshDict generator for the furnace riser, optionally with the intake
(w_intake_fidelity.stp): two plenums + pipes feeding 13 interleaved slots.

Usage:
    python gen_blockMeshDict.py <caseDir>             riser only (wo_intake)
    python gen_blockMeshDict.py <caseDir> --intake    riser + intake (w_intake)

Local frame: x = intake-pipe axis (air pipe at -x, gas pipe at +x),
             y = across the slots, z = riser axis (up). Converted to the
             global STEP frame on output.

Riser   : rectangular core split in 25 y-strips (13 slots + 12 lips) +
          O-grid ring, annular ring above the sudden expansion.
Channels: every slot is a 2D x-z profile (5 blocks) extruded over its strip.
Plenums : 3x3 H-grid whose centre cell is an O-grid "tube"; the tube starts
          as the pipe circle on the curved end wall and ends as a rectangle
          on the vertical face of the divider, so that the divider kinks
          (z=-3.55/-4.35) and all strip planes stay block boundaries.
Baffles : the V-slopes between a channel and the opposite plenum are
          zero-thickness walls, made with duplicated vertices.
"""
import math
import os
import sys

INTAKE = '--intake' in sys.argv
ARGS = [a for a in sys.argv[1:] if not a.startswith('--')]

# ================================================================ geometry
# frame taken from w_intake_fidelity.stp (metres)
CX, CY = -2.75528782867519, 1.47488765579627          # riser axis (x, y)
UX, UY = 0.994951279178985, 0.100359115480861         # intake pipe axis
ALIGN_X = True     # True: pipes along global x (CAD frame rotated by -5.76 deg
                   # about the riser axis, air pipe stays at -x); False: CAD frame
R1, R2 = 2.107, 2.26                                  # riser radii
Z0, Z1, Z2 = 0.0, 6.44, 16.82                         # inlet / step / outlet
A = 0.75                                              # slot half-length (x)
# slot / lip boundaries in y (13 slots: 7 x 0.20 gas, 6 x 0.32 air, lips 0.031)
Y = [-1.846, -1.646, -1.615, -1.295, -1.264, -1.064, -1.033, -0.713, -0.682,
     -0.482, -0.451, -0.131, -0.100, 0.100, 0.131, 0.451, 0.482, 0.682,
     0.713, 1.033, 1.064, 1.264, 1.295, 1.615, 1.646, 1.846]
B = Y[-1]
NSTRIP = len(Y) - 1


def strip_type(k):
    return ('gas', 'lip', 'air', 'lip')[k % 4]


# channel profile (xi = distance towards the strip's own plenum, z)
XI_EXT = 1.596973490568276      # end of slot extension (= plenum section S1)
XI_STEP = 0.927850474529371     # slope foot of the extension
Z_STEP = -1.76135983719114      # slope top
XI_V = 0.353638490568272        # half width of the V floor
Z_EXT, Z_PT, Z_V = -2.25, -2.7, -3.55     # extension top, plenum top, V floor
Z_KINK, Z_PB = -4.35, -5.2                # divider kink, plenum bottom
ZC_PIPE = -3.95                           # pipe axis height

SIDES = {
    # air plenum / pipe at -x
    'A': dict(sgn=-1, slot='air', R=0.9, xi_in=3.853638490568284,
              xc=0.7219793068134183, Rw=2.83138184275781, xi_bot=0.75,
              jlo=2, jhi=23, clo=7, chi=18, jstar=21,
              zH1=(-3.45, -4.45), r1=0.15, r3=0.12,
              Nc=16, Nring=8, N01=12, Npipe=30,
              wall='inlet_air_wall', path='inlet_air_path', inlet='inlet_air',
              chwall='inlet_air_sep_path'),
    # gas plenum / pipe at +x
    'G': dict(sgn=1, slot='gas', R=0.7, xi_in=4.353638490568243,
              xc=-0.47950469212568664, Rw=2.77839509218136,
              xi_bot=0.8269738866844714,
              jlo=0, jhi=25, clo=9, chi=16, jstar=19,
              zH1=(-3.5, -4.4), r1=0.12, r3=0.10,
              Nc=14, Nring=7, N01=12, Npipe=30,
              wall='inlet_gas_wall', path='inlet_gas_path', inlet='inlet_gas',
              chwall='inlet_gas_sep_path'),
}

# ================================================================ resolution
DS = 0.05          # in-plane target cell size in core / slots [m]
N_LIP = 2          # cells across a 0.031 m lip
NR1 = 16        # radial cells core -> R1
NR2 = 8        # radial cells R1 -> R2 (expansion annulus)
NZ1 = 80           # axial cells Z0 -> Z1
NZ2 = 105          # axial cells Z1 -> Z2 (keeps max dz ~0.1 m: thin wall
                   # cells with larger dz fail checkMesh cell determinant)
DW = 0.008        # first cell at walls (placeholder until y+ is fixed)
DSHEAR = 0.008        # cell size at r = R1 above the step (shear layer)
DZ_IN = 0.02       # axial cell at the riser bottom (z = 0)
DZ_STEP = 0.01     # axial cell at both sides of the step
DZ_OUT = 0.03      # axial cell at outlet (boundary cells must stay compact)
GROWTH = 1.2       # max geometric growth ratio inside clustering zones
# intake (uniform cells unless noted)
NB1 = 45           # channel, z from slope top to z = 0 (clustered to DZ_IN)
NB2 = 12           # channel, extension slope
NR = 11            # channel, extension height
NL = 21            # extension length = plenum length S1 -> divider
NV = 22            # V slope = plenum top row
NBOT = 18          # plenum bottom row
TMIN = 0.35        # min. plenum thickness end wall -> S1 below the top wall
DX_S1 = 0.01       # first cell after S1 towards the divider

# ================================================================ helpers
L = math.hypot(UX, UY)
COS, SIN = UX / L, UY / L
if ALIGN_X:          # undo the 5.76 deg CAD rotation about the riser axis
    COS, SIN = 1.0, 0.0


def to_global(x, y, z):
    return (CX + COS * x - SIN * y, CY + SIN * x + COS * y, z)


def cluster(length, n, s0=None, s1=None, g=GROWTH):
    """Multi-grading for an edge: geometric from s0 (begin) and s1 (end),
    uniform in between. Returns list of (lenFrac, cellFrac, ratio)."""
    best = None
    for n0 in (range(0, n) if s0 else [0]):
        l0 = s0 * (g ** n0 - 1) / (g - 1) if n0 else 0.0
        for n1 in (range(0, n - n0) if s1 else [0]):
            l1 = s1 * (g ** n1 - 1) / (g - 1) if n1 else 0.0
            nm, lm = n - n0 - n1, length - l0 - l1
            if nm < 1 or lm <= 0:
                continue
            h = lm / nm
            cost = 0.0
            for s, k in ((s0, n0), (s1, n1)):
                if s:
                    # k == 0: first cell is h instead of s; else junction mismatch
                    cost += abs(math.log(h / s)) * 4 if k == 0 else \
                        abs(math.log(s * g ** k / h))
            if best is None or cost < best[0]:
                best = (cost, n0, l0, n1, l1, nm, lm)
    if best is None:  # edge too short for requested clustering
        return [(1.0, 1.0, 1.0)]
    _, n0, l0, n1, l1, nm, lm = best
    secs = []
    if n0:
        secs.append((l0 / length, n0 / n, g ** (n0 - 1)))
    secs.append((lm / length, nm / n, 1.0))
    if n1:
        secs.append((l1 / length, n1 / n, g ** -(n1 - 1)))
    return secs


def reverse(secs):
    return [(lf, cf, 1.0 / r) for lf, cf, r in reversed(secs)]


def fmt_grading(secs):
    if len(secs) == 1 and abs(secs[0][2] - 1.0) < 1e-12:
        return "1"
    return "(" + " ".join("(%.6g %.6g %.6g)" % s for s in secs) + ")"


def dist(p, q):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q)))


# ================================================================ registry
V = {}          # vertex key -> local (x, y, z)
BLOCKS = []     # dict(keys, n, grading, region)
CURVE = {}      # frozenset(edge keys) -> ('arc', mid) | ('project', surfaces)
FPROJ = {}      # frozenset(face keys) -> (keys, surface)
GEOM = {}       # surface name -> searchableCylinder (local axis points, radius)

HEXFACES = [(0, 4, 7, 3), (1, 2, 6, 5), (0, 1, 5, 4),
            (3, 7, 6, 2), (0, 3, 2, 1), (4, 5, 6, 7)]
EDGES12 = [(0, 1), (3, 2), (7, 6), (4, 5), (0, 3), (1, 2),
           (5, 6), (4, 7), (0, 4), (1, 5), (2, 6), (3, 7)]


def vx(key, p):
    if key in V:
        assert dist(V[key], p) < 1e-7, (key, V[key], p)
    else:
        V[key] = tuple(p)
    return key


def hexvol(pts):
    vol = 0.0
    for f in HEXFACES:
        q = [pts[i] for i in f]
        c = [sum(p[i] for p in q) / 4 for i in range(3)]
        for a, b in zip(q, q[1:] + q[:1]):
            u = [a[i] - c[i] for i in range(3)]
            w = [b[i] - c[i] for i in range(3)]
            s = (u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2],
                 u[0] * w[1] - u[1] * w[0])
            vol += sum(c[i] * s[i] for i in range(3)) / 6
    return vol


STATION = {}    # intake vertex key -> plenum station ('S0', 'S1', 'S3', 'X')


def auto_grading(keys, n):
    """Uniform, except channel edges running up to z = 0 (match riser DZ_IN)
    and edges leaving section S1 towards the divider (DX_S1 at S1, to grow
    smoothly out of the thin wedge cells where the end wall meets S1)."""
    out = []
    for d, (i, j) in enumerate(EDGES12):
        p, q = V[keys[i]], V[keys[j]]
        tp, tq = STATION.get(keys[i]), STATION.get(keys[j])
        nn = n[d // 4]
        if abs(q[2] - p[2]) > 0.5 * dist(p, q) and (abs(p[2]) < 1e-9 or abs(q[2]) < 1e-9):
            secs = cluster(dist(p, q), nn, s1=DZ_IN)
            out.append(fmt_grading(secs if abs(q[2]) < 1e-9 else reverse(secs)))
        elif tp in ('S0', 'S1') and tq in ('S3', 'X'):
            out.append(fmt_grading(cluster(dist(p, q), nn, s0=DX_S1)))
        elif tq in ('S0', 'S1') and tp in ('S3', 'X'):
            out.append(fmt_grading(reverse(cluster(dist(p, q), nn, s0=DX_S1))))
        else:
            out.append("1")
    return out


def add_block(qa, qb, n1, n2, n3, region, grading=None):
    """Hex from cross-section quad qa (station a) and qb (station b)."""
    keys = list(qa) + list(qb)
    if hexvol([V[k] for k in keys]) < 0:
        assert grading is None, ("left-handed block with explicit grading", region)
        qa = [qa[0], qa[3], qa[2], qa[1]]
        qb = [qb[0], qb[3], qb[2], qb[1]]
        keys = list(qa) + list(qb)
        n1, n2 = n2, n1
    n = (n1, n2, n3)
    BLOCKS.append(dict(keys=keys, n=n, region=region,
                       grading=grading or auto_grading(keys, n)))


# ================================================================ riser
# corner diagonal: 45 deg outward from core corner (a, b) to circle R1
t = (-(A + B) + math.sqrt((A + B) ** 2 - 2 * (A * A + B * B - R1 * R1))) / 2
TH_C = math.atan2(B + t, A + t)

nodes = {}   # name -> (x, y)      ('C', i, s) core, ('R1'|'R2', side, i) circle
ang = {}     # circle node name -> angle
for i, y in enumerate(Y):
    nodes[('C', i, 1)] = (A, y)
    nodes[('C', i, -1)] = (-A, y)
    th = TH_C * y / B
    for rk, r in (('R1', R1), ('R2', R2)):
        for side, a in (('E', th), ('W', math.pi - th)):
            nodes[(rk, side, i)] = (r * math.cos(a), r * math.sin(a))
            ang[(rk, side, i)] = a

ny = [max(N_LIP, round((Y[i + 1] - Y[i]) / DS)) if Y[i + 1] - Y[i] > 0.05
      else N_LIP for i in range(NSTRIP)]
NX = round(2 * A / DS)
last = NSTRIP

# faces: (n0, n1, n2, n3) counter-clockwise from +z, dims (along n0->n1, n0->n3)
inner_faces = []
for i in range(last):
    inner_faces.append((('C', i, -1), ('C', i, 1), ('C', i + 1, 1), ('C', i + 1, -1)))
    inner_faces.append((('C', i, 1), ('R1', 'E', i), ('R1', 'E', i + 1), ('C', i + 1, 1)))
    inner_faces.append((('R1', 'W', i), ('C', i, -1), ('C', i + 1, -1), ('R1', 'W', i + 1)))
inner_faces.append((('C', last, -1), ('C', last, 1), ('R1', 'E', last), ('R1', 'W', last)))
inner_faces.append((('R1', 'W', 0), ('R1', 'E', 0), ('C', 0, 1), ('C', 0, -1)))

# R1 circle nodes in CCW order -> annulus faces
ring_seq = [('R1', 'E', i) for i in range(last + 1)] + \
           [('R1', 'W', i) for i in range(last, -1, -1)]
ring_faces = []
for k in range(len(ring_seq)):
    p, q = ring_seq[k], ring_seq[(k + 1) % len(ring_seq)]
    ring_faces.append((p, ('R2',) + p[1:], ('R2',) + q[1:], q))


def kind(n):
    return n[0]


def edge_cells(p, q):
    kp, kq = kind(p), kind(q)
    if kp != kq:                                   # radial-type edge
        return NR2 if {kp, kq} == {'R1', 'R2'} else NR1
    if kp == 'C':
        return NX if p[1] == q[1] else ny[min(p[1], q[1])]
    # arc along a circle
    if p[1] == q[1]:
        return ny[min(p[2], q[2])]
    return NX                                      # N / S arcs


def edge_grading(p, q):
    kp, kq = kind(p), kind(q)
    (xp, yp), (xq, yq) = nodes[p], nodes[q]
    length = math.hypot(xq - xp, yq - yp)
    n = edge_cells(p, q)
    if kp == kq:
        return [(1.0, 1.0, 1.0)]
    if {kp, kq} == {'C', 'R1'}:
        secs = cluster(length, n, s0=min(DS, length / n), s1=DW)
        return secs if kp == 'C' else reverse(secs)
    secs = cluster(length, n, s0=DSHEAR, s1=DW)    # R1 -> R2
    return secs if kp == 'R1' else reverse(secs)


def is_arc(p, q):
    return kind(p) == kind(q) and kind(p) in ('R1', 'R2')


def rkey(n, z):
    return vx(('R', n, z), nodes[n] + (z,))


sections = [
    # zlo, zhi, nz, z-grading, faces, region
    (Z0, Z1, NZ1, cluster(Z1 - Z0, NZ1, DZ_IN, DZ_STEP), inner_faces, 'riser_lo'),
    (Z1, Z2, NZ2, cluster(Z2 - Z1, NZ2, DZ_STEP, DZ_OUT), inner_faces + ring_faces, 'riser_hi'),
]
for zlo, zhi, nz, zg, faces, region in sections:
    for f in faces:
        lo = [rkey(n, zlo) for n in f]
        hi = [rkey(n, zhi) for n in f]
        n1, n2 = edge_cells(f[0], f[1]), edge_cells(f[0], f[3])
        assert edge_cells(f[3], f[2]) == n1 and edge_cells(f[1], f[2]) == n2, f
        g = [edge_grading(f[0], f[1]), edge_grading(f[3], f[2]),
             edge_grading(f[3], f[2]), edge_grading(f[0], f[1]),
             edge_grading(f[0], f[3]), edge_grading(f[1], f[2]),
             edge_grading(f[1], f[2]), edge_grading(f[0], f[3]),
             zg, zg, zg, zg]
        strip = f[0][1] if kind(f[0]) == 'C' and kind(f[1]) == 'C' else None
        add_block(lo, hi, n1, n2, nz, (region, strip), [fmt_grading(s) for s in g])
        for k in range(4):
            p, q = f[k], f[(k + 1) % 4]
            if is_arc(p, q):
                r = R1 if kind(p) == 'R1' else R2
                d = (ang[q] - ang[p]) % (2 * math.pi)
                if d > math.pi:
                    d -= 2 * math.pi
                m = ang[p] + d / 2
                for z in (zlo, zhi):
                    CURVE[frozenset((rkey(p, z), rkey(q, z)))] = \
                        ('arc', (r * math.cos(m), r * math.sin(m), z))


# ================================================================ intake
def cum_frac(lo, hi):
    """Cell-count fraction of every strip boundary j in [lo, hi]."""
    tot = sum(ny[lo:hi])
    out, acc = {}, 0
    for j in range(lo, hi + 1):
        out[j] = acc / tot
        if j < hi:
            acc += ny[j]
    return out


def own_boundary(c, j):
    """Strip boundary j touches a slot fed by this side's plenum."""
    return any(0 <= k < NSTRIP and strip_type(k) == c['slot'] for k in (j - 1, j))


def build_side(name):
    c = SIDES[name]
    sgn, R = c['sgn'], c['R']
    jlo, jhi, clo, chi = c['jlo'], c['jhi'], c['clo'], c['chi']
    s45 = math.sqrt(0.5)
    y45, c0 = R * s45, 0.42 * R
    fr = cum_frac(clo, chi)

    def xi_end(y):
        return sgn * c['xc'] + math.sqrt(c['Rw'] ** 2 - y * y)

    def ymap0(j):
        """y of strip plane j on the end wall (S0): outer planes kept clear of
        the pipe circle, planes inside the tube spread by cell count."""
        if clo <= j <= chi:
            return -y45 + 2 * y45 * fr[j]
        s, jj = (1, j) if j > chi else (-1, jlo + jhi - j)   # mirror to the right
        yb = Y[jj] if j > chi else -Y[j]
        js = c['jstar']
        if jj >= js:
            return s * yb
        y_f = R + 0.08
        y0, y1 = Y[chi + 1], Y[js]
        return s * (y_f + (yb - y0) / (y1 - y0) * (Y[js] - y_f))

    def pos(st, kd, j):
        """(xi, y, z) of node kind kd on strip plane j at station st."""
        if st in ('S0', 'IN'):
            if kd in ('Ht', 'Hb') and clo <= j <= chi:          # pipe circle
                th = (135 - 90 * fr[j]) if kd == 'Ht' else (225 + 90 * fr[j])
                y = R * math.cos(math.radians(th))
                z = ZC_PIPE + R * math.sin(math.radians(th))
            elif kd in ('Ct', 'Cb'):
                y = -c0 + 2 * c0 * fr[j]
                z = ZC_PIPE + (c0 if kd == 'Ct' else -c0)
            else:
                y = ymap0(j)
                z = {'T': Z_PT, 'B': Z_PB, 'Ht': ZC_PIPE + R * s45,
                     'Hb': ZC_PIPE - R * s45}[kd]
            xi = xi_end(y) if st == 'S0' else c['xi_in']
            return xi, y, z
        zt, zb = c['zH1'] if st == 'S1' else (Z_V, Z_KINK)
        r = c['r1'] if st == 'S1' else c['r3']
        if kd in ('Ct', 'Cb'):
            y = Y[clo] + r + (Y[chi] - Y[clo] - 2 * r) * fr[j]
            z = zt - r if kd == 'Ct' else zb + r
        else:
            y = Y[j]
            z = {'T': Z_PT, 'B': Z_PB, 'Ht': zt, 'Hb': zb}[kd]
        if st == 'S1':
            # S1 must sit at the slot-extension end only on the top wall; below
            # it, lean towards the divider near the side walls, where the end
            # wall comes close to S1 (zone S0->S1 keeps >= TMIN thickness)
            xi = XI_EXT
            if kd != 'T':
                xi -= max(0.0, TMIN - (xi_end(y) - XI_EXT))
        else:
            xi = {'T': A, 'B': c['xi_bot']}.get(kd, XI_V)
        return xi, y, z

    def key(st, kd, j):
        """Vertex key with aliases: the top side-wall corners collapse S1 onto
        S0 (end wall meets the slot extension end there), and strip planes of
        this side's slots share the channel vertices (open connection)."""
        if st == 'S1' and j in (jlo, jhi) and kd == 'T':
            st = 'S0'
        if st == 'S1' and kd == 'T' and own_boundary(c, j):
            k = ('ch', name, 'b0', j)
        elif st == 'S3' and kd == 'T' and own_boundary(c, j):
            k = ('ch', name, 'b1', j)
        elif st == 'S3' and kd == 'Ht' and own_boundary(c, j):
            k = ('ch', name, 'v0', j)
        else:
            k = ('pl', name, st, kd, j)
        xi, y, z = pos(st, kd, j)
        STATION[k] = st
        return vx(k, (sgn * xi, y, z))

    # ---- curved surfaces
    GEOM['end' + name] = (((c['xc'], 0, -30), (c['xc'], 0, 30)), c['Rw'])
    GEOM['pipe' + name] = (((-30, 0, ZC_PIPE), (30, 0, ZC_PIPE)), R)

    # ---- plenum and pipe blocks
    zones = [('S0', 'S1', c['N01'], ('pl', name)), ('S1', 'S3', NL, ('pl', name))]
    for a, b, nx, region in zones + [('IN', 'S0', c['Npipe'], ('pipe', name))]:
        pipe = a == 'IN'

        def q(kinds_j):
            return [key(a, kd, j) for kd, j in kinds_j], [key(b, kd, j) for kd, j in kinds_j]

        for k in range(jlo, jhi):
            centre = clo <= k < chi
            if pipe and not centre:
                continue
            rows = []
            if not pipe:
                rows.append(((('Ht', k), ('Ht', k + 1), ('T', k + 1), ('T', k)), NV))
            if centre:
                rows += [((('Ct', k), ('Ct', k + 1), ('Ht', k + 1), ('Ht', k)), c['Nring']),
                         ((('Cb', k), ('Cb', k + 1), ('Ct', k + 1), ('Ct', k)), c['Nc']),
                         ((('Hb', k), ('Hb', k + 1), ('Cb', k + 1), ('Cb', k)), c['Nring'])]
            else:
                rows.append(((('Hb', k), ('Hb', k + 1), ('Ht', k + 1), ('Ht', k)), c['Nc']))
            if not pipe:
                rows.append(((('B', k), ('B', k + 1), ('Hb', k + 1), ('Hb', k)), NBOT))
            for quad, nz in rows:
                qa, qb = q(quad)
                add_block(qa, qb, ny[k], nz, nx, region)
        for quad in ((('Cb', chi), ('Hb', chi), ('Ht', chi), ('Ct', chi)),
                     (('Hb', clo), ('Cb', clo), ('Ct', clo), ('Ht', clo))):
            qa, qb = q(quad)
            add_block(qa, qb, c['Nring'], c['Nc'], nx, region)

    # ---- channels of this side's slots
    prof = {'t0': (A, 0.0), 't1': (-A, 0.0), 'm0': (A, Z_STEP), 'm1': (-A, Z_STEP),
            's0': (XI_STEP, Z_EXT), 's1': (-A, Z_EXT), 'e0': (XI_EXT, Z_EXT),
            'b0': (XI_EXT, Z_PT), 'b1': (A, Z_PT), 'b2': (-A, Z_PT),
            'v0': (XI_V, Z_V), 'v1': (-XI_V, Z_V)}

    def ckey(p, j):
        if p in ('t0', 't1'):
            s = sgn if p == 't0' else -sgn
            return rkey(('C', j, s), Z0)
        if p == 'b0' and j in (jlo, jhi):
            return key('S1', 'T', j)                     # collapsed corner
        xi, z = prof[p]
        if p == 'e0' and j in (jlo, jhi):
            xi = xi_end(Y[j])                            # follow the end wall
        k = ('ch', name, p, j)
        if p in ('e0', 's0'):
            STATION[k] = 'S1' if p == 'e0' else 'X'      # extension length edges
        return vx(k, (sgn * xi, Y[j], z))

    blocks = [(('m0', 'm1', 't1', 't0'), NX, NB1), (('s0', 's1', 'm1', 'm0'), NX, NB2),
              (('b1', 'b2', 's1', 's0'), NX, NR), (('b0', 'b1', 's0', 'e0'), NL, NR),
              (('v0', 'v1', 'b2', 'b1'), NX, NV)]
    for k in range(NSTRIP):
        if strip_type(k) != c['slot']:
            continue
        for quad, n1, n2 in blocks:
            add_block([ckey(p, k) for p in quad], [ckey(p, k + 1) for p in quad],
                      n1, n2, ny[k], ('ch', name))

    # ---- side metadata for curved edges / patches
    c['xi_end'] = xi_end


if INTAKE:
    build_side('A')
    build_side('G')

# ================================================================ topology
# boundary faces: block faces not shared by two blocks (degenerate skipped)
face_use = {}
for bi, blk in enumerate(BLOCKS):
    for f in HEXFACES:
        ks = [blk['keys'][i] for i in f]
        if len(set(ks)) < 3:
            continue
        face_use.setdefault(frozenset(ks), []).append((bi, ks))

# cell count consistency on shared edges
edge_n = {}
for blk in BLOCKS:
    for d, (i, j) in enumerate(EDGES12):
        a, b = blk['keys'][i], blk['keys'][j]
        if a == b:
            continue
        e = frozenset((a, b))
        n = blk['n'][d // 4]
        assert edge_n.setdefault(e, n) == n, ("cell count mismatch", a, b, edge_n[e], n)


def classify(blk, ks):
    region = blk['region']
    pts = [V[k] for k in ks]
    zs = [p[2] for p in pts]
    if region[0] in ('riser_lo', 'riser_hi'):
        if max(abs(z - Z0) for z in zs) < 1e-9:
            strip = region[1]
            if not INTAKE and strip is not None:
                return {'gas': 'inlet_gas', 'air': 'inlet_air'}.get(strip_type(strip), 'fluid:1')
            return 'fluid:1'
        if max(abs(z - Z2) for z in zs) < 1e-9:
            return 'outlet'
        if max(abs(z - Z1) for z in zs) < 1e-9:
            return 'gap1'
        return 'wall1' if region[0] == 'riser_lo' else 'wall2'
    c = SIDES[region[1]]
    if region[0] == 'ch':
        return c['chwall']
    if region[0] == 'pipe':
        if all(k[0] == 'pl' and k[2] == 'IN' for k in ks):
            return c['inlet']
        return c['path']
    return c['wall']


patches = {}
for fk, uses in face_use.items():
    assert len(uses) <= 2, "face shared by more than two blocks"
    if len(uses) == 2:
        continue
    bi, ks = uses[0]
    patches.setdefault(classify(BLOCKS[bi], ks), []).append(ks)

# curved edges / projected faces of the intake
if INTAKE:
    for name, c in SIDES.items():
        R, sgn = c['R'], c['sgn']

        def on_end(p):
            return abs(sgn * p[0] - c['xi_end'](p[1])) < 1e-9

        def on_pipe(p):
            return abs(math.hypot(p[1], p[2] - ZC_PIPE) - R) < 1e-9

        def end_arc_mid(p, q):
            xc = c['xc']
            a1 = math.atan2(p[1], p[0] - xc)
            a2 = math.atan2(q[1], q[0] - xc)
            m = a1 + ((a2 - a1 + math.pi) % (2 * math.pi) - math.pi) / 2
            return (xc + c['Rw'] * math.cos(m), c['Rw'] * math.sin(m), p[2])

        for ks_list, pname in ((patches.get(c['wall'], []), 'end'),
                               (patches.get(c['path'], []), 'pipe'),
                               (patches.get(c['inlet'], []), 'inlet')):
            for ks in ks_list:
                pts = [V[k] for k in ks]
                if pname == 'end' and not all(on_end(p) for p in pts):
                    continue
                if pname == 'pipe':
                    FPROJ[frozenset(ks)] = (ks, 'pipe' + name)
                elif pname == 'end':
                    FPROJ[frozenset(ks)] = (ks, 'end' + name)
                for a, b in zip(ks, ks[1:] + ks[:1]):
                    if a == b:
                        continue
                    p, q = V[a], V[b]
                    e = frozenset((a, b))
                    if on_end(p) and on_end(q):
                        if on_pipe(p) and on_pipe(q):
                            CURVE[e] = ('project', ['end' + name, 'pipe' + name])
                        elif abs(p[2] - q[2]) < 1e-9:
                            CURVE[e] = ('arc', end_arc_mid(p, q))
                        elif abs(p[1] - q[1]) > 1e-9:
                            CURVE[e] = ('project', ['end' + name])
                    elif on_pipe(p) and on_pipe(q) and abs(p[0] - q[0]) < 1e-9:
                        th1 = math.atan2(p[2] - ZC_PIPE, p[1])
                        th2 = math.atan2(q[2] - ZC_PIPE, q[1])
                        d = (th2 - th1 + math.pi) % (2 * math.pi) - math.pi
                        m = th1 + d / 2
                        CURVE[e] = ('arc', (p[0], R * math.cos(m), ZC_PIPE + R * math.sin(m)))

# ================================================================ write
case = ARGS[0] if ARGS else os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(case, 'system'), exist_ok=True)
idx = {}
for k in V:
    idx[k] = len(idx)
PATCH_TYPES = {'inlet_gas': 'patch', 'inlet_air': 'patch', 'outlet': 'patch'}


def g3(p):
    return "(%.10f %.10f %.10f)" % to_global(*p)


with open(os.path.join(case, 'system', 'blockMeshDict'), 'w', newline='\n') as fh:
    fh.write("FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
             "    class       dictionary;\n    object      blockMeshDict;\n}\n"
             "// generated by gen_blockMeshDict.py%s - do not edit by hand\n\n"
             "convertToMeters 1;\n\n" % (" --intake" if INTAKE else ""))
    if GEOM:
        fh.write("geometry\n{\n")
        for gname, ((p1, p2), r) in GEOM.items():
            fh.write("    %s\n    {\n        type searchableCylinder;\n"
                     "        point1 %s;\n        point2 %s;\n        radius %.12g;\n    }\n"
                     % (gname, g3(p1), g3(p2), r))
        fh.write("}\n\n")
    fh.write("vertices\n(\n")
    for k, i in idx.items():
        fh.write("    %s // %d\n" % (g3(V[k]), i))
    fh.write(");\n\nblocks\n(\n")
    for blk in BLOCKS:
        fh.write("    hex (%s) (%d %d %d) edgeGrading (%s)\n" % (
            " ".join(str(idx[k]) for k in blk['keys']), *blk['n'], " ".join(blk['grading'])))
    fh.write(");\n\nedges\n(\n")
    for e, spec in CURVE.items():
        a, b = sorted(e, key=lambda k: idx[k])
        if spec[0] == 'arc':
            fh.write("    arc %d %d %s\n" % (idx[a], idx[b], g3(spec[1])))
        else:
            fh.write("    project %d %d (%s)\n" % (idx[a], idx[b], " ".join(spec[1])))
    fh.write(");\n\nfaces\n(\n")
    for ks, surf in FPROJ.values():
        fh.write("    project (%s) %s\n" % (" ".join(str(idx[k]) for k in ks), surf))
    fh.write(");\n\nboundary\n(\n")
    for name in sorted(patches):
        fh.write("    %s\n    {\n        type %s;\n        faces\n        (\n"
                 % (name, PATCH_TYPES.get(name, 'wall')))
        for ks in patches[name]:
            fh.write("            (%s)\n" % " ".join(str(idx[k]) for k in ks))
        fh.write("        );\n    }\n")
    fh.write(");\n\nmergePatchPairs\n(\n);\n")

ncells = sum(b['n'][0] * b['n'][1] * b['n'][2] for b in BLOCKS)
print("mode          : %s" % ("riser + intake" if INTAKE else "riser"))
print("blocks        : %d, vertices %d, curved edges %d, projected faces %d"
      % (len(BLOCKS), len(V), len(CURVE), len(FPROJ)))
print("patches       : %s" % ", ".join("%s(%d)" % (k, len(v)) for k, v in sorted(patches.items())))
print("cells         : %d (estimate, collapsed blocks counted as hex)" % ncells)

# debug aid: block index -> region / dims / first vertex keys
with open(os.path.join(case, 'blockInfo.txt'), 'w') as fh:
    for i, blk in enumerate(BLOCKS):
        fh.write("%d %s %s %s\n" % (i, blk['region'], blk['n'], blk['keys'][:4]))
