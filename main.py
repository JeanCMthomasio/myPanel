import jax
import jax.numpy as np

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')

class Surface():
    """
    One lifting surface, built as a chain of trapezoidal segments.

    Stability axes throughout: x+ forward (flight direction), y+ right, z+ down.

    span            spanwise breakpoint stations [m], len = n_segments+1, root first
    chord           chord at each breakpoint [m], same length as span
    sweep           per segment [deg], positive = swept aft
    dihedral        per segment [deg], positive = tip up
    position        (x,y,z) of the root three-quarter-chord point [m] -- the LE sits 0.75*chord ahead of it, the TE 0.25*chord behind
    discretization  per segment (n, m): n nodes along the chord, m along the span. n must be the same in every segment
    symmetry        mirror about y=0; requires the root to sit on that plane
    control_surfaces  list of dicts: name, hinge (x/c from the LE), span (pair of semi-span fractions), mode ('symmetric'/'antisymmetric'), gain (optional multiplier)
    """
    def __init__(self, span, chord, sweep, dihedral, position, discretization, symmetry, control_surfaces=None):
        self.span = span
        self.chord = chord
        self.sweep = sweep
        self.position = position
        self.dihedral = dihedral
        self.discretization = discretization
        self.symmetry = symmetry
        self.control_surfaces = control_surfaces if control_surfaces is not None else []
        self.S = 0.0
        self.MAC = 0.0
        self.b = 0.0
        self.nodes = []
        self.aero_centers = []
        self.collocation = []
        self.normals = []
        self.vortex_a = []
        self.vortex_b = []
    
    def transform(self, x, y, span, chord_root, chord_tip, sweep, dihedral, root):
        # (x, y) are parametric: x = 0 at the LE, 1 at the TE; y = 0 at the segment
        # root, 1 at its tip. `root` carries the LE reference where the previous
        # segment stopped, so segments stay attached instead of each restarting.
        # Swept aft is -x and tip-up is -z in these axes, hence the minus signs.
        x_root, y_root, z_root = root
        x_tip = x_root - span * np.tan(np.deg2rad(sweep))
        y_tip = y_root + span * np.cos(np.deg2rad(dihedral))
        z_tip = z_root - span * np.sin(np.deg2rad(dihedral))

        x = 0.75 - x   # LE (input 0) -> +0.75*chord (forward); TE (input 1) -> -0.25*chord (aft)
        X = x * (chord_root * (1 - y) + chord_tip * y) + x_root + y * (x_tip - x_root) + self.position[0]
        Y = y_root + y * (y_tip - y_root) + self.position[1]
        Z = z_root + y * (z_tip - z_root) + self.position[2]
        return X, Y, Z, (x_tip, y_tip, z_tip)

    def mirror(self, X, Y, Z, s):
        if abs(self.span[0] + self.position[1]) > 1e-12:
            raise ValueError("symmetry needs the root on the plane Y=0")
        # keep root only once
        keep = slice(None, -1)
        X = np.concat([ X[:, ::-1][:, keep], X], axis=1)
        Y = np.concat([-Y[:, ::-1][:, keep], Y], axis=1)
        Z = np.concat([ Z[:, ::-1][:, keep], Z], axis=1)
        s = np.concat([-s[::-1][keep], s])
        return X, Y, Z, s

    def getChordPoint(self, nodes, fraction):
        LE = 0.5*(nodes[:-1,:-1,:] + nodes[:-1,1:,:])
        TE = 0.5*(nodes[ 1:,:-1,:] + nodes[ 1:,1:,:])
        return LE + fraction*(TE - LE)

    def getAeroCenter(self, nodes):
        return self.getChordPoint(nodes, 0.25)

    def getCollocationPoint(self, nodes):
        return self.getChordPoint(nodes, 0.75)

    def getVortexPoints(self, nodes):
        quarter_chord = 0.25*(3*nodes[:-1,:,:] + nodes[1:,:,:])
        return quarter_chord[:,:-1,:], quarter_chord[:,1:,:]

    def getControlNodeMask(self, name):
        active = self.control_gains[name] != 0.0
        padded = np.pad(active, ((0,1),(0,1)))
        return padded | np.roll(padded,1,axis=0) | np.roll(padded,1,axis=1) \
                       | np.roll(np.roll(padded,1,axis=0),1,axis=1)

    def getControlGains(self):
        # per-control deflection gain, one (nc, ns) array per name in
        # self.control_surfaces. n is the same across segments (generateMesh's
        # axis=1 concatenation already requires it), so one chordwise station
        # vector covers the whole surface.
        x = np.linspace(0.0, 1.0, self.discretization[0][0])
        gains = {}
        for c in self.control_surfaces:
            hinge = c['hinge']
            s0, s1 = c['span']
            mode = c.get('mode', 'symmetric')
            gain = c.get('gain', 1.0)
            # fraction of each chordwise panel that sits aft of the hinge line
            aft = np.clip((x[1:] - hinge) / (x[1:] - x[:-1]), 0.0, 1.0)
            inside = (np.abs(self.span_station) >= s0) & (np.abs(self.span_station) <= s1)
            sign = np.where(self.span_station < 0, -1.0, 1.0) if mode == 'antisymmetric' else np.ones_like(self.span_station)
            gains[c['name']] = gain * aft[:,None] * (inside * sign)[None,:]
        return gains

    def getReferences(self):
        integral = half_area = 0.0
        for span_root, span_tip, chord_root, chord_tip in zip(self.span[:-1], self.span[1:], self.chord[:-1], self.chord[1:]):
            dy = span_tip - span_root
            integral  += dy/3 * (chord_root**2 + chord_root*chord_tip + chord_tip**2)
            half_area += dy * (chord_root + chord_tip)/2
        span = (2 if self.symmetry else 1) * (self.span[-1] - self.span[0])
        return integral/half_area, span

    def getNormalAndArea(self, nodes):
        diag1 = nodes[ 1:,1:,:] - nodes[:-1,:-1,:]
        diag2 = nodes[:-1,1:,:] - nodes[ 1:,:-1,:]
        normal = np.cross(diag1, diag2)
        dS =  np.linalg.norm(normal, axis = -1, keepdims = True)
        return normal / dS, 0.5*dS
    
    def generateMesh(self):
        X_segments, Y_segments, Z_segments , station_segments= [], [], [], []
        root = (0.0, self.span[0], 0.0)
        for i, (span_root, span_tip, chord_root, chord_tip, sweep, dihedral, (n,m)) in enumerate(zip(self.span[:-1], self.span[1:], self.chord[:-1], self.chord[1:], self.sweep, self.dihedral, self.discretization)):
            span_nodes = np.linspace(0.0, 1.0, m)
            chord_nodes = np.linspace(0.0, 1.0, n)
            span_mesh, chord_mesh = np.meshgrid(span_nodes, chord_nodes)
            X, Y, Z, root = self.transform(chord_mesh, span_mesh, span_tip - span_root, chord_root, chord_tip, sweep, dihedral, root)
            # consecutive segments share their interface column: keep it once
            shared = slice(None) if i == 0 else slice(1, None)
            X_segments.append(X[:, shared])
            Y_segments.append(Y[:, shared])
            Z_segments.append(Z[:, shared])
            station_segments.append(np.linspace(span_root, span_tip, m)[shared])

        X = np.concatenate(X_segments, axis=1)
        Y = np.concatenate(Y_segments, axis=1)
        Z = np.concatenate(Z_segments, axis=1)
        s = np.concatenate(station_segments)

        if self.symmetry:
            X, Y, Z, s = self.mirror(X, Y, Z, s)

        self.nodes = np.stack((X, Y, Z), axis=-1)
        self.aero_centers = self.getAeroCenter(self.nodes)
        self.collocation = self.getCollocationPoint(self.nodes)
        self.normals, dS = self.getNormalAndArea(self.nodes)
        self.S = dS.sum()
        self.MAC, self.b = self.getReferences()
        self.vortex_a, self.vortex_b = self.getVortexPoints(self.nodes)
        self.span_station = 0.5*(s[:-1] + s[1:]) / self.span[-1]

        dl = self.vortex_b - self.vortex_a
        self.hinge_axis = dl / np.linalg.norm(dl, axis=-1, keepdims=True)
        self.control_gains = self.getControlGains()

    def plot(self, fig_ax=None, plot_aero = False, plot_collocation = False, plot_control_surfaces = False, control_colors = None):

        if fig_ax==None:
            fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
        else:
            fig, ax = fig_ax

        ax.plot_wireframe(*np.moveaxis(self.nodes, -1, 0), rstride=1, cstride=1, color='0.75', linewidth=0.6)

        if plot_control_surfaces:
            colors = control_colors
            if colors is None:
                palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
                colors = {name: palette[i % len(palette)] for i, name in enumerate(self.control_gains)}
            for name in self.control_gains:
                active = self.getControlNodeMask(name)
                if not bool(np.any(active)):
                    continue
                highlighted = np.where(active[...,None], self.nodes, np.nan)
                ax.plot_wireframe(*np.moveaxis(highlighted, -1, 0), rstride=1, cstride=1, linewidth=1.6, color=colors[name], label=name)

        if plot_aero:
            ax.scatter(*np.moveaxis(self.aero_centers, -1, 0))
        if plot_collocation:
            ax.scatter(*np.moveaxis(self.collocation, -1, 0))
        
        return fig, ax
            
class Aircraft():
    """
    Vortex lattice solver over a set of Surfaces (Drela, Flight Vehicle Aerodynamics, ch. 6). 
    Same stability axes as Surface: x+ forward, y+ right, z+ down.

    surfaces  list of Surface; their panels are concatenated into one global system,
              and reference S/MAC/span are taken from the largest surface by area
    CG        (x,y,z) moment reference point [m]

    Angles are in degrees at every boundary: alpha, beta and the control deflections
    passed to simulate/coefficientsAt. Velocities are non-dimensionalised by V_inf, so
    the returned coefficients and the AD derivatives are per degree of deflection.
    """
    def __init__(self, surfaces, CG):
        self.surfaces = surfaces
        self.CG = CG
        self.nodes  = []
        self.aero_centers = []
        self.collocation = []
        self.normals = []
        self.vortex_a = []
        self.vortex_b = []
        self.hinge_axis = []
        self.control_gain = []
        self.control_names = []
        self.generateMesh()  

    def generateMesh(self):
        nodes, aero_centers, collocation, normals = [], [], [], []
        vortex_a, vortex_b, hinge_axis, per_surface_gains = [], [], [], []
        for surface in self.surfaces:
            surface.generateMesh()
            nodes.append(surface.nodes.reshape(-1, 3))
            aero_centers.append(surface.aero_centers.reshape(-1, 3))
            collocation.append(surface.collocation.reshape(-1, 3))
            normals.append(surface.normals.reshape(-1, 3))
            vortex_a.append(surface.vortex_a.reshape(-1, 3))
            vortex_b.append(surface.vortex_b.reshape(-1, 3))
            hinge_axis.append(surface.hinge_axis.reshape(-1, 3))
            per_surface_gains.append(surface.control_gains)

        self.nodes = np.concat(nodes)
        self.aero_centers = np.concat(aero_centers)
        self.collocation = np.concat(collocation)
        self.normals = np.concat(normals)
        self.vortex_a = np.concat(vortex_a)
        self.vortex_b = np.concat(vortex_b)
        self.hinge_axis = np.concat(hinge_axis)

        # global, deterministic order: first appearance while walking the surfaces
        names = []
        for gains in per_surface_gains:
            for name in gains:
                if name not in names:
                    names.append(name)
        self.control_names = names

        rows = []
        for name in names:
            parts = []
            for surface, gains in zip(self.surfaces, per_surface_gains):
                n_panels = surface.collocation.reshape(-1, 3).shape[0]
                # surfaces that don't carry this control contribute zero rows,
                # so a global deflection never touches their panels
                parts.append(gains[name].reshape(-1) if name in gains else np.zeros(n_panels))
            rows.append(np.concatenate(parts))
        self.control_gain = np.stack(rows) if names else np.zeros((0, self.collocation.shape[0]))

        # the horseshoe kernels depend on the mesh alone, so they are built with it
        self.computeInfluences()

    def plot(self, **kwargs):

        if kwargs.get('plot_control_surfaces') and 'control_colors' not in kwargs:
            # one shared name->color map, in self.control_names' order, so a
            # control keeps the same color regardless of which surface it's
            # drawn on -- without this each Surface.plot() would pick its own
            # local cycle and could give aileron and elevator the same color
            palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
            kwargs['control_colors'] = {name: palette[i % len(palette)]
                                         for i, name in enumerate(self.control_names)}

        fig, ax = plt.subplots(figsize=(8,8), subplot_kw={'projection': '3d'})
        fig.subplots_adjust(
                            left=0,
                            right=1,
                            bottom=0,
                            top=1
                            )

        ax.plot(*self.CG, marker='x', color='k', markersize=10, markeredgewidth=2, linestyle='none', label='CG')
        ax.plot(*self.neutralPoint(), marker='^', color='tab:red', markersize=9, linestyle='none', label='NP')
                            
        for surface in self.surfaces:
            fig, ax = surface.plot(fig_ax = (fig, ax), **kwargs)
            
        ax.set_box_aspect([
                            np.ptp(self.nodes[:,0]),
                            np.ptp(self.nodes[:,1]),
                            np.ptp(self.nodes[:,2])
                            ])

        # stability axes put z+ down, and matplotlib always draws z+ up, so the default
        # camera shows the aircraft belly-up. roll=180 turns the camera over instead of
        # flipping an axis: inverting z would mirror the scene and swap left/right wing,
        # which would misread any antisymmetric deflection. elev is arctan(1/sqrt(2)),
        # the isometric angle; with azim=60 the camera sits ahead, right and above.
        ax.set_proj_type('ortho')
        ax.view_init(elev=-35.264, azim=60, roll=180)
                            
        ax.legend()
        ax.set_axis_off()
        plt.show()

    def _horseshoeKernel(self, field, x_hat, zero_diagonal, eps):
        """Unit-strength horseshoe kernel, Drela eq. 6.33, evaluated at `field`"""
        a = field[:,None,:] - self.vortex_a[None,:,:]
        b = field[:,None,:] - self.vortex_b[None,:,:]
        norm_a = np.linalg.norm(a, axis = -1, keepdims = True)
        norm_b = np.linalg.norm(b, axis = -1, keepdims = True)

        bound = np.cross(a, b) * (1/norm_a + 1/norm_b) / (norm_a*norm_b + np.sum(a*b, axis = -1, keepdims = True) + eps)
        if zero_diagonal:
            bound = bound.at[np.diag_indices(bound.shape[0])].set(0.0)
        leg_a = np.cross(a, x_hat) / (norm_a * (norm_a - np.sum(a*x_hat, axis = -1, keepdims = True)) + eps)
        leg_b = np.cross(b, x_hat) / (norm_b * (norm_b - np.sum(b*x_hat, axis = -1, keepdims = True)) + eps)

        return (bound + leg_a - leg_b) / (4*np.pi)

    def computeInfluences(self, x_hat = np.array([-1.0, 0.0, 0.0]), eps = 1e-12):
        # x_hat is the wake direction, fixed to the body (-x is aft here) rather than
        # aligned with the freestream: letting the wake rotate with alpha drives the
        # eq. 6.33 denominator |a| - a.x_hat towards zero on low-sweep vertical surfaces.
        # Both kernels depend on geometry alone, so they are built once per mesh.
        self.influence_collocation = self._horseshoeKernel(self.collocation,   x_hat, False, eps)   # for the AIC
        self.influence_aero        = self._horseshoeKernel(self.aero_centers,  x_hat, True,  eps)   # for eq. 6.42

    def deflect(self, deltas=None):
        """
        Panel normals rotated about their hinge line
        """
        deltas = np.zeros(len(self.control_names)) if deltas is None else np.asarray(deltas)
        theta = self.control_gain.T @ np.deg2rad(deltas)
        n, h = self.normals, self.hinge_axis
        return n*np.cos(theta)[:,None] + np.cross(h, n)*np.sin(theta)[:,None]

    def computeAIC(self, normals):
        # normal velocity each unit-strength horseshoe induces at every collocation point
        return np.sum(self.influence_collocation * normals[:,None,:], axis = -1)

    def solveSystem(self, AIC, normals, V_bar, omega_bar=np.array([0.0, 0.0, 0.0])):
        # flow tangency: the circulations must cancel the onset velocity normal to each
        # panel, onset being the freestream plus the rotation rate at that point
        V_panel = V_bar + np.cross(omega_bar[None,:], self.collocation, axis=-1)
        b =  np.sum(V_panel * normals, axis = -1)
        return np.linalg.solve(AIC, b)

    def computeCoefficients(self, circulation, V_bar, omega_bar=np.array([0.0, 0.0, 0.0]), T_a = np.eye(3)):
        # every reference length comes from one surface, the largest by area
        reference = max(self.surfaces, key = lambda surf: surf.S)
        S_ref, chord_ref, span_ref = reference.S, reference.MAC, reference.b

        # eq. 6.42: total velocity at each bound vortex, then Kutta-Joukowski for the
        # force it carries. The panel normal plays no part here -- the force follows the
        # vortex segment, which is why a control deflection acts only through the AIC.
        induced = np.sum(self.influence_aero * circulation[None,:,None], axis=1)
        V_bar_i = induced - V_bar - np.cross(omega_bar[None,:], self.aero_centers)

        F_bar_i = (2/S_ref) * np.cross(V_bar_i, self.vortex_b-self.vortex_a) * circulation[:,None]
        M_bar_i = np.cross(self.aero_centers - self.CG, F_bar_i)   # eq. 6.49, arm from the CG
        F_bar   = np.sum(F_bar_i, axis=0)
        M_bar   = np.sum(M_bar_i, axis=0)

        # eq. 6.50: the induced drag is the force along the freestream. Reading it
        # off the body x axis instead is dominated by the tilted lift.
        CD_i       = np.dot(F_bar, V_bar)
        _, CY, CL  = T_a @ F_bar
        Cl, Cm, Cn = T_a @ (M_bar * np.array([-1/span_ref, 1/chord_ref, -1/span_ref]))

        return CD_i, CY, CL, Cl, Cm, Cn
 
    def simulate(self, V_inf, alpha, beta, omega=np.array([0.0, 0.0, 0.0]), deltas=None):
        """
        One flight condition -> (CD_i, CY, CL, Cl, Cm, Cn) in stability axes.
        alpha, beta and deltas in degrees;
        """
        # freestream direction: the aircraft flies towards +x, so the oncoming flow
        # points along -x, tilted by alpha in the x-z plane and by beta in x-y
        V_bar = np.array([
            -np.cos(np.deg2rad(alpha))*np.cos(np.deg2rad(beta)),
            -np.sin(np.deg2rad(beta)),
            -np.sin(np.deg2rad(alpha))*np.cos(np.deg2rad(beta))])

        omega_bar = omega/V_inf

        # body -> stability axes: a rotation by alpha about y, so the coefficients come
        # out along lift/side-force/drag rather than along the body axes
        T_a = np.array([
            [-np.cos(np.deg2rad(alpha)), 0.0, -np.sin(np.deg2rad(alpha))],
            [                       0.0, 1.0,                       0.0],
            [ np.sin(np.deg2rad(alpha)), 0.0, -np.cos(np.deg2rad(alpha))]
        ])

        normals     = self.deflect(deltas)

        AIC         = self.computeAIC(normals)

        circulation = self.solveSystem(AIC, normals, V_bar, omega_bar)

        return self.computeCoefficients(circulation, V_bar, omega_bar, T_a)
      
    def coefficientsAt(self, alpha, beta, deltas, V_inf=1.0, omega=np.array([0.0, 0.0, 0.0])):
        """
        The same pipeline as simulate().
        written as a pure function of the deflection so jax.jacfwd can differentiate it. 
        """
        V_bar = np.array([
            -np.cos(np.deg2rad(alpha))*np.cos(np.deg2rad(beta)),
            -np.sin(np.deg2rad(beta)),
            -np.sin(np.deg2rad(alpha))*np.cos(np.deg2rad(beta))])

        omega_bar = omega/V_inf

        # body -> stability axes: a rotation by alpha about y, so the coefficients come
        # out along lift/side-force/drag rather than along the body axes
        T_a = np.array([
            [-np.cos(np.deg2rad(alpha)), 0.0, -np.sin(np.deg2rad(alpha))],
            [                       0.0, 1.0,                       0.0],
            [ np.sin(np.deg2rad(alpha)), 0.0, -np.cos(np.deg2rad(alpha))]
        ])

        normals     = self.deflect(deltas)

        AIC         = self.computeAIC(normals)

        circulation = self.solveSystem(AIC, normals, V_bar, omega_bar)

        return self.computeCoefficients(circulation, V_bar, omega_bar, T_a)

    def stabilityDerivatives(self, alpha, beta, deltas, V_inf=1.0, omega=np.array([0.0, 0.0, 0.0])):
            """d(CD_i,CY,CL,Cl,Cm,Cn)/d(alpha)"""
            f = lambda alpha: np.array(self.coefficientsAt(alpha, beta, deltas, V_inf, omega))
            return jax.jacfwd(f)(alpha)
    
    def controlDerivatives(self, alpha, beta, deltas, V_inf=1.0, omega=np.array([0.0, 0.0, 0.0])):
        """d(CD_i,CY,CL,Cl,Cm,Cn)/d(delta)"""
        f = lambda deltas: np.array(self.coefficientsAt(alpha, beta, deltas, V_inf, omega))
        return jax.jacfwd(f)(deltas)

    def neutralPoint(self, alpha=0.0, beta=0.0, deltas=None):
        """Whole-aircraft aerodynamic center: the CG_x where dCm/dalpha = 0.

        Cm is linear in CG_x (the moment arm aero_centers - CG enters the pitching
        moment through a fixed total force that doesn't itself depend on CG), so one
        evaluation of stabilityDerivatives at the current CG gives it exactly --
        x_np = CG_x + MAC * (dCm/dalpha)/(dCL/dalpha), no sweep over CG needed.
        Confirmed against a CG sweep to within float32 noise (~1e-7).
        """
        reference = max(self.surfaces, key = lambda surf: surf.S)
        d = self.stabilityDerivatives(alpha, beta, deltas)
        x_np = self.CG[0] + reference.MAC * d[4] / d[2]
        return np.array([x_np, self.CG[1], self.CG[2]])
    
    
if __name__ == "__main__":
    
    wing = Surface(span= [0.0, 1.0, 3.5], 
                   chord = [1.75, 1.0, 0.5], 
                   sweep = [10.0, 10.0],
                   dihedral = [1.0,2.0], 
                   position = [0.0, 0.0, 0.0],
                   discretization = [(6, 11), (6, 31)],
                   symmetry = True, 
                   control_surfaces = [dict(name='aileron', 
                                            hinge=0.75, 
                                            span=(0.55, 0.95), 
                                            mode='antisymmetric'),
                                        dict(name='flap',
                                            hinge=0.70, 
                                            span=(0.05, 0.50), 
                                            mode='symmetric')])
    
    winglet_right = Surface(span=[0.0, 0.5],
                            chord=[0.5, 0.2],
                            sweep=[10.0],
                            dihedral=[90.0],
                            position=[-3.5*np.sin(np.deg2rad(10)),  3.4983, -0.1047],
                            discretization=[(5, 10)],
                            symmetry=False)

    winglet_left = Surface(span=[0.0, 0.5],
                           chord=[0.5, 0.2],
                           sweep=[10.0],
                           dihedral=[90.0],
                           position=[-3.5*np.sin(np.deg2rad(10)), -3.4983, -0.1047],
                           discretization=[(5, 10)],
                           symmetry=False)

    hTail = Surface(span= [0.0, 1.5], 
                    chord = [0.75, 0.4], 
                    sweep = [25.0],
                    dihedral = [0.0], 
                    position = [-3.0, 0.0, -1.0],
                    discretization = [(10, 25)],
                    symmetry = True,
                    control_surfaces = [dict(name='elevator',
                                            hinge=0.75, 
                                            span=(0.15, 0.95), 
                                            mode='symmetric')])

    vTail = Surface(span= [0.0, 1.25], 
                    chord = [0.75, 0.4], 
                    sweep = [25.0],
                    dihedral = [90.0], 
                    position = [-3.0, 0.0, -1.0],
                    discretization = [(10, 25)],
                    symmetry = False,
                    control_surfaces = [dict(name='rudder',
                                            hinge=0.50, 
                                            span=(0.05, 0.95), 
                                            mode='symmetric')])
                                      
    airplane = Aircraft(surfaces = [wing, winglet_right, winglet_left, hTail, vTail], CG = np.array([-0.0, 0.0, 0.0]))
    
    width, labels = 9, ('CD_i', 'CY', 'CL', 'Cl', 'Cm', 'Cn')
    row_label_width = 4 + max(len(n) for n in airplane.control_names)   # fits "d/d<name>"
    header = f"\n{'':>{row_label_width}} " + " ".join(f"{name:>{width}}" for name in labels)

    deltas = np.zeros(len(airplane.control_names))
    for a in range(-10, 11, 1):
        print(header)
        # deltas = deltas.at[i].set(float(a))

        c  = airplane.coefficientsAt(alpha=float(a), beta=0.0, deltas=deltas)
        da = airplane.stabilityDerivatives(alpha=float(a), beta=0.0, deltas=deltas)
        dd = airplane.controlDerivatives(alpha=float(a), beta=0.0, deltas=deltas)

        print(f"{f'AoA:{a:.1f}':>{row_label_width}} " + " ".join(f"{v:{width}.4f}" for v in c))
        print(f"{'d/dalpha':>{row_label_width}} " + " ".join(f"{v:{width}.4f}" for v in da))
        for j, name in enumerate(airplane.control_names):
            print(f"{'d/d'+name:>{row_label_width}} " + " ".join(f"{v:{width}.4f}" for v in dd[:, j]))


    airplane.plot(plot_control_surfaces = True)
    
    