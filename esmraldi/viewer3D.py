from __future__ import division, print_function
import vtk
from vedo.addons import addScalarBar
from vedo.plotter import Plotter
from vedo.pyplot import cornerHistogram
from vedo.utils import mag, precision, linInterpolate, isSequence
from vedo.colors import printc, colorMap, getColor
from vedo.shapes import Text2D
from vedo import settings
import numpy as np

_cmap_slicer='gist_ncar_r'
_alphaslider0, _alphaslider1, _alphaslider2 = 0.33, 0.66, 1  # defaults
_kact=0
_showing_mesh = False


class Slicer(Plotter):
    """
    Generate a ``Plotter`` window with slicing planes for the input Volume.
    Returns the ``Plotter`` object.

    :param float alpha: transparency of the slicing planes
    :param list cmaps: list of color maps names to cycle when clicking button
    :param bool map2cells: scalars are mapped to cells, not intepolated.
    :param bool clamp: clamp scalar to reduce the effect of tails in color mapping
    :param bool useSlider3D: show sliders attached along the axes
    :param list size: rendering window size in pixels
    :param list screensize: size of the screen can be specified
    :param str title: window title
    :param bg: background color
    :param bg2: background gradient color
    :param int axes: axis type number
    :param bool draggable: make the icon draggable
    """

    def __init__(self, volume,
                 qtWidget,
                 alpha=1,
                 cmaps=('gist_ncar_r', "hot_r", "bone_r", "jet", "Spectral_r"),
                 map2cells=False,  # buggy
                 clamp=True,
                 useSlider3D=False,
                 size=(850,700),
                 screensize="auto",
                 title="",
                 bg="white",
                 bg2="lightblue",
                 axes=1,
                 draggable=False,
                 verbose=True):

        super().__init__(qtWidget=qtWidget, bg=bg, bg2=bg2,
                         size=size,
                         screensize=screensize,
                         title=title,
                         interactive=False,
                         offscreen=True,
                         verbose=verbose)
        ################################
        self.volume = volume
        self.box = volume.box().wireframe().alpha(0)
        self.alpha = alpha
        self.add(self.box, render=False)

        # inits
        la, ld = 0.7, 0.3 #ambient, diffuse
        dims = volume.dimensions()
        data = volume.getPointArray()
        rmin, rmax = volume.imagedata().GetScalarRange()
        if clamp:
            hdata, edg = np.histogram(data, bins=50)
            logdata = np.log(hdata+1)
            # mean  of the logscale plot
            meanlog = np.sum(np.multiply(edg[:-1], logdata))/np.sum(logdata)
            rmax = min(rmax, meanlog+(meanlog-rmin)*0.9)
            rmin = max(rmin, meanlog-(rmax-meanlog)*0.9)
            if verbose:
                printc('scalar range clamped to: (' +
                       precision(rmin, 3) +', '+  precision(rmax, 3)+')', c='m', bold=0)
        _cmap_slicer = cmaps[0]
        visibles = [None, None, None]
        self.msh = volume.zSlice(int(dims[2]/2))
        self.msh.alpha(alpha).lighting('', la, ld, 0)
        self.msh.pointColors(cmap=_cmap_slicer, vmin=rmin, vmax=rmax)
        if map2cells: self.msh.mapPointsToCells()
        self.renderer.AddActor(self.msh)
        visibles[2] = self.msh
        addScalarBar(self.msh, pos=(0.04,0.0), horizontal=True, titleFontSize=0)

        def sliderfunc_x(widget, event):
            i = int(widget.GetRepresentation().GetValue())
            self.msh = volume.xSlice(i).alpha(alpha).lighting('', la, ld, 0)
            self.msh.pointColors(cmap=_cmap_slicer, vmin=rmin, vmax=rmax)
            if map2cells: self.msh.mapPointsToCells()
            self.renderer.RemoveActor(visibles[0])
            if i and i<dims[0]: self.renderer.AddActor(self.msh)
            visibles[0] = self.msh

        def sliderfunc_y(widget, event):
            i = int(widget.GetRepresentation().GetValue())
            self.msh = volume.ySlice(i).alpha(alpha).lighting('', la, ld, 0)
            self.msh.pointColors(cmap=_cmap_slicer, vmin=rmin, vmax=rmax)
            if map2cells: self.msh.mapPointsToCells()
            self.renderer.RemoveActor(visibles[1])
            if i and i<dims[1]: self.renderer.AddActor(self.msh)
            visibles[1] = self.msh

        def sliderfunc_z(widget, event):
            i = int(widget.GetRepresentation().GetValue())
            self.msh = volume.zSlice(i).alpha(alpha).lighting('', la, ld, 0)
            self.msh.pointColors(cmap=_cmap_slicer, vmin=rmin, vmax=rmax)
            if map2cells: self.msh.mapPointsToCells()
            self.renderer.RemoveActor(visibles[2])
            if i and i<dims[2]: self.renderer.AddActor(self.msh)
            visibles[2] = self.msh

        cx, cy, cz, ch = 'dr', 'dg', 'db', (0.3,0.3,0.3)
        if np.sum(self.renderer.GetBackground()) < 1.5:
            cx, cy, cz = 'lr', 'lg', 'lb'
            ch = (0.8,0.8,0.8)

        self.addSlider2D(sliderfunc_x, 0, dims[0], title='X', titleSize=0.5,
                         pos=[(0.8,0.12), (0.95,0.12)], showValue=False, c=cx)
        self.addSlider2D(sliderfunc_y, 0, dims[1], title='Y', titleSize=0.5,
                         pos=[(0.8,0.08), (0.95,0.08)], showValue=False, c=cy)
        self.addSlider2D(sliderfunc_z, 0, dims[2], title='Z', titleSize=0.6,
                         value=int(dims[2]/2),
                         pos=[(0.8,0.04), (0.95,0.04)], showValue=False, c=cz)


        #################
        def keyfunc(iren, event):
            print("keyfunc")
            global _showing_mesh
            key = iren.GetKeySym()
            if key=='x':
                _showing_mesh = not _showing_mesh
                if _showing_mesh:
                    self.add(self.volume)
                    self.interactive = True
                else:
                    self.remove(self.volume)

        def buttonfunc(iren, event):
            global _cmap_slicer
            clickPos = iren.GetEventPosition()

            picker = vtk.vtkPropPicker()
            picker.Pick(clickPos[0], clickPos[1], 0, self.renderer)
            if not picker.GetActor2D():
                return
            bu.switch()
            _cmap_slicer = bu.status()
            for mesh in visibles:
                if mesh:
                    mesh.pointColors(cmap=_cmap_slicer, vmin=rmin, vmax=rmax)
                    if map2cells:
                        mesh.mapPointsToCells()
            self.renderer.RemoveActor(mesh.scalarbar)
            volume.mode(0).color(_cmap_slicer).jittering(True)

            mesh.scalarbar = addScalarBar(mesh,
                                          pos=(0.04,0.0),
                                          horizontal=True,
                                          titleFontSize=0)
            self.renderer.AddActor(mesh.scalarbar)

        bu = self.addButton(buttonfunc,
            pos=(0.27, 0.005),
            states=cmaps,
            c=["db"]*len(cmaps),
            bc=["lb"]*len(cmaps),  # colors of states
            size=14,
            bold=True,
        )
        self.interactor.AddObserver("KeyPressEvent", keyfunc)

        self.interactor.AddObserver("LeftButtonPressEvent", buttonfunc)

        #################

        comment = None
        if verbose:
            comment = Text2D("Use sliders to slice volume\nClick button to change colormap",
                             font='Montserrat', s=0.8)

        self.add([self.msh, comment])
        if verbose:
            printc("Press button to cycle through color maps,", c="m")
            printc("Use sliders to select the slicing planes.", c="m")

    def update(self, vol):
        la, ld = 0.7, 0.3 #ambient, diffuse
        dims = vol.dimensions()
        rmin, rmax = vol.imagedata().GetScalarRange()
        self.remove([self.volume, self.box, self.msh])
        self.volume = vol
        self.box = vol.box().wireframe().alpha(0)
        self.msh = vol.zSlice(int(dims[2]/2))
        self.msh.alpha(self.alpha).lighting('', la, ld, 0)
        self.msh.pointColors(cmap=_cmap_slicer, vmin=rmin, vmax=rmax)
        self.add([self.volume, self.box, self.msh])
