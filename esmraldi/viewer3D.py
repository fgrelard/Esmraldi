"""
3D viewer for MS images
Based on vedo (inherits Plotter)
"""
from __future__ import division, print_function
import vtk
from vedo.addons import addScalarBar
from vedo.plotter import Plotter
from vedo.pyplot import cornerHistogram
from vedo.utils import mag, precision, linInterpolate, isSequence
from vedo.colors import printc, colorMap, getColor
from vedo.shapes import Text2D
from vedo.mesh import Mesh
from vedo import settings
import numpy as np
import matplotlib.pyplot as plt



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
        self.volume_copy = volume.clone()
        self.cmap_slicer = cmaps[0]
        self.alpha = alpha
        self.alphas = [1 for i in range(100)]
        self.showing_mesh = False
        self.map2cells = map2cells

        dims = volume.dimensions()
        self.rmin, self.rmax = volume.imagedata().GetScalarRange()

        self.pos_slider = [dims[0], dims[1], int(volume.dimensions()[2]/2)]


        self.volume.mode(1).color(self.cmap_slicer).jittering(True)
        self.setOTF()

        self.box = volume.box().wireframe().alpha(0)
        self.add(self.box, render=False)

        # inits
        la, ld = 0.7, 0.3 #ambient, diffuse
        data = volume.getPointArray()

        if clamp:
            hdata, edg = np.histogram(data, bins=50)
            logdata = np.log(hdata+1)
            # mean  of the logscale plot
            meanlog = np.sum(np.multiply(edg[:-1], logdata))/np.sum(logdata)
            self.rmax = min(self.rmax, meanlog+(meanlog-self.rmin)*0.9)
            self.rmin = max(self.rmin, meanlog-(self.rmax-meanlog)*0.9)
            if verbose:
                printc('scalar range clamped to: (' +
                       precision(self.rmin, 3) +', '+  precision(self.rmax, 3)+')', c='m', bold=0)

        self.visibles = [None, None, None]
        self.all_slices = [[], [], []]
        self.scalar_msh = volume.zSlice(self.pos_slider[2])
        self.scalar_msh.alpha(self.alpha).lighting('', la, ld, 0)
        self.scalar_msh.pointColors(cmap=self.cmap_slicer, vmin=self.rmin, vmax=self.rmax, alpha=self.alphas)
        self.scalar_msh.SetVisibility(False)
        self.scalar_msh.scalarbar = addScalarBar(self.scalar_msh, pos=(0.04,0.0), horizontal=True, titleFontSize=0)

        self.add(self.scalar_msh)

        self.msh = self.scalar_msh.clone()
        if map2cells: self.msh.mapPointsToCells()
        self.renderer.AddActor(self.msh)
        self.visibles[2] = self.msh

        def sliderfunc_x(widget, event):
            """
            Event on change slider x
            """
            i = int(widget.GetRepresentation().GetValue())
            self.pos_slider[0] = i
            self.msh = self.volume.xSlice(i).alpha(self.alpha).lighting('', la, ld, 0)
            self.msh.pointColors(cmap=self.cmap_slicer, vmin=self.rmin, vmax=self.rmax, alpha=self.alphas)
            if map2cells: self.msh.mapPointsToCells()
            self.renderer.RemoveActor(self.visibles[0])
            if i<dims[0]: self.renderer.AddActor(self.msh)
            self.visibles[0] = self.msh

        def sliderfunc_y(widget, event):
            """
            Event on change slider y
            """
            i = int(widget.GetRepresentation().GetValue())
            self.pos_slider[1] = i
            self.msh = self.volume.ySlice(i).alpha(self.alpha).lighting('', la, ld, 0)
            self.msh.pointColors(cmap=self.cmap_slicer, vmin=self.rmin, vmax=self.rmax, alpha=self.alphas)
            if map2cells: self.msh.mapPointsToCells()
            self.renderer.RemoveActor(self.visibles[1])
            if i<dims[1]: self.renderer.AddActor(self.msh)
            self.visibles[1] = self.msh

        def sliderfunc_z(widget, event):
            """
            Event on change slider z
            """
            i = int(widget.GetRepresentation().GetValue())
            self.pos_slider[2] = i
            self.comment.SetText(4, "z="+str(self.pos_slider[2]))
            self.msh = self.volume.zSlice(i).alpha(self.alpha).lighting('', la, ld, 0)
            self.msh.pointColors(cmap=self.cmap_slicer, vmin=self.rmin, vmax=self.rmax, alpha=self.alphas)
            if map2cells: self.msh.mapPointsToCells()
            self.renderer.RemoveActor(self.visibles[2])
            if i<dims[2]: self.renderer.AddActor(self.msh)
            self.visibles[2] = self.msh

        cx, cy, cz, ch = 'dr', 'dg', 'db', (0.3,0.3,0.3)
        if np.sum(self.renderer.GetBackground()) < 1.5:
            cx, cy, cz = 'lr', 'lg', 'lb'
            ch = (0.8,0.8,0.8)

        self.addSlider2D(sliderfunc_x, 0, dims[0]+1, title='X', titleSize=0.5,
                         value=self.pos_slider[0],

                         pos=[(0.8,0.12), (0.95,0.12)], showValue=False, c=cx)
        self.addSlider2D(sliderfunc_y, 0, dims[1]+1, title='Y', titleSize=0.5,
                         value=self.pos_slider[1],
                         pos=[(0.8,0.08), (0.95,0.08)], showValue=False, c=cy)
        self.addSlider2D(sliderfunc_z, 0, dims[2]+1, title='Z', titleSize=0.6,
                         value=self.pos_slider[2],
                         pos=[(0.8,0.04), (0.95,0.04)], showValue=False, c=cz)
        self.scalar_thresh = self.addSlider2D(self.sliderThreshold, 0, 100, value=100, pos=[(0.04, 0.1), (0.2, 0.1)], showValue=True, title="Scalar")

        self.opacity_thresh = self.addSlider2D(self.sliderOpacityThreshold, 0, 100, value=0, pos=[(0.04, 0.2), (0.2, 0.2)], showValue=True, title="Opacity")



        #################
        def keyfunc(iren, event):
            """
            Keyboard events on the viewer
            s : save image
            x : display all slice on x axis
            y : display all slice on y axis
            z : display all slice on z axis
            v : display volume rendering
            """
            if not iren.GetKeyCode():
                return
            key = iren.GetKeySym()
            if key=='s':
                array = self.volume.getDataArray()
                index = self.pos_slider[2]
                if index < array.shape[2]:
                    image = self.volume.getDataArray()[..., index].T
                    fig, ax = plt.subplots(1,1)
                    ax.axis("off")
                    ax.imshow(image, cmap="gray")
                    plt.savefig("saved_fig.png", bbox_inches="tight", pad_inches=0)
            if key=='x':
                self.display_all_slices(0)
            if key=='y':
                self.display_all_slices(1)
            if key=='z':
                self.display_all_slices(2)
            if key=='v':
                self.showing_mesh = not self.showing_mesh
                if self.volume.GetBounds()[-1] == 0:
                    return
                if self.showing_mesh:
                    self.add(self.volume)
                    self.interactive = True
                else:
                    self.remove(self.volume)


        def buttonfunc(iren, event):
            """
            Mouse event
            Allows to change color map
            """
            clickPos = iren.GetEventPosition()

            picker = vtk.vtkPropPicker()
            picker.Pick(clickPos[0], clickPos[1], 0, self.renderer)
            if not picker.GetActor2D():
                return
            bu.switch()
            self.cmap_slicer = bu.status()
            self.refresh()
            # self.add(self.scalar_msh)


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

        self.comment = Text2D("z="+str(self.pos_slider[2]),pos=5,
                              font='Montserrat', s=0.6)
        self.add([self.msh, self.comment])
        if verbose:
            printc("Press button to cycle through color maps,", c="m")
            printc("Use sliders to select the slicing planes.", c="m")

    def display_all_slices(self, axis):
        """
        Display all 2D slices in the 3D volume
        """
        la, ld = 0.7, 0.3 #ambient, diffuse
        if len(self.all_slices[axis]) > 0:
            for elem in self.all_slices[axis]:
                self.remove(elem)
            self.all_slices[axis].clear()
        else:
            dim = self.volume.dimensions()
            for i in range(dim[axis]):
                size = dim[axis]
                n = size / 10
                if axis == 0:
                    if i % n == 0:
                        msh = self.volume.xSlice(i).alpha(self.alpha).lighting('', la, ld, 0)
                elif axis == 1:
                    if i % n == 0:
                        msh = self.volume.ySlice(i).alpha(self.alpha).lighting('', la, ld, 0)
                elif axis == 2:
                    msh = self.volume.zSlice(i).lighting('', la, ld, 0)
                msh.alpha(self.alpha).lighting('', la, ld, 0).reverse(False,True)
                msh.pointColors(cmap=self.cmap_slicer, vmin=self.rmin, vmax=self.rmax, alpha=self.alphas)
                self.all_slices[axis].append(msh)
                self.add(msh)

    def setOTF(self):
        """
        Function to initialize transfer function.
        Useful for volume rendering
        """
        opacity_function = self.volume.GetProperty().GetScalarOpacity()
        opacity_function.RemoveAllPoints()
        opacity_function.AddPoint(self.rmin, 0.0)
        opacity_function.AddPoint(self.rmin + (self.rmax - self.rmin) * 0.1, 0.1)
        opacity_function.AddPoint(self.rmin + (self.rmax - self.rmin) * 0.25, 0.7)
        opacity_function.AddPoint(self.rmin + (self.rmax - self.rmin) * 0.5, 0.7)
        opacity_function.AddPoint(self.rmin + (self.rmax - self.rmin) * 1.0, 1.0)


    def sliderThreshold(self, widget, event):
        """
        Action on intensity changes
        """
        value = widget.GetRepresentation().GetValue()
        pmin, pmax = self.volume.GetProperty().GetScalarOpacity().GetRange()
        self.rmax = self.rmin + pmax*value*0.01
        self.refresh()

    def sliderOpacityThreshold(self, widget, event):
        """
        Action on opacity changes
        """
        value = int(widget.GetRepresentation().GetValue())
        self.alphas = [0 for i in range(value)] + [1 for i in range(100-value)]
        self.refresh()

    def refresh(self):
        """
        Refresh view on internal events

        """
        for mesh in self.visibles:
            if mesh:
                mesh.pointColors(cmap=self.cmap_slicer, vmin=self.rmin, vmax=self.rmax, alpha=self.alphas)
                if self.map2cells:
                    mesh.mapPointsToCells()

        self.volume.mode(1).color(self.cmap_slicer).jittering(True)
        img = self.volume_copy.imagedata()
        self.volume._update(img)
        self.volume.threshold(above=self.rmax, replaceWith=self.rmax)

        self.scalar_msh.pointColors(cmap=self.cmap_slicer, vmin=self.rmin, vmax=self.rmax)
        self.renderer.RemoveActor(self.scalar_msh.scalarbar)


        self.scalar_msh.scalarbar = addScalarBar(self.scalar_msh,
                                      pos=(0.04,0.0),
                                      horizontal=True,
                                      titleFontSize=0)

        self.renderer.AddActor(self.scalar_msh.scalarbar)


    def update(self, vol):
        """
        Update the volume on external events
        (key or mouse events), which happen on
        other elements in the Application (e.g. matplotlib plot).

        Example: selection of a new m/z image on the
        spectrum

        Parameters
        ----------
        self: type
            description
        vol: Volume
            Volume
        """
        la, ld = 0.7, 0.3 #ambient, diffuse
        dims = vol.dimensions()
        self.rmin, self.rmax = vol.imagedata().GetScalarRange()
        self.remove([self.volume, self.box, self.msh])

        self.volume = vol
        self.volume_copy = vol.clone()
        self.volume.mode(1).color(self.cmap_slicer).jittering(False)

        self.scalar_thresh.GetSliderRepresentation().SetValue(100)

        self.setOTF()
        self.box = vol.box().wireframe().alpha(0)
        self.add(self.box, render=False)

        for m in self.visibles:
            self.renderer.RemoveActor(m)

        previous_visibles = self.visibles
        self.visibles = [None, None, None]
        for i in range(len(previous_visibles)):
            elem = previous_visibles[i]
            index = self.pos_slider[i]
            if elem and index < dims[i]:
                previous_visibles[i] = elem
            else:
                previous_visibles[i] = None

        for i in range(len(previous_visibles)):
            if i == 0:
                self.msh = self.volume.xSlice(self.pos_slider[0]).alpha(self.alpha).lighting('', la, ld, 0)
            elif i == 1:
                self.msh = self.volume.ySlice(self.pos_slider[1]).alpha(self.alpha).lighting('', la, ld, 0)
            elif i == 2:
                self.msh = self.volume.zSlice(self.pos_slider[2]).alpha(self.alpha).lighting('', la, ld, 0)
            self.msh.alpha(self.alpha).lighting('', la, ld, 0)
            self.msh.pointColors(cmap=self.cmap_slicer, vmin=self.rmin, vmax=self.rmax, alpha=self.alphas)
            if previous_visibles[i]:
                self.visibles[i] = self.msh
                self.add(self.msh)
            else:
                self.visibles[i] = None

        indices = np.argwhere(self.all_slices).flatten()
        for i in indices:
            self.display_all_slices(i)
            self.display_all_slices(i)

        if self.showing_mesh:
            self.add(self.volume)
            self.interactive=True

