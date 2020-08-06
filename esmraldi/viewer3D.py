from __future__ import division, print_function
import vtk
from vedo.addons import addScalarBar
from vedo.plotter import Plotter
from vedo.pyplot import cornerHistogram,cornerPlot
from vedo.utils import mag, precision, linInterpolate, isSequence
from vedo.colors import printc, colorMap, getColor
from vedo.shapes import Text2D
from vedo import settings
import numpy as np
import vedo
import math



def spectrumPlot(points, c):
    """
    Return a ``vtkChartXY`` that is a plot of `x` versus `y`,
    where `points` is a list of `(x,y)` points.

    :param int pos: assign position:

        - 1, topleft,
        - 2, topright,
        - 3, bottomleft,
        - 4, bottomright.
    """
    # if len(points) == 2:  # passing [allx, ally]
    #     # points = list(zip(points[0], points[1]))
    #     points = np.stack((points[0], points[1]), axis=1)


    # c = vedo.colors.getColor(c)  # allow different codings
    chart = vtk.vtkChartXY()
    array_x = vtk.vtkFloatArray()
    array_x.SetName("m/z")

    array_y = vtk.vtkFloatArray()
    array_y.SetName("Intensities")


    table = vtk.vtkTable()
    table.AddColumn(array_x)
    table.AddColumn(array_y)

    table.SetNumberOfRows(len(points[0]))
    for i in range(len(points[0])):
        table.SetValue(i, 0, points[0][i])
        table.SetValue(i, 1, points[1][i])

    chart.GetAxis(0).SetTitle("Intensities")
    chart.GetAxis(1).SetTitle("m/z")


    points = chart.AddPlot(vtk.vtkChart.LINE)
    points.SetInputData(table, 0, 1)
    points.SetColor(0, 50, 125, 255)
    points.SetWidth(2.0)

    return chart

# globals
_cmap_slicer='gist_ncar_r'
_alphaslider0, _alphaslider1, _alphaslider2 = 0.33, 0.66, 1  # defaults
_kact=0
_showing_mesh = False

##########################################################################
def Slicer(volume,
           mz,
           mean_spectrum,
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
           showHisto=True,
           showIcon=True,
           draggable=False,
           verbose=True,
           ):
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
    :param bool showHisto: show histogram on bottom left
    :param bool showIcon: show a small 3D rendering icon of the volume
    :param bool draggable: make the icon draggable
    """
    global _cmap_slicer

    if verbose: printc("Slicer tool", invert=1, c="m")

    custom_shape = [
        dict(bottomleft=(0.0,0.0), topright=(1.00,1.00), bg=bg, bg2=bg2),
        dict(bottomleft=(0.01,0.01), topright=(0.95,0.20), bg=bg,  bg2=bg2)
]
    ################################
    vp = Plotter(shape=custom_shape, bg=bg, bg2=bg2,
                 size=size,
                 screensize=screensize,
                 title=title,
                 interactive=False,
                 verbose=verbose)

    ################################
    box = volume.box().wireframe().alpha(0)

    if showIcon:
        vp.showInset(volume, pos=(.85,.85), size=0.15, c='w', draggable=draggable)

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
    msh = volume.zSlice(int(dims[2]/2))
    msh.alpha(alpha).lighting('', la, ld, 0)
    msh.pointColors(cmap=_cmap_slicer, vmin=rmin, vmax=rmax)
    if map2cells: msh.mapPointsToCells()
    vp.renderers[0].AddActor(msh)
    visibles[2] = msh
    addScalarBar(msh, pos=(0.04,0.0), horizontal=True, titleFontSize=0)

    def sliderfunc_x(widget, event):
        i = int(widget.GetRepresentation().GetValue())
        msh = volume.xSlice(i).alpha(alpha).lighting('', la, ld, 0)
        msh.pointColors(cmap=_cmap_slicer, vmin=rmin, vmax=rmax)
        if map2cells: msh.mapPointsToCells()
        vp.renderers[0].RemoveActor(visibles[0])
        if i and i<dims[0]: vp.renderers[0].AddActor(msh)
        visibles[0] = msh

    def sliderfunc_y(widget, event):
        i = int(widget.GetRepresentation().GetValue())
        msh = volume.ySlice(i).alpha(alpha).lighting('', la, ld, 0)
        msh.pointColors(cmap=_cmap_slicer, vmin=rmin, vmax=rmax)
        if map2cells: msh.mapPointsToCells()
        vp.renderers[0].RemoveActor(visibles[1])
        if i and i<dims[1]: vp.renderers[0].AddActor(msh)
        visibles[1] = msh

    def sliderfunc_z(widget, event):
        i = int(widget.GetRepresentation().GetValue())
        msh = volume.zSlice(i).alpha(alpha).lighting('', la, ld, 0)
        msh.pointColors(cmap=_cmap_slicer, vmin=rmin, vmax=rmax)
        if map2cells: msh.mapPointsToCells()
        vp.renderers[0].RemoveActor(visibles[2])
        if i and i<dims[2]: vp.renderers[0].AddActor(msh)
        visibles[2] = msh

    cx, cy, cz, ch = 'dr', 'dg', 'db', (0.3,0.3,0.3)
    if np.sum(vp.renderers[0].GetBackground()) < 1.5:
        cx, cy, cz = 'lr', 'lg', 'lb'
        ch = (0.8,0.8,0.8)

    if not useSlider3D:
        vp.addSlider2D(sliderfunc_x, 0, dims[0], title='X', titleSize=0.5,
                       pos=[(0.8,0.12), (0.95,0.12)], showValue=False, c=cx)
        vp.addSlider2D(sliderfunc_y, 0, dims[1], title='Y', titleSize=0.5,
                       pos=[(0.8,0.08), (0.95,0.08)], showValue=False, c=cy)
        vp.addSlider2D(sliderfunc_z, 0, dims[2], title='Z', titleSize=0.6,
                       value=int(dims[2]/2),
                       pos=[(0.8,0.04), (0.95,0.04)], showValue=False, c=cz)
    else: # 3d sliders attached to the axes bounds
        bs = box.bounds()
        vp.addSlider3D(sliderfunc_x,
            pos1=(bs[0], bs[2], bs[4]),
            pos2=(bs[1], bs[2], bs[4]),
            xmin=0, xmax=dims[0],
            t=box.diagonalSize()/mag(box.xbounds())*0.6,
            c=cx,
            showValue=False,
        )
        vp.addSlider3D(sliderfunc_y,
            pos1=(bs[1], bs[2], bs[4]),
            pos2=(bs[1], bs[3], bs[4]),
            xmin=0, xmax=dims[1],
            t=box.diagonalSize()/mag(box.ybounds())*0.6,
            c=cy,
            showValue=False,
        )
        vp.addSlider3D(sliderfunc_z,
            pos1=(bs[0], bs[2], bs[4]),
            pos2=(bs[0], bs[2], bs[5]),
            xmin=0, xmax=dims[2],
            value=int(dims[2]/2),
            t=box.diagonalSize()/mag(box.zbounds())*0.6,
            c=cz,
            showValue=False,
        )





    #################
    def buttonfunc():
        global _cmap_slicer
        bu.switch()
        _cmap_slicer = bu.status()
        for mesh in visibles:
            if mesh:
                mesh.pointColors(cmap=_cmap_slicer, vmin=rmin, vmax=rmax)
                if map2cells:
                    mesh.mapPointsToCells()
        vp.renderers[0].RemoveActor(mesh.scalarbar)
        mesh.scalarbar = addScalarBar(mesh,
                                      pos=(0.04,0.0),
                                      horizontal=True,
                                      titleFontSize=0)
        vp.renderers[0].AddActor(mesh.scalarbar)

    def keyfunc(key, vol, vp):
        global _showing_mesh
        if key=='x':
            _showing_mesh = not _showing_mesh
            if _showing_mesh:
                vp.add(vol)
            else:
                vp.remove(vol)


    def clickfunc(iren, event):

        x, y = iren.GetEventPosition()
        renderer = iren.FindPokedRenderer(x, y)
        if renderer == vp.renderers[1]:
            picker = iren.GetPicker()
            picker.PickProp(x, y, renderer)
            clickedActor = picker.GetActor2D()
            print(clickedActor, picker.GetActor())
            vp.clickedActor = clickedActor
            if hasattr(clickedActor, 'picked3d'):
                clickedActor.picked3d = picker.GetPickPosition()

            if vp.mouseLeftClickFunction:
                vp.mouseLeftClickFunction(clickedActor)



    def mousemovefunc(interactor, event):
        lastPos = interactor.GetLastEventPosition()
        currPos = interactor.GetEventPosition()
        renderer = interactor.FindPokedRenderer(currPos[0], currPos[1])
        if renderer == vp.renderers[1]:
            contextInteractorStyle.SetScene(scene)
            interactor.SetInteractorStyle(contextInteractorStyle)

        else:
            contextInteractorStyle.SetScene(None)
            interactor.SetInteractorStyle(trackInteractorStyle)


    bu = vp.addButton(buttonfunc,
        pos=(0.27, 0.005),
        states=cmaps,
        c=["db"]*len(cmaps),
        bc=["lb"]*len(cmaps),  # colors of states
        size=14,
        bold=True,
    )

    #################

    mean_spectrum_plot = cornerPlot([mz,mean_spectrum],
                                    s=0.9,
                                    c=(0.0,0.1,0.3),
                                    bg=(0.3,0.3,0.3),
                                    pos=(0,0)
    )
    chart = spectrumPlot([mz, mean_spectrum], c=(0.0,0.1,0.3))

    mean_spectrum_plot = vtk.vtkContextActor()
    scene = vtk.vtkContextScene()
    scene.AddItem(chart)
    mean_spectrum_plot.SetScene(scene)

    vp.renderers[1].AddActor(mean_spectrum_plot)
    scene.SetRenderer(vp.renderers[1])


    hist = None
    if showHisto:
        hist = cornerHistogram(data, s=0.2,
                               bins=25, logscale=1, pos=(0.02, 0.02),
                               c=ch, bg=ch, alpha=0.7)

    comment = None
    if verbose:
        comment = Text2D("Use sliders to slice volume\nClick button to change colormap",
                         font='Montserrat', s=0.8)

    # vp.keyPressFunction = lambda key, vol=volume, vp=vp: keyfunc(key, vol, vp)

    contextInteractorStyle = vtk.vtkContextInteractorStyle()
    contextInteractorStyle.SetScene(None)

    trackInteractorStyle = vtk.vtkInteractorStyleTrackballCamera()

    vp.interactor.RemoveObservers("TimerEvent")

    vp.interactor.AddObserver("MouseMoveEvent", mousemovefunc)

    # vp.interactor.AddObserver("LeftButtonPressEvent", clickfunc)
    vp.show([msh, box], at=0, viewup="z", interactive=False)
    # vp.show(box, at=0, viewup="z", interactive=False)

    # vp.show(mean_spectrum_plot, at=1, sharecam=False, viewup="z", interactive=False)



    if verbose:
        printc("Press button to cycle through color maps,", c="m")
        printc("Use sliders to select the slicing planes.", c="m")
    return vp
