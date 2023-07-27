=======
Modules
=======


General organization
====================

The core of the Esmraldi library is contained in the ``esmraldi`` directory.

The ``examples`` directory contains various scripts to process MSI data.

The ``gui`` directory is everything GUI related (views and controllers).

General script usage
====================

Several examples are available for each step of the fusion workflow.
They are located in the ``examples`` directory in the Esmraldi repository. These examples include code embedded in the `Quickstart tutorial`_ and `Advanced example`_, and make it easier to reuse the code.

.. _Quickstart tutorial: quickstart.ipynb
.. _Advanced example: advanced_example.ipynb

The general command is the following: ::

  python -m examples.<module_name> [--options] [--help]

We strongly advise to refer to the output given by the ``help`` argument before using these examples; it describes the example file and its parameters.


The scripts generally expect arguments, usually the input image, as well as other parameters depending on the script. The *\-\-help* argument lists the arguments and provides a short description of them.

The path to the images and output must be given as a path (e.g. **D:\\Data\\image.imzML** for Windows or **/home/user/Data/image.imzML** for Linux).

Arguments have a full version with two dashes (e.g. *\-\-input* and generally a shorter version with a single dash (e.g. *-i*).

Common errors
-------------

::

    No module named xxx

Make sure the path to the module is correct. Modules should be called by prefixing their name with "examples".


::

    FileNotFoundError: [Errno 2]

The path does not exist. Make sure your path to the input is correctly formatted. Make sure the directory exists (create a new directory first).

::

   TypeError: expected str, bytes or os.PathLike object, not NoneType

Two options: a mandatory argument was omitted, or you did not provide the correct type for the argument (float or integer numbers, character strings...). Check the help for more information.


Module organization
===================

Here's a short description of the most useful scripts, along with examples how to call them:

Segmentation
------------

* ``segmentation.py``: applies `Alexandrov et al.'s spatial chaos <https://academic.oup.com/bioinformatics/article/29/18/2335/240053?login=true>`_ to remove noise in images. Segmentation of the tissue is then done by region growing.::

    python -m examples.segmentation -i /home/Data/myimage.tif  --normalization 746.711 -f 1.003 -q 70 75 80 85 90 95 -o /home/Data/spatial_chaos.tif

* ``find_structured_images.py``: find spatially coherent images by applying distance-based methods. Filters out **dispersed** ion images as well as off-sample images. Uses quantile thresholding, similarly to the spatial chaos method. It selects ion images matching the iinput criteria with the best thresholds (_binary.tif)::

    python -m examples.find_structured_images -i /home/Data/myimage.tif --normalization 746.711  -f 0.5 --offsample_threshold 0.1 -q 30 50 60 70 80 90 95 -o /home/Data/structured.tif


Regitration
-----------

* `registration.py`: use intensity similarity based methods for linear registration. Transforms the image using distance transformations.::

    python -m examples.registration -f /home/Data/fixed.tif --moving /home/Data/moving.tif -r /home/Data/moving.tif --relaxation_factor 0.5 --learning_rate 1.5 -s --min_step 0.00001 -o /home/Data/registered.tif --resize

* For **non-linear registration**, use the following repository (C++): `https://github.com/fgrelard/RegistrationUDT <https://github.com/fgrelard/RegistrationUDT>`_.

Example command-line after following installation and build procedures::

   ./bin/VariationalRegistrationDT2D -F /home/Data/fixed.tif -R /home/Data/moving.tif -M /home/Data/moving.tif -W /home/Data/deformed.tif -t 0.5 -r 2  -b 1 -m 0.1 -f 1 -O /home/Data/field.mha -l 1

The deformation field can be applied on multiple ion images, typically in the case of registration of MSI (moving image) onto another modality (reference image) by calling: ::

    python -m examples.apply_itk_transform -i /home/Data/msi.tif -t  /home/Data/field.mha -o /home/Data/deformed_msi.tif


Clustering
----------

* ``clustering.py``: applies k-means pixel clustering to MSI data/ ::

    python -m examples.clustering -i /home/Data/myimage.tif  --cosine -o /home/Data/kmeans.tif -k 15


* ``spatially_aware_clustering.py``: applies `Alexandrov et al.'s spatially-aware clustering <https://link.springer.com/article/10.1007/s00216-021-03179-w>`_ to reduce the impact of noise. The dimensions are reduced first using dimension reduction technique (hence the number of components argument)::

    python -m examples.spatially_aware_clustering -i /home/Data/myimage.tif --cosine  -o /home/Data/spatially_aware.tif -n 30 -k 15 --radius 1

* ``hierarchical_clustering.py``: applies hierarchical clustering to group ion that have similar spatial distributions. Generates the average image for each cluster. A UMAP projection is generated to identify the clusters, along with average image for each cluster. Images can be filtered when they resemble supplied images (using ``--correlation_names``).::

     python -m examples.hierarchical_clustering -i /home/Data/myimage.tif --mds -p  --correlation_names /home/Data/mask1.tif /home/Data/mask2.tif --value 14 --regions  /home/Data/mask1.tif /home/Data/mask2.tif -o /home/Data/hierarchical_clustering_dir/

* ``extract_all_clusters.py``: extracting binary cluster images for each cluster from a labelled cluster image.::

     python -m examples.extract_all_clusters -i /home/Data/myimage.tif -o /home/Data/output_dir/

* ``compare_clustering.py``: identifies the images in common using the distance criterion between two clustering techniques (i.e. spectral vs spatial), provided they have the same number of clusters.::

     python -m examples.compare_clustering -i /home/Data/hierarchical_clustering_dir/av_image* --target /home/Data/output_dir/cluster_* -o /home/Data/comparison.xlsx --value 5


Receiver Operating Characteristic
---------------------------------

* ``roc.py``: main roc script to obtain region-specific ions.::

   python -m examples.roc -i /home/Data/myimage.tif --regions /home/Data/masks/*.tif --normalization 746.711 -o /home/Data/roc.xlsx

* ``roc_ion.py``: display ROC curve for one or several ions. ::

   python -m examples.roc_ion -i /home/Data/myimage.tif   --regions  /home/Data/masks/mymask.tif -n 746.711 --mz 863.56 279.23

* ``roc_display_graph.py``: tSNE visualization of highest AUC ROC for each region, above a given value. Annotations can be supplied by Metaspace. ::

   python -m examples.roc_display_graph -i /home/Data/roc.xlsx -v 0.8 --annotations /home/Data/metaspace_annotations.csv 

* ``roc_best_images.py``: ion image visualizer which sorts AUC ROC values by descending order.::

   python -m examples.roc_best_images -i /home/Data/myimage.tif  -n 746.711 --roc /home/Data/roc.xls --names My\ region


Supervised learning
-------------------

* ``create_image_for_pls.py``: script to generate a dataset used for learning, with subsampling specified with *sample_size* argument. ::

    python -m create_image_for_pls -i /home/Data/dataset1/peakpicked.imzML /home/Data/dataset2/peakpicked.imzML  --regions /home/Data/dataset1/masks/resized/*.tif --regions /home/Data/dataset1/masks/resized/*.tif  --sample_size 1000 -o /home/Data/train/train_dataset.tif --normalization


* ``pls.py``: Training with LASSO or PLS. Expects either an *alpha* value for Lasso or *nb_component* for PLS. Generates a model file (joblib extension).::

    python -m examples.pls -i /home/Data/train/train_dataset.tif -r /home/Data/train/regions/*.tif  -o /home/Data/models/model.joblib --lasso --alpha 0.002

* ``bootstrap_models.py``: Combining several Lasso or PLS on different trained models.::

    python -m examples.bootstrap_model -i /home/Data/models/model1.joblib /home/Data/models/model2.joblib --lasso -o /home/Data/combination.joblib

* ``evaluate_models.py``: Validation of the model, using a validation dataset (should be created first using ``create_image_for_pls.py``). ::

    python -m examples.evaluate_models -i /home/Data/models/ --validation_dataset /home/Data/validation/validation.tif --lasso 


* ``model_assign_gmm.py``: fit a gaussian mixture model (GMM) onto the trained model, to obtain an "uncertain" class. ::

    python -m examples.model_assign_gmm -i /home/Data/models/model.joblib --msi /home/Data/train/train.tif --names Binder1 Binder2 Binder3 -o /home/Data/models/model_gmm.joblib


* ``pls_test.py``: Applies the previously trained model to dataset to get predictions. Can use a GMM, and specify a probability to assign to the "Uncertain" class (*proba* argument)::

    python -m examples.pls_test -i /home/Data/models/model.joblib -t /home/Data/dataset3/peakpicked.imzML  --gmm  /home/Data/models/model_gmm.joblib --names Binder1 Binder2 Binder3 -o /home/Data/dataset3/prediction.png --proba 0.95 --normalization


* ``evaluation_prediction_confusion.py``: Get sensibility, specificity, precision (and more) matrices, typically for a training dataset. ::

   python -m examples.evaluation_prediction_confusion -i /home/Data/models/model.joblib -t /home/Data/dataset1/peakpicked.imzML -o /home/Data/evaluation.xlsx --names Binder1 Binder2 Binder3 --normalization  --gmm /home/Data/models/model_gmm.joblib --proba 0.95


* ``compare_prediction.py``: viewer to compare predictions across various parameters. ::

    python -m examples.compare_prediction -i /home/Data/models/ --parameters 0.01 0.02 0.03 0.04 --keys P2D3 P2F4 P2D6


* ``display_model_graph_from_dataset.py``: Display a tSNE projection of the Gaussian Mixture Model of the training dataset. ::

    python -m examples.display_model_graph_from_dataset -i /home/Data/models/model.joblib --msi /home/Data/train/train.tif --names Binder1 Binder2 Binder3 --gmm /home/Data/models/model_gmm.joblib


Misc
----
* ``deisotoping.py``: performs MSI deisotoping. ::

    python -m examples.deisotoping -i /home/Data/peakpicked.imzML -o /home/Data/deisotoped.imzML

* ``extract_mean_spectra.py``: extracts various statistics (averages, medians, std, n) for each ion image. ::

    python -m examples.extract_mean_spectra -i /home/Data/peakpicked.imzML --regions /home/Data/regions/*.tif -n 746.711 -o /home/Data/stats.xlsx


* ``intersection_image.py``: Combines two m/z lists and creates images of their intersection and difference. The optional *thresholds* argument is optional and expects a filename containing percentile thresholds for each ion image, such that the output images will be thresholded according to this value. ::

    python -m examples.intersection_image -i /home/Data/msi.tif --first /home/Data/peaklist1.csv --second /home/Data/peaklist2.csv -o /home/Data/combination.tif --thresholds ~/Data/Rate4#35/segmentation/dispersion/spatialcoherence_values.csv

* ``quant_linear_regression.py``: generates a summary for quantitative MSI. The *mask* argument expects an image of the multiple regions of the mimetic. The *peak_list* arguments expects a formatted Excel file with concentrations matching each selected area from the *mask*. Finally the *tissue_regions* argument expects segmented regions from which the average intensity and concentrations are derived. It is possible to enable weighted linear regression  by adding the *weight* argument: ::

    python -m examples.quant_linear_regression -i /home/Data/data.imzml --peak_list /home/Data/drug_list.xlsx --mask /home/Data/mask.tif --normalization -1  -o /home/Data/output_quantification.xlsx --tissue_regions /home/Data/Regions/*.tif 
