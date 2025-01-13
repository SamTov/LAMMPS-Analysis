"""MDSuite run module.

This program and the accompanying materials are made available under the terms of the
Eclipse Public License v2.0 which accompanies this distribution, and is available at
https://www.eclipse.org/legal/epl-v20.html
SPDX-License-Identifier: EPL-2.0.

Copyright Contributors to the Zincware Project.

Description: Collection of calculators / transformations for exp.run
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, List, Type, Union

from mdsuite.calculators import (
    AngularDistributionFunction,  # SpatialDistributionFunction,
)
from mdsuite.calculators import (  # StructureFactor,
    CoordinationNumbers,
    EinsteinDiffusionCoefficients,
    EinsteinDistinctDiffusionCoefficients,
    EinsteinHelfandIonicConductivity,
    EinsteinHelfandThermalConductivity,
    EinsteinHelfandThermalKinaci,
    GreenKuboDiffusionCoefficients,
    GreenKuboDistinctDiffusionCoefficients,
    GreenKuboIonicConductivity,
    GreenKuboThermalConductivity,
    GreenKuboViscosity,
    GreenKuboViscosityFlux,
    KirkwoodBuffIntegral,
    NernstEinsteinIonicConductivity,
    PotentialOfMeanForce,
    RadialDistributionFunction,
)
from mdsuite.transformations import (
    CoordinateUnwrapper,
    CoordinateWrapper,
    IntegratedHeatCurrent,
    IonicCurrent,
    KinaciIntegratedHeatCurrent,
    MolecularMap,
    MomentumFlux,
    ScaleCoordinates,
    ThermalFlux,
    Transformations,
    TranslationalDipoleMoment,
    UnwrapViaIndices,
    VelocityFromPositions,
)

if TYPE_CHECKING:
    from mdsuite.experiment import Experiment


class RunComputation:
    """Collection of all calculators that can be used by an experiment."""

    def __init__(
        self, experiment: Experiment = None, experiments: List[Experiment] = None
    ):
        """Collection of all calculators.

        Parameters
        ----------
        experiment: Experiment
            Experiment to run the computations for
        experiments: List[Experiment]
            A list of experiments passed by running the computation from the project
            class

        """
        self.experiment = experiment
        self.experiments = experiments

        self.kwargs = {"experiment": experiment, "experiments": experiments}

    def exp_wrapper(self, func):
        # preparation for https://github.com/zincware/MDSuite/issues/404
        # currently this does nothing
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_instance = func(*args, **kwargs)
            # self.experiment._run(func_instance)
            return func_instance

        return wrapper

    def transformation_wrapper(self, func: Union[Type[Transformations], Any]):
        """Run the transformation for every selected experiment.

        Parameters
        ----------
        func: a transformation to be attached to the experiment/s

        """

        @functools.wraps(func.run_transformation)
        def wrapper(*args, **kwargs):
            if self.experiments is None:
                self.experiments = [self.experiment]
            for experiment in self.experiments:
                func_instance = func()
                # attach the transformation to the experiment
                experiment.cls_transformation_run(func_instance, *args, **kwargs)

        return wrapper

    #######################
    ### Transformations ###
    #######################

    @property
    def CoordinateWrapper(self) -> Type[CoordinateWrapper]:
        return self.transformation_wrapper(CoordinateWrapper)

    @property
    def CoordinateUnwrapper(self) -> Type[CoordinateUnwrapper]:
        return self.transformation_wrapper(CoordinateUnwrapper)

    @property
    def IntegratedHeatCurrent(self) -> Type[IntegratedHeatCurrent]:
        return self.transformation_wrapper(IntegratedHeatCurrent)

    @property
    def IonicCurrent(self) -> Type[IonicCurrent]:
        return self.transformation_wrapper(IonicCurrent)

    @property
    def KinaciIntegratedHeatCurrent(self) -> Type[KinaciIntegratedHeatCurrent]:
        return self.transformation_wrapper(KinaciIntegratedHeatCurrent)

    @property
    def MolecularMap(self) -> Type[MolecularMap]:
        return self.transformation_wrapper(MolecularMap)

    @property
    def MomentumFlux(self) -> Type[MomentumFlux]:
        return self.transformation_wrapper(MomentumFlux)

    @property
    def ScaleCoordinates(self) -> Type[ScaleCoordinates]:
        return self.transformation_wrapper(ScaleCoordinates)

    @property
    def ThermalFlux(self) -> Type[ThermalFlux]:
        return self.transformation_wrapper(ThermalFlux)

    @property
    def TranslationalDipoleMoment(self) -> Type[TranslationalDipoleMoment]:
        return self.transformation_wrapper(TranslationalDipoleMoment)

    @property
    def UnwrapViaIndices(self) -> Type[UnwrapViaIndices]:
        return self.transformation_wrapper(UnwrapViaIndices)

    @property
    def VelocityFromPositions(self) -> Type[VelocityFromPositions]:
        return self.transformation_wrapper(VelocityFromPositions)

    #####################
    #### Calculators ####
    #####################
    @property
    def AngularDistributionFunction(self) -> AngularDistributionFunction:
        return self.exp_wrapper(AngularDistributionFunction)(**self.kwargs)

    @property
    def CoordinationNumbers(self) -> CoordinationNumbers:
        return self.exp_wrapper(CoordinationNumbers)(**self.kwargs)

    @property
    def EinsteinDiffusionCoefficients(self) -> EinsteinDiffusionCoefficients:
        return self.exp_wrapper(EinsteinDiffusionCoefficients)(**self.kwargs)

    @property
    def EinsteinDistinctDiffusionCoefficients(
        self,
    ) -> EinsteinDistinctDiffusionCoefficients:
        return self.exp_wrapper(EinsteinDistinctDiffusionCoefficients)(**self.kwargs)

    @property
    def EinsteinHelfandIonicConductivity(self) -> EinsteinHelfandIonicConductivity:
        return self.exp_wrapper(EinsteinHelfandIonicConductivity)(**self.kwargs)

    @property
    def EinsteinHelfandThermalKinaci(self) -> EinsteinHelfandThermalKinaci:
        return self.exp_wrapper(EinsteinHelfandThermalKinaci)(**self.kwargs)

    @property
    def GreenKuboViscosityFlux(self) -> GreenKuboViscosityFlux:
        return self.exp_wrapper(GreenKuboViscosityFlux)(**self.kwargs)

    @property
    def GreenKuboDistinctDiffusionCoefficients(
        self,
    ) -> GreenKuboDistinctDiffusionCoefficients:
        return self.exp_wrapper(GreenKuboDistinctDiffusionCoefficients)(**self.kwargs)

    @property
    def GreenKuboIonicConductivity(self) -> GreenKuboIonicConductivity:
        return self.exp_wrapper(GreenKuboIonicConductivity)(**self.kwargs)

    @property
    def GreenKuboDiffusionCoefficients(self) -> GreenKuboDiffusionCoefficients:
        return self.exp_wrapper(GreenKuboDiffusionCoefficients)(**self.kwargs)

    @property
    def GreenKuboThermalConductivity(self) -> GreenKuboThermalConductivity:
        return self.exp_wrapper(GreenKuboThermalConductivity)(**self.kwargs)

    @property
    def GreenKuboViscosity(self) -> GreenKuboViscosity:
        return self.exp_wrapper(GreenKuboViscosity)(**self.kwargs)

    @property
    def KirkwoodBuffIntegral(self) -> KirkwoodBuffIntegral:
        return self.exp_wrapper(KirkwoodBuffIntegral)(**self.kwargs)

    @property
    def NernstEinsteinIonicConductivity(self) -> NernstEinsteinIonicConductivity:
        return self.exp_wrapper(NernstEinsteinIonicConductivity)(**self.kwargs)

    @property
    def PotentialOfMeanForce(self) -> PotentialOfMeanForce:
        return self.exp_wrapper(PotentialOfMeanForce)(**self.kwargs)

    @property
    def RadialDistributionFunction(self) -> RadialDistributionFunction:
        return self.exp_wrapper(RadialDistributionFunction)(**self.kwargs)

    # @property
    # def StructureFactor(self) -> StructureFactor:
    #     return self.exp_wrapper(StructureFactor)(**self.kwargs)

    @property
    def EinsteinHelfandThermalConductivity(self) -> EinsteinHelfandThermalConductivity:
        return self.exp_wrapper(EinsteinHelfandThermalConductivity)(**self.kwargs)

    # @property
    # def SpatialDistributionFunction(self) -> SpatialDistributionFunction:
    #     return self.exp_wrapper(SpatialDistributionFunction)(**self.kwargs)
