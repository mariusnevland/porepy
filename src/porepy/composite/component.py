"""Contains the abstract base class for all components (species/ pure substances)
used in this framework.
"""

from __future__ import annotations

import abc
from typing import Union

import porepy as pp

__all__ = ["Component", "FluidComponent", "SolidComponent"]

VarLike = Union[pp.ad.MergedVariable, pp.ad.Variable, pp.ad.Operator, float, int]


class Component(abc.ABC):
    """Abstract base class for chemical components, providing abstract physical properties
    which need to be implemented for concrete child classes to work in PorePy.

    Provides and manages component-related physical quantities and properties.

    The AD framework is is used to represent the fractional variables and the component class
    becomes a singleton with respect to the mixed-dimensional domain contained in the
    AD system.
    Unique instantiation over a given domain is assured by using this class's name as an unique
    identifier.
    Ambiguities and uniqueness must be assured due to central storage of the fractional values
    in the grid data dictionaries.

    Parameters:
        ad_system: AD system in which this component is present cell-wise in each subdomain.

    """

    # contains per mdg the singleton, using the class name as a unique identifier
    __ad_singletons: dict[pp.MixedDimensionalGrid, dict[str, Component]] = dict()
    # flag if a singleton has recently been accessed, to skip re-instantiation
    __singleton_accessed: bool = False

    def __new__(cls, ad_system: pp.ad.ADSystem) -> Component:
        # class name is used as unique identifier
        name = str(cls.__name__)
        # check for AD singletons per grid
        mdg = ad_system.dof_manager.mdg
        if mdg in Component.__ad_singletons:
            if name in Component.__ad_singletons[mdg]:
                # flag that the singleton has been accessed and return it.
                Component.__singleton_accessed = True
                return Component.__ad_singletons[mdg][name]
        else:
            Component.__ad_singletons.update({mdg: dict()})

        # create a new instance and store it, if no previous instantiations were found
        new_instance = super().__new__(cls)
        Component.__ad_singletons[mdg].update({name: new_instance})

        return new_instance

    def __init__(self, ad_system: pp.ad.ADSystem) -> None:

        # skipping re-instantiation if class if __new__ returned the previous reference
        if Component.__singleton_accessed:
            Component.__singleton_accessed = False
            return

        super().__init__()

        ### PUBLIC

        self.ad_system: pp.ad.ADSystem = ad_system
        """The AD system passed at instantiation."""

        #### PRIVATE

        # creating the overall molar fraction variable
        self._fraction: pp.ad.MergedVariable = ad_system.create_variable(self.fraction_name)

    @property
    def name(self) -> str:
        """
        Returns: name of the class, used as a unique identifier.

        """
        return str(self.__class__.__name__)

    @property
    def fraction_name(self) -> str:
        """Name of the feed fraction variable, given by the general symbol and :meth:`name`."""
        return "z" + "_" + self.name

    @property
    def fraction(self) -> pp.ad.MergedVariable:
        """
        | Math. Dimension:        scalar
        | Phys. Dimension:        [-] fractional

        Returns:
            feed fraction, a primary variable on the whole domain (cell-wise).
            Indicates how many of the total moles belong to this component.

        """
        return self._fraction

    ### PHYSICAL PROPERTIES -------------------------------------------------------------------

    def mass_density(self, p: VarLike, T: VarLike) -> VarLike:
        """Uses the molar mass and molar density to compute the mass density.

        | Math. Dimension:        scalar
        | Phys. Dimension:        [kg / REV]

        Parameters:
            p: pressure
            T: temperature

        Returns: mass density.

        """
        return self.molar_mass() * self.density(p, T)

    @staticmethod
    @abc.abstractmethod
    def molar_mass() -> float:
        """This is a constant value, hence to be a static function.

        | Math. Dimension:        scalar
        | Phys. Dimension:        [kg / mol]

        Returns: molar mass.

        """
        pass

    @abc.abstractmethod
    def density(self, p: VarLike, T: VarLike) -> VarLike:
        """
        | Math. Dimension:        scalar
        | Phys. Dimension:        [mol / REV]

        Parameters:
            p: pressure
            T: temperature

        Returns: molar density.

        """
        pass

    @abc.abstractmethod
    def Fick_diffusivity(self, p: VarLike, T: VarLike) -> VarLike:
        """
        Notes:
            This can also be a tensor and will be submitted respective changes in the future.

        | Math. Dimension:        scalar
        | Phys. Dimension:        m^2 / s

        Parameters:
            p: pressure
            T: temperature

        Returns: Fick diffusivity coefficient.

        """
        pass

    @abc.abstractmethod
    def thermal_conductivity(self, p: VarLike, T: VarLike) -> VarLike:
        """
        Notes:
            This can also be a tensor and will be submitted respective changes in the future.

        | Math. Dimension:        scalar
        | Phys. Dimension:        [W / m / K]

        Parameters:
            p: pressure
            T: temperature

        Returns: thermal conductivity for Fourier's law.

        """
        pass


class FluidComponent(Component):
    """Extends the list of necessary physical attributes for components by those which are
    usually used for fluids in flow problems.

    """

    @staticmethod
    @abc.abstractmethod
    def critical_pressure() -> float:
        """This is a constant value, hence to be a static function.

        | Math. Dimension:        scalar
        | Phys. Dimension:        [kPa]

        Returns: critical pressure for this component (critical point in p-T diagram).

        """
        pass

    @staticmethod
    @abc.abstractmethod
    def critical_temperature() -> float:
        """This is a constant value, hence to be a static function.

        | Math. Dimension:        scalar
        | Phys. Dimension:        [K]

        Returns: critical temperature for this component (critical point in p-T diagram).

        """
        pass

    @staticmethod
    @abc.abstractmethod
    def triple_point_pressure() -> float:
        """This is a constant value, hence to be a static function.

        | Math. Dimension:        scalar
        | Phys. Dimension:        [kPa]

        Returns:
            triple point pressure for this component
            (intersection of vapor and melting curve in p-T diagram).

        """
        pass

    @staticmethod
    @abc.abstractmethod
    def triple_point_temperature() -> float:
        """This is a constant value, hence to be a static function.

        | Math. Dimension:        scalar
        | Phys. Dimension:        [K]

        Returns:
            triple point temperature for this component
            (intersection of vapor and melting curve in p-T diagram).

        """
        pass

    @abc.abstractmethod
    def dynamic_viscosity(self, p: VarLike, T: VarLike) -> VarLike:
        """
        | Math. Dimension:        scalar
        | Phys. Dimension:        [Pa s] = [kg / m / s]

        Args:
            p: pressure
            T: temperature

        Returns: dynamic viscosity.

        """
        pass


class SolidComponent(Component):
    """Extends the list of necessary physical attributes for components by those which are
    usually used for solids in elasticity and plasticity problems, or which are used to
    for the porous medium.

    """

    @staticmethod
    @abc.abstractmethod
    def base_porosity() -> float:
        """Constant value, hence to be a static function.

        | Math. Dimension:        scalar
        | Phys. Dimension:        [-] fractional

        Returns: base porosity of the material.

        """
        pass

    @staticmethod
    @abc.abstractmethod
    def base_permeability() -> float:
        """Constant value, hence to be a static function.

        | Math. Dimension:        scalar
        | Phys. Dimension:        [m^2] ( [Darcy] not official SI unit)

        Returns: base permeability of the material.

        """
        pass
