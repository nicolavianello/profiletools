# Copyright 2017 Nicola Vianello
# This is an implementation of reading routine for TCV
# in order to be used with the profiletools and Gaussian Process Regression
#
# This program is distributed under the terms of the
# GNU General Purpose License (GPL).
# Refer to http://www.gnu.org/licenses/gpl.txt
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Provides classes for working with TCV data via MDSplus.
"""

from __future__ import division

from .core import Profile, Channel, read_csv, read_NetCDF
from . import transformations
from . import elm_detection
import warnings

try:
    import MDSplus
except ImportError:
    warnings.warn("Module MDSplus could not be loaded!", RuntimeWarning)
try:
    import eqtools
except ImportError:
    warnings.warn("Module eqtools could not be loaded!", RuntimeWarning)
import scipy
import scipy.interpolate
import scipy.stats
import gptools
import matplotlib.pyplot as plt
import numpy as np

try:
    import TRIPPy
except ImportError:
    warnings.warn("Module TRIPPy could not be loaded!", RuntimeWarning)

_X_label_mapping = {
    "psinorm": r"$\psi_n$",
    "phinorm": r"$\phi_n$",
    "volnorm": r"$V_n$",
    "Rmid": r"$R_{mid}$",
    "r/a": "$r/a$",
    "sqrtpsinorm": r"$\sqrt{\psi_n}$",
    "sqrtphinorm": r"$\sqrt{\phi_n}$",
    "sqrtvolnorm": r"$\sqrt{V_n}$",
    "sqrtr/a": r"$\sqrt{r/a}$",
}
_abscissa_mapping = {y: x for x, y in _X_label_mapping.items()}
_X_unit_mapping = {
    "psinorm": "",
    "phinorm": "",
    "volnorm": "",
    "Rmid": "m",
    "r/a": "",
    "sqrtpsinorm": "",
    "sqrtphinorm": "",
    "sqrtvolnorm": "",
    "sqrtr/a": "",
}


class BivariatePlasmaProfile(Profile):
    """Class to represent bivariate (y=f(t, psi)) plasma data.

    The first column of `X` is always time. If the abscissa is 'RZ', then the
    second column is `R` and the third is `Z`. Otherwise the second column is
    the desired abscissa (psinorm, etc.).
    """

    def remake_efit_tree(self):
        """Remake the EFIT tree.

        This is needed since EFIT tree instances aren't pickleable yet, so to
        store a :py:class:`BivariatePlasmaProfile` in a pickle file, you must
        delete the EFIT tree.
        """
        self.efit_tree = eqtools.TCVLIUQEMATTree(self.shot)

    def convert_abscissa(self, new_abscissa, drop_nan=True, ddof=1):
        """Convert the internal representation of the abscissa to new coordinates.

        The target abcissae are what are supported by `rho2rho` from the
        `eqtools` package. Namely,

            ======= ========================
            psinorm Normalized poloidal flux
            phinorm Normalized toroidal flux
            volnorm Normalized volume
            Rmid    Midplane major radius
            r/a     Normalized minor radius
            ======= ========================

        Additionally, each valid option may be prepended with 'sqrt' to return
        the square root of the desired normalized unit.

        Parameters
        ----------
        new_abscissa : str
            The new abscissa to convert to. Valid options are defined above.
        drop_nan : bool, optional
            Set this to True to drop any elements whose value is NaN following
            the conversion. Default is True (drop NaN elements).
        ddof : int, optional
            Degree of freedom correction to use when time-averaging a
            conversion.
        """
        if self.abscissa == new_abscissa:
            return
        elif self.X_dim == 1 or (self.X_dim == 2 and self.abscissa == "RZ"):
            if self.abscissa.startswith("sqrt") and self.abscissa[4:] == new_abscissa:
                if self.X is not None:
                    new_rho = scipy.power(self.X[:, 0], 2)
                    # Approximate form from uncertainty propagation:
                    err_new_rho = self.err_X[:, 0] * 2 * self.X[:, 0]

                # Handle transformed quantities:
                for p in self.transformed:
                    p.X[:, :, 0] = scipy.power(p.X[:, :, 0], 2)
                    p.err_X[:, :, 0] = p.err_X[:, :, 0] * 2 * p.X[:, :, 0]
            elif new_abscissa.startswith("sqrt") and self.abscissa == new_abscissa[4:]:
                if self.X is not None:
                    new_rho = scipy.power(self.X[:, 0], 0.5)
                    # Approximate form from uncertainty propagation:
                    err_new_rho = self.err_X[:, 0] / (2 * scipy.sqrt(self.X[:, 0]))

                # Handle transformed quantities:
                for p in self.transformed:
                    p.X[:, :, 0] = scipy.power(p.X[:, :, 0], 0.5)
                    p.err_X[:, :, 0] = p.err_X[:, :, 0] / (2 * scipy.sqrt(p.X[:, :, 0]))
            else:
                times = self._get_efit_times_to_average()

                if self.abscissa == "RZ":
                    if self.X is not None:
                        new_rhos = self.efit_tree.rz2rho(
                            new_abscissa, self.X[:, 0], self.X[:, 1], times, each_t=True
                        )
                        self.channels = self.channels[:, 0:1]
                    self.X_dim = 1

                    # Handle transformed quantities:
                    for p in self.transformed:
                        new_rhos = self.efit_tree.rz2rho(
                            new_abscissa, p.X[:, :, 0], p.X[:, :, 1], times, each_t=True
                        )
                        p.X = scipy.delete(p.X, 1, axis=2)
                        p.err_X = scipy.delete(p.err_X, 1, axis=2)
                        p.X[:, :, 0] = scipy.atleast_3d(scipy.mean(new_rhos, axis=0))
                        p.err_X[:, :, 0] = scipy.atleast_3d(
                            scipy.std(new_rhos, axis=0, ddof=ddof)
                        )
                        p.err_X[scipy.isnan(p.err_X)] = 0
                else:
                    if self.X is not None:
                        new_rhos = self.efit_tree.rho2rho(
                            self.abscissa,
                            new_abscissa,
                            self.X[:, 0],
                            times,
                            each_t=True,
                        )

                    # Handle transformed quantities:
                    for p in self.transformed:
                        new_rhos = self.efit_tree.rho2rho(
                            self.abscissa,
                            new_abscissa,
                            p.X[:, :, 0],
                            times,
                            each_t=True,
                        )
                        p.X[:, :, 0] = scipy.atleast_3d(scipy.mean(new_rhos, axis=0))
                        p.err_X[:, :, 0] = scipy.atleast_3d(
                            scipy.std(new_rhos, axis=0, ddof=ddof)
                        )
                        p.err_X[scipy.isnan(p.err_X)] = 0
                if self.X is not None:
                    new_rho = scipy.mean(new_rhos, axis=0)
                    err_new_rho = scipy.std(new_rhos, axis=0, ddof=ddof)
                    err_new_rho[scipy.isnan(err_new_rho)] = 0

            if self.X is not None:
                self.X = scipy.atleast_2d(new_rho).T
                self.err_X = scipy.atleast_2d(err_new_rho).T
            self.X_labels = [_X_label_mapping[new_abscissa]]
            self.X_units = [_X_unit_mapping[new_abscissa]]
        else:
            if self.abscissa.startswith("sqrt") and self.abscissa[4:] == new_abscissa:
                if self.X is not None:
                    new_rho = scipy.power(self.X[:, 1], 2)

                # Handle transformed quantities:
                for p in self.transformed:
                    p.X[:, :, 1] = scipy.power(p.X[:, :, 1], 2)
                    p.err_X[:, :, 1] = scipy.zeros_like(p.X[:, :, 1])
            elif new_abscissa.startswith("sqrt") and self.abscissa == new_abscissa[4:]:
                if self.X is not None:
                    new_rho = scipy.power(self.X[:, 1], 0.5)

                # Handle transformed quantities:
                for p in self.transformed:
                    p.X[:, :, 1] = scipy.power(p.X[:, :, 1], 0.5)
                    p.err_X[:, :, 1] = scipy.zeros_like(p.X[:, :, 1])
            elif self.abscissa == "RZ":
                # Need to handle this case separately because of the extra
                # column:
                if self.X is not None:
                    new_rho = self.efit_tree.rz2rho(
                        new_abscissa,
                        self.X[:, 1],
                        self.X[:, 2],
                        self.X[:, 0],
                        each_t=False,
                    )
                    self.channels = self.channels[:, 0:2]
                self.X_dim = 2

                # Handle transformed quantities:
                for p in self.transformed:
                    p.X[:, :, 1] = self.efit_tree.rz2rho(
                        new_abscissa,
                        p.X[:, :, 1],
                        p.X[:, :, 2],
                        p.X[:, :, 0],
                        each_t=False,
                    )
                    p.X = scipy.delete(p.X, 2, axis=2)
                    p.err_X = scipy.delete(p.err_X, 2, axis=2)
                    p.err_X[:, :, 1] = scipy.zeros_like(p.X[:, :, 1])
            else:
                if self.X is not None:
                    new_rho = self.efit_tree.rho2rho(
                        self.abscissa,
                        new_abscissa,
                        self.X[:, 1],
                        self.X[:, 0],
                        each_t=False,
                    )

                # Handle transformed quantities:
                for p in self.transformed:
                    p.X[:, :, 1] = self.efit_tree.rho2rho(
                        self.abscissa,
                        new_abscissa,
                        p.X[:, :, 1],
                        p.X[:, :, 0],
                        each_t=False,
                    )
                    p.err_X[:, :, 1] = scipy.zeros_like(p.X[:, :, 1])

            if self.X is not None:
                err_new_rho = scipy.zeros_like(self.X[:, 0])

                self.X = scipy.hstack(
                    (scipy.atleast_2d(self.X[:, 0]).T, scipy.atleast_2d(new_rho).T)
                )
                self.err_X = scipy.hstack(
                    (
                        scipy.atleast_2d(self.err_X[:, 0]).T,
                        scipy.atleast_2d(err_new_rho).T,
                    )
                )

            self.X_labels = [self.X_labels[0], _X_label_mapping[new_abscissa]]
            self.X_units = [self.X_units[0], _X_unit_mapping[new_abscissa]]
        self.abscissa = new_abscissa
        if drop_nan and self.X is not None:
            self.remove_points(scipy.isnan(self.X).any(axis=1))

    def time_average(self, **kwargs):
        """Compute the time average of the quantity.

        Stores the original bounds of `t` to `self.t_min` and `self.t_max`.

        All parameters are passed to :py:meth:`average_data`.
        """
        if self.X is not None:
            self.t_min = self.X[:, 0].min()
            self.t_max = self.X[:, 0].max()
        if len(self.transformed) > 0:
            t_min_T = min([p.X[:, :, 0].min() for p in self.transformed])
            t_max_T = max([p.X[:, :, 0].max() for p in self.transformed])
            if self.X is None:
                self.t_min = t_min_T
                self.t_max = t_max_T
            else:
                self.t_min = min(self.t_min, t_min_T)
                self.t_max = max(self.t_max, t_max_T)
        self.average_data(axis=0, **kwargs)

    def drop_axis(self, axis):
        """Drops a selected axis from `X`.

        Parameters
        ----------
        axis : int
            The index of the axis to drop.
        """
        if self.X_labels[axis] == "$t$":
            if self.X is not None:
                self.t_min = self.X[:, 0].min()
                self.t_max = self.X[:, 0].max()
            if len(self.transformed) > 0:
                t_min_T = min([p.X[:, :, 0].min() for p in self.transformed])
                t_max_T = max([p.X[:, :, 0].max() for p in self.transformed])
                if self.X is None:
                    self.t_min = t_min_T
                    self.t_max = t_max_T
                else:
                    self.t_min = min(self.t_min, t_min_T)
                    self.t_max = max(self.t_max, t_max_T)
        super(BivariatePlasmaProfile, self).drop_axis(axis)

    def keep_times(self, times, **kwargs):
        """Keeps only the nearest points to vals along the time axis for each channel.

        Parameters
        ----------
        times : array of float
            The values the time should be close to.
        **kwargs : optional kwargs
            All additional kwargs are passed to
            :py:meth:`~profiletools.core.Profile.keep_slices`.
        """
        if self.X_labels[0] != "$t$":
            raise ValueError("Cannot keep specific time slices after time-averaging!")
        try:
            iter(times)
        except TypeError:
            times = [times]
        self.times = times
        self.keep_slices(0, times, **kwargs)

    def add_profile(self, other):
        """Absorbs the data from another profile object.

        Parameters
        ----------
        other : :py:class:`Profile`
            :py:class:`Profile` to absorb.
        """
        # Warn about merging profiles from different shots:
        if self.shot != other.shot:
            warnings.warn(
                "Merging data from two different shots: %d and %d"
                % (self.shot, other.shot)
            )
        other.convert_abscissa(self.abscissa)
        # Split off the diagnostic description when merging profiles:
        super(BivariatePlasmaProfile, self).add_profile(other)
        if self.y_label != other.y_label:
            self.y_label = self.y_label.split(", ")[0]

    def remove_edge_points(self, allow_conversion=True):
        """Removes points that are outside the LCFS.

        Must be called when the abscissa is a normalized coordinate. Assumes
        that the last column of `self.X` is space: so it will do the wrong
        thing if you have already taken an average along space.

        Parameters
        ----------
        allow_conversion : bool, optional
            If True and self.abscissa is 'RZ', then the profile will be
            converted to psinorm and the points will be dropped. Default is True
            (allow conversion).
        """
        if self.X is not None:
            if self.abscissa == "RZ":
                if allow_conversion:
                    warnings.warn(
                        "Removal of edge points not supported with abscissa RZ. Will "
                        "convert to psinorm."
                    )
                    self.convert_abscissa("psinorm")
                else:
                    raise ValueError(
                        "Removal of edge points not supported with abscissa RZ!"
                    )
            if "r/a" in self.abscissa or "norm" in self.abscissa:
                x_out = 1.0
            elif self.abscissa == "Rmid":
                if self.X_dim == 1:
                    t_EFIT = self._get_efit_times_to_average()
                    x_out = scipy.mean(self.efit_tree.getRmidOutSpline()(t_EFIT))
                else:
                    assert self.X_dim == 2
                    x_out = self.efit_tree.getRmidOutSpline()(
                        scipy.asarray(self.X[:, 0]).ravel()
                    )
            else:
                raise ValueError(
                    "Removal of edge points not supported with abscissa %s!"
                    % (self.abscissa,)
                )
            self.remove_points((self.X[:, -1] >= x_out) | scipy.isnan(self.X[:, -1]))

    def constrain_slope_on_axis(self, err=0, times=None):
        """Constrains the slope at the magnetic axis of this Profile's Gaussian process to be zero.

        Note that this is accomplished approximately for bivariate data by
        specifying the slope to be zero at the magnetic axis for a number of
        points in time.

        It is assumed that the Gaussian process has already been created with
        a call to :py:meth:`create_gp`.

        It is required that the abscissa be either Rmid or one of the
        normalized coordinates.

        Parameters
        ----------
        err : float, optional
            The uncertainty to place on the slope constraint. The default is 0
            (slope constraint is exact). This could also potentially be an
            array for bivariate data where you wish to have the uncertainty
            vary in time.
        times : array-like, optional
            The times to impose the constraint at. Default is to use the
            unique time values in `X[:, 0]`.
        """
        if self.X_dim == 1:
            if self.abscissa == "Rmid":
                t_EFIT = self._get_efit_times_to_average()
                x0 = scipy.mean(self.efit_tree.getMagRSpline()(t_EFIT))
            elif "norm" in self.abscissa or "r/a" in self.abscissa:
                x0 = 0
            else:
                raise ValueError(
                    "Magnetic axis slope constraint is not "
                    "supported for abscissa '%s'. Convert to a "
                    "normalized coordinate or Rmid to use this "
                    "constraint." % (self.abscissa,)
                )
            self.gp.add_data(x0, 0, err_y=err, n=1)
        elif self.X_dim == 2:
            if times is None:
                times = scipy.unique(self.X[:, 0])
            if self.abscissa == "Rmid":
                x0 = self.efit_tree.getMagRSpline()(times)
            elif "norm" in self.abscissa or "r/a" in self.abscissa:
                x0 = scipy.zeros_like(times)
            else:
                raise ValueError(
                    "Magnetic axis slope constraint is not "
                    "supported for abscissa '%s'. Convert to a "
                    "normalized coordinate or Rmid to use this "
                    "constraint." % (self.abscissa,)
                )
            y = scipy.zeros_like(x0)
            X = scipy.hstack((scipy.atleast_2d(times).T, scipy.atleast_2d(x0).T))
            n = scipy.tile([0, 1], (len(y), 1))
            self.gp.add_data(X, y, err_y=err, n=n)
        else:
            raise ValueError(
                "Magnetic axis slope constraint is not supported "
                "for X_dim=%d, abscissa '%s'. Convert to a "
                "normalized coordinate or Rmid to use this "
                "constraint." % (self.X_dim, self.abscissa)
            )

    def constrain_at_limiter(
        self, err_y=0.01, err_dy=0.1, times=None, n_pts=4, expansion=1.25
    ):
        """Constrains the slope and value of this Profile's Gaussian
        process to be zero at the GH limiter.

        The specific value of `X` coordinate to impose this constraint at is
        determined by finding the point of the GH limiter which has the
        smallest mapped coordinate.

        If the limiter location is not found in the tree, the system will
        instead use R=0.91m, Z=0.0m as the limiter location. This is a bit
        outside of where the limiter is, but will act as a conservative
        approximation for cases where the exact position is not available.

        Note that this is accomplished approximately for bivariate data by
        specifying the slope and value to be zero at the limiter for a number
        of points in time.

        It is assumed that the Gaussian process has already been created with
        a call to :py:meth:`create_gp`.

        The abscissa cannot be 'Z' or 'RZ'.

        Parameters
        ----------
        err_y : float, optional
            The uncertainty to place on the value constraint. The default is
            0.01. This could also potentially be an array for bivariate data
            where you wish to have the uncertainty vary in time.
        err_dy : float, optional
            The uncertainty to place on the slope constraint. The default is
            0.1. This could also potentially be an array for bivariate data
            where you wish to have the uncertainty vary in time.
        times : array-like, optional
            The times to impose the constraint at. Default is to use the
            unique time values in `X[:, 0]`.
        n_pts : int, optional
            The number of points outside of the limiter to use. It helps to use
            three or more points outside the plasma to ensure appropriate
            behavior. The constraint is applied at `n_pts` linearly spaced
            points between the limiter location (computed as discussed above)
            and the limiter location times `expansion`. If you set this to one
            it will only impose the constraint at the limiter. Default is 4.
        expansion : float, optional
            The factor by which the coordinate of the limiter location is
            multiplied to get the outer limit of the `n_pts` constraint points.
            Default is 1.25.
        """
        if self.abscissa in ["RZ", "Z"]:
            raise ValueError(
                "Limiter constraint is not supported for abscissa '%s'. Convert "
                "to a normalized coordinate or Rmid to use this constraint."
                % (self.abscissa,)
            )
        R_lim, Z_lim = self.get_limiter_locations()
        if self.X_dim == 1:
            t_EFIT = self._get_efit_times_to_average()
            rho_lim = scipy.nanmean(
                self.efit_tree.rz2rho(self.abscissa, R_lim, Z_lim, t_EFIT, each_t=True),
                axis=0,
            )
            xa = rho_lim.min()
            if scipy.isnan(xa):
                warnings.warn(
                    "Could not convert limiter location, limiter constraint "
                    "will NOT be applied!"
                )
                return
            print("limiter location=%g" % (xa,))
            x_pts = scipy.linspace(xa, xa * expansion, n_pts)
            y = scipy.zeros_like(x_pts)
            self.gp.add_data(x_pts, y, err_y=err_y, n=0)
            self.gp.add_data(x_pts, y, err_y=err_dy, n=1)
        elif self.X_dim == 2:
            if times is None:
                times = scipy.unique(scipy.asarray(self.X[:, 0]).ravel())
            rho_lim = self.efit_tree.rz2rho(
                self.abscissa, R_lim, Z_lim, times, each_t=True
            )
            xa = rho_lim.min(axis=1)
            x_pts = scipy.asarray(
                [scipy.linspace(x, x * expansion, n_pts) for x in xa]
            ).flatten()
            times = scipy.tile(times, n_pts)
            X = scipy.hstack((scipy.atleast_2d(times).T, scipy.atleast_2d(x_pts).T))
            y = scipy.zeros_like(x_pts)
            n = scipy.tile([0, 1], (len(y), 1))
            self.gp.add_data(X, y, err_y=err_y, n=0)
            self.gp.add_data(X, y, err_y=err_dy, n=n)
        else:
            raise ValueError(
                "Limiter constraint is not supported for X_dim=%d, abscissa "
                "'%s'. Convert to a normalized coordinate or Rmid to use this "
                "constraint." % (self.X_dim, self.abscissa)
            )

    def remove_quadrature_points_outside_of_limiter(self):
        """Remove any of the quadrature points which lie outside of the limiter.

        This is accomplished by setting their weights to zero. When
        :py:meth:`create_gp` is called, it will call
        :py:meth:`GaussianProcess.condense_duplicates` which will remove any
        points for which all of the weights are zero.

        This only affects the transformed quantities in `self.transformed`.
        """
        if self.abscissa in ["RZ", "Z"]:
            raise ValueError(
                "Removal of quadrature points outside of the limiter is not "
                "supported for abscissa '%s'. Convert to a normalized coordinate "
                "or Rmid to use this method." % (self.abscissa,)
            )
        R_lim, Z_lim = self.get_limiter_locations()
        if self.X_dim == 1:
            # In this case, there is a unique set of times, and we can just find
            # one unique limiter location:
            t_EFIT = self._get_efit_times_to_average()
            rho_lim = scipy.mean(
                self.efit_tree.rz2rho(self.abscissa, R_lim, Z_lim, t_EFIT, each_t=True),
                axis=0,
            )
            xa = rho_lim.min()
            for t in self.transformed:
                t.T[t.X[:, :, 0] > xa] = 0.0
        elif self.X_dim == 2:
            # This case is harder. For each transformed variable, we must find
            # the limiter location at each time value present.
            for t in self.transformed:
                times = scipy.unique(scipy.asarray(t.X[:, :, 0]).ravel())
                rho_lim = self.efit_tree.rz2rho(
                    self.abscissa, R_lim, Z_lim, times, each_t=True
                )
                xa = rho_lim.min(axis=1)
                for t_val, xa_val in zip(times, xa):
                    t.T[(t.X[:, :, 0] == t_val) & (t.X[:, :, 1] > xa_val)] = 0.0
        else:
            raise ValueError(
                "Removal of quadrature points outside of the limiter is not "
                "supported for X_dim=%d, abscissa '%s'. Convert to a normalized "
                "coordinate or Rmid to use this method." % (self.X_dim, self.abscissa)
            )

    def get_limiter_locations(self):
        """Retrieve the location of the limiter from the tree.

        If the data are not there (they are missing for some old shots), use
        R=0.91m, Z=0.0m.
        """
        # Fail back to a conservative position if the limiter data are not in
        # the tree:
        try:
            analysis = MDSplus.Tree("tcv_shot", self.shot)
            R_lim = MDSplus.Data.execute('static("r_t")').getValue().data()
            Z_lim = MDSplus.Data.execute('static("z_t")').getValue().data()
        except BaseException:
            warnings.warn(
                "No limiter data, defaulting to R=0.91, Z=0.0!", RuntimeWarning
            )
            Z_lim = [0.0]
            R_lim = [0.91]
        return R_lim, Z_lim

    def create_gp(
        self,
        constrain_slope_on_axis=True,
        constrain_at_limiter=True,
        axis_constraint_kwargs={},
        limiter_constraint_kwargs={},
        **kwargs
    ):
        """Create a Gaussian process to handle the data.

        Calls :py:meth:`~profiletools.core.Profile.create_gp`, then imposes
        constraints as requested.

        Defaults to using a squared exponential kernel in two dimensions or a
        Gibbs kernel with tanh warping in one dimension.

        Parameters
        ----------
        constrain_slope_on_axis : bool, optional
            If True, a zero slope constraint at the magnetic axis will be
            imposed after creating the gp. Default is True (constrain slope).
        constrain_at_limiter : bool, optional
            If True, a zero slope and value constraint at the GH limiter will
            be imposed after creating the gp. Default is True (constrain at
            axis).
        axis_constraint_kwargs : dict, optional
            The contents of this dictionary are passed as kwargs to
            :py:meth:`constrain_slope_on_axis`.
        limiter_constraint_kwargs : dict, optional
            The contents of this dictionary are passed as kwargs to
            :py:meth:`constrain_at_limiter`.
        **kwargs : optional kwargs
            All remaining kwargs are passed to :py:meth:`Profile.create_gp`.
        """
        # Increase the diagonal factor for multivariate data -- I was having
        # issues with the default level when using slope constraints.
        if self.X_dim > 1 and "diag_factor" not in kwargs:
            kwargs["diag_factor"] = 1e4
        if "k" not in kwargs and self.X_dim == 1:
            kwargs["k"] = "gibbstanh"
        if kwargs.get("k", None) == "gibbstanh":
            # Set the bound on x0 intelligently according to the abscissa:
            if "x0_bounds" not in kwargs:
                kwargs["x0_bounds"] = (
                    (0.87, 0.915) if self.abscissa == "Rmid" else (0.94, 1.1)
                )
        super(BivariatePlasmaProfile, self).create_gp(**kwargs)
        if constrain_slope_on_axis:
            self.constrain_slope_on_axis(**axis_constraint_kwargs)
        if constrain_at_limiter:
            self.constrain_at_limiter(**limiter_constraint_kwargs)

    def compute_a_over_L(
        self,
        X,
        force_update=False,
        plot=False,
        gp_kwargs={},
        MAP_kwargs={},
        plot_kwargs={},
        return_prediction=False,
        special_vals=0,
        special_X_vals=0,
        compute_2=False,
        **predict_kwargs
    ):
        """Compute the normalized inverse gradient scale length.

        Only works on data that have already been time-averaged at the moment.

        Parameters
        ----------
        X : array-like
            The points to evaluate a/L at.
        force_update : bool, optional
            If True, a new Gaussian process will be created even if one already
            exists. Set this if you have added data or constraints since you
            created the Gaussian process. Default is False (use current Gaussian
            process if it exists).
        plot : bool, optional
            If True, a plot of a/L is produced. Default is False (no plot).
        gp_kwargs : dict, optional
            The entries of this dictionary are passed as kwargs to
            :py:meth:`create_gp` if it gets called. Default is {}.
        MAP_kwargs : dict, optional
            The entries of this dictionary are passed as kwargs to
            :py:meth:`find_gp_MAP_estimate` if it gets called. Default is {}.
        plot_kwargs : dict, optional
            The entries of this dictionary are passed as kwargs to `plot` when
            plotting the mean of a/L. Default is {}.
        return_prediction : bool, optional
            If True, the full prediction of the value and gradient are returned
            in a dictionary. Default is False (just return value and stddev of
            a/L).
        special_vals : int, optional
            The number of special return values incorporated into
            `output_transform` that should be dropped before computing a/L. This
            is used so that things like volume averages can be efficiently
            computed at the same time as a/L. Default is 0 (no extra values).
        special_X_vals : int, optional
            The number of special points included in the abscissa that should
            not be included in the evaluation of a/L. Default is 0 (no extra
            values).
        compute_2 : bool, optional
            If True, the second derivative and some derived quantities will be
            computed and added to the output structure (if `return_prediction`
            is True). You should almost always have r/a for your abscissa when
            using this: the expressions for other coordinate systems are not as
            well-vetted. Default is False (don't compute second derivative).
        **predict_kwargs : optional parameters
            All other parameters are passed to the Gaussian process'
            :py:meth:`predict` method.
        """
        # TODO: Add ability to just compute value.
        # TODO: Make finer-grained control over what to return.
        if force_update or self.gp is None:
            self.create_gp(**gp_kwargs)
            if not predict_kwargs.get("use_MCMC", False):
                self.find_gp_MAP_estimate(**MAP_kwargs)
        if self.X_dim == 1:
            # Get GP fit:
            XX = scipy.concatenate((X, X[special_X_vals:]))
            if compute_2:
                XX = scipy.concatenate((XX, X[special_X_vals:]))
            n = scipy.concatenate(
                (scipy.zeros_like(X), scipy.ones_like(X[special_X_vals:]))
            )
            if compute_2:
                n = scipy.concatenate((n, 2 * scipy.ones_like(X[special_X_vals:])))
            out = self.gp.predict(XX, n=n, full_output=True, **predict_kwargs)
            mean = out["mean"]
            cov = out["cov"]
            if predict_kwargs.get("return_mean_func", False) and self.gp.mu is not None:
                mean_func = out["mean_func"]
                std_func = out["std_func"]
                mean_without_func = out["mean_without_func"]
                std_without_func = out["std_without_func"]

            # Ditch the special values:
            special_mean = mean[:special_vals]
            special_cov = cov[:special_vals, :special_vals]
            X = X[special_X_vals:]

            cov = cov[special_vals:, special_vals:]
            mean = mean[special_vals:]
            if predict_kwargs.get("return_mean_func", False) and self.gp.mu is not None:
                mean_func = mean_func[special_vals:]
                std_func = std_func[special_vals:]
                mean_without_func = mean_without_func[special_vals:]
                std_without_func = std_without_func[special_vals:]

            var = scipy.diagonal(cov)
            mean_val = mean[: len(X)]
            var_val = var[: len(X)]
            mean_grad = mean[len(X) : 2 * len(X)]
            var_grad = var[len(X) : 2 * len(X)]
            if predict_kwargs.get("return_mean_func", False) and self.gp.mu is not None:
                mean_func_val = mean_func[: len(X)]
                std_func_val = std_func[: len(X)]
                mean_func_grad = mean_func[len(X) :]
                std_func_grad = std_func[len(X) :]

                mean_without_func_val = mean_without_func[: len(X)]
                std_without_func_val = std_without_func[: len(X)]
                mean_without_func_grad = mean_without_func[len(X) :]
                std_without_func_grad = std_without_func[len(X) :]
            if compute_2:
                mean_2 = mean[2 * len(X) :]
                var_2 = var[2 * len(X) :]
            i = range(0, len(X))
            j = range(len(X), 2 * len(X))
            cov_val_grad = scipy.asarray(cov[i, j]).flatten()

            if compute_2:
                k = range(2 * len(X), 3 * len(X))
                cov_val_2 = scipy.asarray(cov[i, k]).flatten()
                cov_grad_2 = scipy.asarray(cov[j, k]).flatten()

            # Get geometry from EFIT:
            t_efit = self.efit_tree.getTimeBase()
            ok_idxs = self._get_efit_times_to_average(return_idxs=True)

            # Get correction factor for converting the abscissa back to Rmid:
            if self.abscissa == "Rmid":
                a = self.efit_tree.getAOut()[ok_idxs]
                var_a = scipy.var(a, ddof=1)
                if scipy.isnan(var_a):
                    var_a = 0
                mean_a = scipy.mean(a)

                mean_dX_droa = mean_a
                var_dX_droa = var_a

                if compute_2:
                    mean_dX_droa_2 = 0.0
                    var_dX_droa_2 = 0.0
                    cov_dX_droa_2 = 0.0
            elif self.abscissa == "r/a":
                mean_dX_droa = 1.0
                var_dX_droa = 0.0

                if compute_2:
                    mean_dX_droa_2 = 0.0
                    var_dX_droa_2 = 0.0
                    cov_dX_droa_2 = 0.0
            else:
                # Code taken from core.py of eqtools, modified to use
                # InterpolatedUnivariateSpline to get access to derivatives:
                dX_droa = scipy.zeros((len(X), len(ok_idxs)))
                if compute_2:
                    dX_droa_2 = scipy.zeros((len(X), len(ok_idxs)))
                # Loop over time indices:
                for idx, k in zip(ok_idxs, range(0, len(ok_idxs))):
                    resample_factor = 3
                    roa_grid = scipy.linspace(
                        0, 2, resample_factor * len(self.efit_tree.getRGrid())
                    )

                    X_on_grid = self.efit_tree.roa2rho(
                        self.abscissa, roa_grid, t_efit[idx]
                    )
                    # Rmid is handled specially up here, so we can filter the
                    # origin out properly:
                    X_on_grid[roa_grid == 0.0] = 0.0

                    good_idxs = ~scipy.isnan(X_on_grid)
                    X_on_grid = X_on_grid[good_idxs]
                    roa_grid = roa_grid[good_idxs]

                    spline = scipy.interpolate.InterpolatedUnivariateSpline(
                        roa_grid, X_on_grid, k=3
                    )
                    roa_X = self.efit_tree.rho2rho(self.abscissa, "r/a", X, t_efit[idx])
                    roa_X[X == 0.0] = 0.0
                    dX_droa[:, k] = spline(roa_X, nu=1)

                    if compute_2:
                        # dX_droa_2[:, k] = -spline(X, nu=2) * (dX_droa[:, k])**3.0
                        dX_droa_2[:, k] = spline(roa_X, nu=2)

                mean_dX_droa = scipy.mean(dX_droa, axis=1)
                var_dX_droa = scipy.var(
                    dX_droa, ddof=predict_kwargs.get("ddof", 1), axis=1
                )
                var_dX_droa[scipy.isnan(var_dX_droa)] = 0.0

                if compute_2:
                    mean_dX_droa_2 = scipy.mean(dX_droa_2, axis=1)
                    var_dX_droa_2 = scipy.var(
                        dX_droa_2, ddof=predict_kwargs.get("ddof", 1), axis=1
                    )
                    var_dX_droa_2[scipy.isnan(var_dX_droa_2)] = 0.0
                    cov_dX_droa_2 = scipy.cov(
                        dX_droa, dX_droa_2, ddof=predict_kwargs.get("ddof", 1)
                    )[i, j]

            if predict_kwargs.get("full_MC", False):
                # TODO: Doesn't include uncertainty in EFIT quantities!
                # Use samples:
                val_samps = out["samp"][special_vals : len(X) + special_vals]
                grad_samps = out["samp"][
                    len(X) + special_vals : 2 * len(X) + special_vals
                ]
                if scipy.ndim(mean_dX_droa) > 0:
                    mean_dX_droa = scipy.tile(mean_dX_droa, (val_samps.shape[1], 1)).T
                a_L_samps = -grad_samps * mean_dX_droa / val_samps
                mean_a_L = scipy.mean(a_L_samps, axis=1)
                std_a_L = scipy.std(
                    a_L_samps, axis=1, ddof=predict_kwargs.get("ddof", 1)
                )
                if compute_2:
                    g2_samps = out["samp"][2 * len(X) + special_vals :]
                    if scipy.ndim(mean_dX_droa_2) > 0:
                        mean_dX_droa_2 = scipy.tile(
                            mean_dX_droa_2, (val_samps.shape[1], 1)
                        ).T
                    a_L_grad_samps = (
                        g2_samps * mean_dX_droa / grad_samps
                        + mean_dX_droa_2 / mean_dX_droa
                    )
                    mean_a_L_grad = scipy.mean(a_L_grad_samps, axis=1)
                    std_a_L_grad = scipy.std(
                        a_L_grad_samps, axis=1, ddof=predict_kwargs.get("ddof", 1)
                    )
                    a2_2_samps = (
                        g2_samps * mean_dX_droa ** 2.0 + grad_samps * mean_dX_droa_2
                    ) / val_samps
                    mean_a2_2 = scipy.mean(a2_2_samps, axis=1)
                    std_a2_2 = scipy.std(
                        a2_2_samps, axis=1, ddof=predict_kwargs.get("ddof", 1)
                    )
            else:
                # Compute using error propagation:
                mean_a_L = -mean_grad * mean_dX_droa / mean_val
                std_a_L = scipy.sqrt(
                    var_val * (mean_grad * mean_dX_droa / mean_val ** 2.0) ** 2.0
                    + var_grad * (mean_dX_droa / mean_val) ** 2.0
                    + var_dX_droa * (mean_grad / mean_val) ** 2
                    - 2.0
                    * cov_val_grad
                    * (mean_grad * mean_dX_droa ** 2.0 / mean_val ** 3.0)
                )
                if compute_2:
                    mean_a_L_grad = (
                        mean_2 * mean_dX_droa / mean_grad
                        + mean_dX_droa_2 / mean_dX_droa
                    )
                    std_a_L_grad = scipy.sqrt(
                        var_grad * (mean_2 * mean_dX_droa / mean_grad ** 2.0) ** 2.0
                        + var_2 * (mean_dX_droa / mean_grad) ** 2.0
                        - 2.0
                        * cov_grad_2
                        * mean_2
                        * mean_dX_droa ** 2.0
                        / mean_grad ** 3.0
                        + var_dX_droa
                        * (mean_2 / mean_grad - mean_dX_droa_2 / mean_dX_droa ** 2.0)
                        ** 2.0
                        + var_dX_droa_2 / mean_dX_droa ** 2.0
                        + 2.0
                        * cov_dX_droa_2
                        * (mean_2 / mean_grad - mean_dX_droa_2 / mean_dX_droa ** 2.0)
                        / mean_dX_droa
                    )
                    mean_a2_2 = (
                        mean_2 * mean_dX_droa ** 2.0 + mean_grad * mean_dX_droa_2
                    ) / mean_val
                    std_a2_2 = scipy.sqrt(
                        var_val
                        * (mean_2 * mean_dX_droa ** 2.0 + mean_grad * mean_dX_droa_2)
                        ** 2
                        / mean_val ** 4
                        + var_2 * (mean_dX_droa ** 2.0 / mean_val) ** 2.0
                        - 2.0
                        * cov_val_2
                        * mean_dX_droa ** 2.0
                        / mean_val ** 3.0
                        * (mean_2 * mean_dX_droa ** 2.0 + mean_grad * mean_dX_droa_2)
                        + 4.0 * var_dX_droa * (mean_2 * mean_dX_droa / mean_val) ** 2.0
                        + var_dX_droa_2 * (mean_grad / mean_val) ** 2.0
                        + 4.0
                        * cov_dX_droa_2
                        * mean_grad
                        * mean_2
                        * mean_dX_droa
                        / mean_val ** 2.0
                    )

            # Plot result:
            if plot:
                ax = plot_kwargs.pop("ax", None)
                envelopes = plot_kwargs.pop("envelopes", [1, 3])
                base_alpha = plot_kwargs.pop("base_alpha", 0.375)
                if ax is None:
                    f = plt.figure()
                    ax = f.add_subplot(1, 1, 1)
                elif ax == "gca":
                    ax = plt.gca()

                l = ax.plot(X, mean_a_L, **plot_kwargs)
                color = plt.getp(l[0], "color")
                for i in envelopes:
                    ax.fill_between(
                        X,
                        mean_a_L - i * std_a_L,
                        mean_a_L + i * std_a_L,
                        facecolor=color,
                        alpha=base_alpha / i,
                    )
        elif self.X_dim == 2:
            raise NotImplementedError("Not there yet!")
        else:
            raise ValueError(
                "Cannot compute gradient scale length on data with "
                "X_dim=%d!" % (self.X_dim,)
            )

        if return_prediction:
            retval = {
                "mean_val": mean_val,
                "std_val": scipy.sqrt(var_val),
                "mean_grad": mean_grad,
                "std_grad": scipy.sqrt(var_grad),
                "mean_a_L": mean_a_L,
                "std_a_L": std_a_L,
                "cov": cov,
                # 'out': out,
                "special_mean": special_mean,
                "special_cov": special_cov,
            }
            if predict_kwargs.get("return_mean_func", False) and self.gp.mu is not None:
                retval["mean_func_val"] = mean_func_val
                retval["mean_func_grad"] = mean_func_grad
                retval["std_func_val"] = std_func_val
                retval["std_func_grad"] = std_func_grad

                retval["mean_without_func_val"] = mean_without_func_val
                retval["mean_without_func_grad"] = mean_without_func_grad
                retval["std_without_func_val"] = std_without_func_val
                retval["std_without_func_grad"] = std_without_func_grad

                retval["cov_func"] = out["cov_func"]
                retval["cov_without_func"] = out["cov_without_func"]
            if compute_2:
                retval["mean_2"] = mean_2
                retval["std_2"] = scipy.sqrt(var_2)
                retval["mean_a_L_grad"] = mean_a_L_grad
                retval["std_a_L_grad"] = std_a_L_grad
                retval["mean_a2_2"] = mean_a2_2
                retval["std_a2_2"] = std_a2_2
            if predict_kwargs.get("full_MC", False) or predict_kwargs.get(
                "return_samples", False
            ):
                retval["samp"] = out["samp"]
            return retval
        else:
            return (mean_a_L, std_a_L)

    def _get_efit_times_to_average(self, return_idxs=False):
        """Get the EFIT times to average over for a profile that has already been time-averaged.

        If this instance has a :py:attr:`times` attribute, the nearest indices
        to those times are used. Failing that, if :py:attr:`t_min` and
        :py:attr:`t_max` are distinct, all points between them are used. Failing
        that, the nearest index to t_min is used.
        """
        t_efit = self.efit_tree.getTimeBase()
        if hasattr(self, "times"):
            ok_idxs = self.efit_tree._getNearestIdx(self.times, t_efit)
        elif self.t_min != self.t_max:
            ok_idxs = scipy.where((t_efit >= self.t_min) & (t_efit <= self.t_max))[0]
            # Handle case where there are none:
            if len(ok_idxs) == 0:
                ok_idxs = self.efit_tree._getNearestIdx([self.t_min], t_efit)
        else:
            ok_idxs = self.efit_tree._getNearestIdx([self.t_min], t_efit)

        if return_idxs:
            return ok_idxs
        else:
            return t_efit[ok_idxs]

    def _make_volume_averaging_matrix(self, rho_grid=None, npts=400):
        """Generate a matrix of weights to find the volume average using the trapezoid rule.

        At present, this only supports data that have already been time-averaged.

        Parameters
        ----------
        rho_grid : array-like, optional
            The points (of the instance's abscissa) to use as the quadrature
            points. If these aren't provided, a grid of points uniformly spaced
            in volnorm will be produced. Default is None (produce uniform
            volnorm grid).
        npts : int, optional
            The number of points to use when producing a uniform volnorm grid.
            Default is 400.

        Returns
        -------
        rho_grid : array, (`npts`,)
            The quadrature points to be used on the instance's abscissa.
        weights : array, (1, `npts`)
            The matrix of weights to multiply the values at each location in
            `rho_grid` by to get the volume average.
        """
        if self.X_dim == 1:
            times = self._get_efit_times_to_average()

            if rho_grid is None:
                vol_grid = scipy.linspace(0, 1, npts)

                if "volnorm" not in self.abscissa:
                    rho_grid = self.efit_tree.rho2rho(
                        "volnorm", self.abscissa, vol_grid, times, each_t=True
                    )
                    rho_grid = scipy.mean(rho_grid, axis=0)
                    # Correct for the NaN that shows up sometimes:
                    rho_grid[0] = 0.0
                else:
                    if self.abscissa.startswith("sqrt"):
                        rho_grid = scipy.sqrt(vol_grid)
                    else:
                        rho_grid = vol_grid

                N = npts - 1
                a = 0
                b = 1

                # Use the trapezoid rule:
                weights = 2 * scipy.ones_like(vol_grid)
                weights[0] = 1
                weights[-1] = 1
                weights *= (b - a) / (2.0 * N)
            else:
                if "volnorm" not in self.abscissa:
                    vol_grid = self.efit_tree.rho2rho(
                        self.abscissa, "volnorm", rho_grid, times, each_t=True
                    )
                    # Use nanmean in case there is a value which is teetering
                    # -- we want to keep it in.
                    vol_grid = scipy.stats.nanmean(vol_grid, axis=0)
                else:
                    if self.abscissa.startswith("sqrt"):
                        vol_grid = scipy.asarray(rho_grid, dtype=float) ** 2
                    else:
                        vol_grid = scipy.asarray(rho_grid, dtype=float)

                ok_mask = (~scipy.isnan(vol_grid)) & (vol_grid <= 1.0)
                delta_vol = scipy.diff(vol_grid[ok_mask])
                weights = scipy.zeros_like(vol_grid)
                weights[ok_mask] = 0.5 * (
                    scipy.insert(delta_vol, 0, 0) + scipy.insert(delta_vol, -1, 0)
                )

            weights = scipy.atleast_2d(weights)
            return (rho_grid, weights)
        else:
            raise NotImplementedError(
                "Volume averaging not yet supported for X_dim > 1!"
            )

    def compute_volume_average(
        self,
        return_std=True,
        grid=None,
        npts=400,
        force_update=False,
        gp_kwargs={},
        MAP_kwargs={},
        **predict_kwargs
    ):
        """Compute the volume average of the profile.

        Right now only supports data that have already been time-averaged.

        Parameters
        ----------
        return_std : bool, optional
            If True, the standard deviation of the volume average is computed
            and returned. Default is True (return mean and stddev of volume average).
        grid : array-like, optional
            The quadrature points to use when finding the volume average. If
            these are not provided, a uniform grid over volnorm will be used.
            Default is None (use uniform volnorm grid).
        npts : int, optional
            The number of uniformly-spaced volnorm points to use if `grid` is
            not specified. Default is 400.
        force_update : bool, optional
            If True, a new Gaussian process will be created even if one already
            exists. Set this if you have added data or constraints since you
            created the Gaussian process. Default is False (use current Gaussian
            process if it exists).
        gp_kwargs : dict, optional
            The entries of this dictionary are passed as kwargs to
            :py:meth:`create_gp` if it gets called. Default is {}.
        MAP_kwargs : dict, optional
            The entries of this dictionary are passed as kwargs to
            :py:meth:`find_gp_MAP_estimate` if it gets called. Default is {}.
        **predict_kwargs : optional parameters
            All other parameters are passed to the Gaussian process'
            :py:meth:`predict` method.

        Returns
        -------
        mean : float
            The mean of the volume average.
        std : float
            The standard deviation of the volume average. Only returned if
            `return_std` is True. Note that this is only sufficient as an error
            estimate if you separately verify that the integration error is less
            than this!
        """
        if self.X_dim == 1:
            if force_update or self.gp is None:
                self.create_gp(**gp_kwargs)
                if not predict_kwargs.get("use_MCMC", False):
                    self.find_gp_MAP_estimate(**MAP_kwargs)
            rho_grid, weights = self._make_volume_averaging_matrix(
                rho_grid=grid, npts=npts
            )

            res = self.gp.predict(
                rho_grid,
                output_transform=weights,
                return_std=return_std,
                return_cov=False,
                full_output=False,
                **predict_kwargs
            )
            if return_std:
                return (res[0][0], res[1][0])
            else:
                return res[0]
        else:
            raise NotImplementedError(
                "Volume averaging not yet supported for " "X_dim > 1!"
            )

    def compute_peaking(
        self,
        return_std=True,
        grid=None,
        npts=400,
        force_update=False,
        gp_kwargs={},
        MAP_kwargs={},
        **predict_kwargs
    ):
        r"""Compute the peaking of the profile.

        Right now only supports data that have already been time-averaged.

        Uses the definition from Greenwald, et al. (2007):
        :math:`w(\psi_n=0.2)/\langle w \rangle`.

        Parameters
        ----------
        return_std : bool, optional
            If True, the standard deviation of the volume average is computed
            and returned. Default is True (return mean and stddev of peaking).
        grid : array-like, optional
            The quadrature points to use when finding the volume average. If
            these are not provided, a uniform grid over volnorm will be used.
            Default is None (use uniform volnorm grid).
        npts : int, optional
            The number of uniformly-spaced volnorm points to use if `grid` is
            not specified. Default is 400.
        force_update : bool, optional
            If True, a new Gaussian process will be created even if one already
            exists. Set this if you have added data or constraints since you
            created the Gaussian process. Default is False (use current Gaussian
            process if it exists).
        gp_kwargs : dict, optional
            The entries of this dictionary are passed as kwargs to
            :py:meth:`create_gp` if it gets called. Default is {}.
        MAP_kwargs : dict, optional
            The entries of this dictionary are passed as kwargs to
            :py:meth:`find_gp_MAP_estimate` if it gets called. Default is {}.
        **predict_kwargs : optional parameters
            All other parameters are passed to the Gaussian process'
            :py:meth:`predict` method.
        """
        if self.X_dim == 1:
            if force_update or self.gp is None:
                self.create_gp(**gp_kwargs)
                if not predict_kwargs.get("use_MCMC", False):
                    self.find_gp_MAP_estimate(**MAP_kwargs)
            rho_grid, weights = self._make_volume_averaging_matrix(
                rho_grid=grid, npts=npts
            )
            weights = scipy.append(weights, 0)

            # Find the relevant core location:
            if "psinorm" in self.abscissa:
                if self.abscissa.startswith("sqrt"):
                    core_loc = scipy.sqrt(0.2)
                else:
                    core_loc = 0.2
            else:
                times = self._get_efit_times_to_average()

                core_loc = self.efit_tree.psinorm2rho(
                    self.abscissa, 0.2, times, each_t=True
                )
                core_loc = scipy.mean(core_loc)

            rho_grid = scipy.append(rho_grid, core_loc)
            core_select = scipy.zeros_like(weights)
            core_select[-1] = 1
            weights = scipy.vstack((weights, core_select))
            res = self.gp.predict(
                rho_grid,
                output_transform=weights,
                return_std=return_std,
                return_cov=return_std,
                full_output=False,
                **predict_kwargs
            )
            if return_std:
                mean_res = res[0]
                cov_res = res[1]
                mean = mean_res[1] / mean_res[0]
                std = scipy.sqrt(
                    cov_res[1, 1] / mean_res[0] ** 2
                    + cov_res[0, 0] * mean_res[1] ** 2 / mean_res[0] ** 4
                    - 2.0 * cov_res[0, 1] * mean_res[1] / mean_res[0] ** 3
                )
                return (mean, std)
            else:
                return res[1] / res[0]
        else:
            raise NotImplementedError(
                "Computation of peaking factors not yet " "supported for X_dim > 1!"
            )


def neTS(
    shot,
    abscissa="RZ",
    t_min=None,
    t_max=None,
    efit_tree=None,
    remove_edge=False,
    remove_zeros=True,
    remove_divertor=False,
    rho_threshold=1,
    interelm=False,
    **kwargs
):
    """Returns a profile representing electron density
       from the core Thomson scattering system.

    Parameters
    ----------
    shot : int
        The shot number to load.
    abscissa : str, optional
        The abscissa to use for the data. The default is 'RZ'.
    t_min : float, optional
        The smallest time to include. Default is None (no lower bound).
    t_max : float, optional
        The largest time to include. Default is None (no upper bound).
    efit_tree : eqtools.TCVLIUQETree, optional
        An eqtools.TCVLIUQETree object open to the correct shot.
        The shot of the given tree is not checked! Default is None
        (open tree).
    remove_edge : bool, optional
        If True, will remove points that are outside the LCFS. It will convert
        the abscissa to psinorm if necessary. Default is False (keep edge).
    remove_zeros: bool, optional
        If True, will remove points that are identically zero. Default is True
        (remove zero points). This was added because clearly bad points aren't
        always flagged with a sentinel value of errorbar.
    remove_divertor: bool, optional
        If True will remove points in the divertor chamber. Useful to then remap upstream
    interelm : bool, optional is False
        if True load the appropriate ELM signal from photodiods
    **kwargs : all the arguments for the ELM detection methods
    """
    p = BivariatePlasmaProfile(
        X_dim=3,
        X_units=["s", "m", "m"],
        y_units="$10^{20}$ m$^{-3}$",
        X_labels=["$t$", "$R$", "$Z$"],
        y_label=r"$n_e$, TS",
    )

    electrons = MDSplus.Tree("tcv_shot", shot)
    Z_CTS = electrons.getNode(r"\diagz::thomson_set_up:vertical_pos").data()
    R_CTS = electrons.getNode(r"\diagz::thomson_set_up:radial_pos").data()
    mask = np.ones(Z_CTS.size, dtype="bool")
    if remove_divertor:
        mask[np.where(Z_CTS <= -0.4)[0]] = False
    # now apply the mas to R and Z
    R_CTS, Z_CTS = R_CTS[mask], Z_CTS[mask]
    channels = range(0, len(Z_CTS))
    N_ne_TS = electrons.getNode(r"\results::thomson:ne")

    t_ne_TS = electrons.getNode(r"\results::thomson:times").data()
    ne_TS = N_ne_TS.data()[mask, :] / 1e20
    dev_ne_TS = (
        electrons.getNode(r"\results::thomson:ne:error_bar").data()[mask, :] / 1e20
    )
    dev_ne_TS[dev_ne_TS < 0] = 0
    # mask alredy in time because it run better for ELM
    _masktime = np.where((t_ne_TS >= t_min) & (t_ne_TS <= t_max))[0]
    t_ne_TS = t_ne_TS[_masktime]
    ne_TS = ne_TS[:, _masktime]
    dev_ne_TS = dev_ne_TS[:, _masktime]

    if interelm:
        _Node = electrons.getNode(r"\base::pd:pd_001")
        _dalpha_time = _Node.getDimensionAt().data()
        # restrict to the time bases
        _idx = np.where((_dalpha_time >= t_min) & (_dalpha_time <= t_max))[0]
        _dalpha = _Node.data()[_idx]
        _dalpha_time = _dalpha_time[_idx]
        ED = elm_detection.elm_detection(
            _dalpha_time,
            _dalpha,
            rho=kwargs.get("rho", 0.85),
            width=kwargs.get("width", 0.2),
            t_sep=kwargs.get("t_sep", 0.002),
            mode=kwargs.get("mode", "fractional"),
            mtime=kwargs.get("mtime", 500),
            hwidth=kwargs.get("hwidth", 8),
            dnpoint=kwargs.get("dnpoint", 5),
        )
        ED.run()
        maskElm = ED.filter_signal(
            t_ne_TS, inter_elm_range=kwargs.get("percentage", [0.7, 0.9])
        )
    else:
        maskElm = np.ones(t_ne_TS.size, dtype="bool")
    # now limit which will make differences only for the interELM call
    t_ne_TS = t_ne_TS[maskElm]
    ne_TS = ne_TS[:, maskElm]
    dev_ne_TS = dev_ne_TS[:, maskElm]

    t_grid, Z_grid = scipy.meshgrid(t_ne_TS, Z_CTS)
    t_grid, R_grid = scipy.meshgrid(t_ne_TS, R_CTS)
    t_grid, channel_grid = scipy.meshgrid(t_ne_TS, channels)

    ne = ne_TS.flatten()
    err_ne = dev_ne_TS.flatten()
    Z = scipy.atleast_2d(Z_grid.flatten())
    R = scipy.atleast_2d(R_grid.flatten())
    channels = channel_grid.flatten()
    t = scipy.atleast_2d(t_grid.flatten())

    X = scipy.hstack((t.T, R.T, Z.T))

    p.shot = shot
    p.efit_tree = eqtools.TCVLIUQEMATTree(shot)
    p.abscissa = "RZ"

    p.add_data(X, ne, err_y=err_ne, channels={1: channels, 2: channels})

    # Remove flagged points:
    p.remove_points(
        scipy.isnan(p.err_y)
        | scipy.isinf(p.err_y)
        | (p.err_y == 0.0)
        | (p.err_y == 1.0)
        | (p.err_y == 2.0)
        | (p.err_y == -1.0)
        | ((p.y == -1) & remove_zeros)
        | scipy.isnan(p.y)
        | scipy.isinf(p.y)
    )
    # no longer needed because we limit before
    # if t_min is not None:
    #     p.remove_points(scipy.asarray(p.X[:, 0]).flatten() < t_min)
    # if t_max is not None:
    #     p.remove_points(scipy.asarray(p.X[:, 0]).flatten() > t_max)
    p.convert_abscissa(abscissa)

    if remove_edge:
        p.remove_edge_points()

    return p


def neRCP(
    shot,
    abscissa="Rmid",
    t_min=None,
    t_max=None,
    electrons=None,
    efit_tree=None,
    remove_edge=False,
    rf=None,
    remove_divertor=False,
    rho_threshold=1,
    interelm=False,
    **kwargs
):
    """Returns a profile representing electron density
    from the Reciprocating Langmuir probes.
    It computes correctly the
    strokes according to the t_min and t_max given

    Parameters
    ----------
    shot : int
        The shot number to load.
    abscissa : str, optional
        The abscissa to use for the data. The default is 'Rmid'.
    t_min : float, optional
        The smallest time to include. Default is None (no lower bound).
    t_max : float, optional
        The largest time to include. Default is None (no upper bound).
    electrons : MDSplus.Tree, optional
        An MDSplus.Tree object open to the electrons tree of the correct shot.
        The shot of the given tree is not checked! Default is None (open tree).
    efit_tree : eqtools.CModEFITTree, optional
        An eqtools.CModEFITTree object open to the correct shot. The shot of the
        given tree is not checked! Default is None (open tree).
    remove_edge : bool, optional
        If True, will remove points that are outside the LCFS. It will convert
        the abscissa to psinorm if necessary. Default is False (keep edge).
    rf : MDSplus.Tree, optional
        An MDSplus.Tree object open to the RF tree of the correct shot.
        The shot of the given tree is not checked! Default is None (open tree).
    remove_divertor: bool, optional
        does not do nothing needed in order to ensure we can can the method which combines Te and RCP
    rho_threshold: by Default the RCP method does not consider the point inside the LCFS. By specifing a
        different value you can cut the profiles at different rho. useful if the probe profile is messed
        up close to the LCFS. Need to be specified in rho_poloidal or sqrtpsinorm.
    """
    p = BivariatePlasmaProfile(
        X_dim=2,
        X_units=["s", "m"],
        y_units="$10^{20}$ m$^{-3}$",
        X_labels=["$t$", r"$R_{mid}$"],
        y_label=r"$n_e$, RCP",
        weightable=False,
    )
    if rf is None:
        rf = MDSplus.Tree("tcv_shot", shot)
    if efit_tree is None:
        p.efit_tree = eqtools.TCVLIUQEMATTree(shot)
    else:
        p.efit_tree = efit_tree

    t = []
    R = []
    ne = []
    for Plunge in ("1", "2"):
        try:
            t = scipy.append(t, rf.getNode(r"\results::fp:t_" + Plunge).data())
            R = scipy.append(R, rf.getNode(r"\results::fp:r_" + Plunge).data())
            ne = scipy.append(
                ne, rf.getNode(r"\results::fp:ne_" + Plunge).data() / 1e20
            )
        except:
            warnings.warn("Plunge " + Plunge + " for shot %5i" % shot + " not found")
            pass

    # for each time and R convert into Rmid but we need to restrict
    # for points inside the limiter
    rlim, zlim = p.efit_tree.getMachineCrossSection()
    _mask = scipy.where(scipy.asarray(R) <= rlim.max())[0]
    t, R, ne = (
        scipy.asarray(t)[_mask],
        scipy.asarray(R)[_mask],
        scipy.asarray(ne)[_mask],
    )
    # now we mask in time because it works better for interELM
    _maskTime = np.where((t >= t_min) & (t <= t_max))[0]
    t, R, ne = t[_maskTime], R[_maskTime], ne[_maskTime]
    if interelm:
        _Node = rf.getNode(r"\base::pd:pd_001")
        _dalpha_time = _Node.getDimensionAt().data()
        # restrict to the time bases
        _idx = np.where((_dalpha_time >= t_min) & (_dalpha_time <= t_max))[0]
        _dalpha = _Node.data()[_idx]
        _dalpha_time = _dalpha_time[_idx]
        ED = elm_detection.elm_detection(
            _dalpha_time,
            _dalpha,
            rho=kwargs.get("rho", 0.85),
            width=kwargs.get("width", 0.2),
            t_sep=kwargs.get("t_sep", 0.002),
            mode=kwargs.get("mode", "fractional"),
            mtime=kwargs.get("mtime", 500),
            hwidth=kwargs.get("hwidth", 8),
            dnpoint=kwargs.get("dnpoint", 5),
        )
        ED.run()
        maskElm = ED.filter_signal(
            t, inter_elm_range=kwargs.get("percentage", [0.7, 0.9])
        )
    else:
        maskElm = np.ones(t.size, dtype="bool")
    t, R, ne = t[maskElm], R[maskElm], ne[maskElm]
    Rmid = []
    for _t, _r in zip(t, R):
        Rmid.append(p.efit_tree.rz2rmid(_r, 0, _t))
    Rmid = scipy.asarray(Rmid)
    # channels = range(0, ne.shape[1])
    # channel_grid, t_grid = scipy.meshgrid(channels, t)

    ne = ne.ravel()
    R = R.ravel()
    # channels = channel_grid.ravel()
    #    t = t_grid.ravel()

    X = scipy.vstack((t, R)).T

    p.shot = shot
    p.abscissa = "Rmid"

    p.add_data(X, ne, err_y=0.1 * scipy.absolute(ne))

    # Remove flagged points:
    # no more needed because we
    # limit before
    p.remove_points(p.y == 0)
    # if t_min is not None:
    #     p.remove_points(scipy.asarray(p.X[:, 0]).flatten() < t_min)
    # if t_max is not None:
    #     p.remove_points(scipy.asarray(p.X[:, 0]).flatten() > t_max)

    p.convert_abscissa("sqrtpsinorm")
    _ = p.remove_points(p.X[:, 1] < rho_threshold)
    if abscissa != "sqrtpsinorm":
        p.convert_abscissa(abscissa)

    if remove_edge:
        p.remove_edge_points()

    return p


def ne(shot, include=["TS", "RCP"], **kwargs):
    """Returns a profile representing electron density from both the
    Thomson scattering and RCP

    Parameters
    ----------
    shot : int
        The shot number to load.
    include : list of str, optional
        The data sources to include. Valid options are:

            ======= ========================
            TS      Thomson scattering
            RCP     Reciprocating Langmuir Probe
            ======= ========================

    **kwargs
        All remaining parameters are passed to the individual loading methods.
    """
    # if 'electrons' not in kwargs:
    #     kwargs['electrons'] = MDSplus.Tree('electrons', shot)
    # if 'efit_tree' not in kwargs:
    #     kwargs['efit_tree'] = eqtools.CModEFITTree(shot)
    p_list = []
    for system in include:
        if system == "TS":
            try:
                p_list.append(neTS(shot, **kwargs))
            except:
                warnings.warn("Thomson profiles not found")
        elif system == "RCP":
            try:
                p_list.append(neRCP(shot, **kwargs))
            except:
                warnings.warn("RCP profile not found")
        else:
            raise ValueError("Unknown profile '%s'." % (system,))

    p = p_list.pop()
    for p_other in p_list:
        p.add_profile(p_other)

    return p


def TeTS(
    shot,
    abscissa="RZ",
    t_min=None,
    t_max=None,
    efit_tree=None,
    remove_edge=False,
    remove_zeros=True,
    remove_divertor=False,
    rho_threshold=1,
    interelm=False,
    **kwargs
):
    """Returns a profile representing electron temperature from the core Thomson scattering system.

    Parameters
    ----------
    shot : int
        The shot number to load.
    abscissa : str, optional
        The abscissa to use for the data. The default is 'RZ'.
    t_min : float, optional
        The smallest time to include. Default is None (no lower bound).
    t_max : float, optional
        The largest time to include. Default is None (no upper bound).
    efit_tree : eqtools.CModEFITTree, optional
        An eqtools.CModEFITTree object open to the correct shot. The shot of the
        given tree is not checked! Default is None (open tree).
    remove_edge : bool, optional
        If True, will remove points that are outside the LCFS. It will convert
        the abscissa to psinorm if necessary. Default is False (keep edge).
    remove_zeros: bool, optional
        If True, will remove points that are identically zero. Default is True
        (remove zero points). This was added because clearly bad points aren't
        always flagged with a sentinel value of errorbar.
    remove_divertor: bool, optional
        If True will remove points in the divertor chamber. Useful to then remap upstream
    """
    p = BivariatePlasmaProfile(
        X_dim=3,
        X_units=["s", "m", "m"],
        y_units="eV",
        X_labels=["$t$", "$R$", "$Z$"],
        y_label=r"$T_e$, TS",
    )

    electrons = MDSplus.Tree("tcv_shot", shot)
    Z_CTS = electrons.getNode(r"\diagz::thomson_set_up:vertical_pos").data()
    R_CTS = electrons.getNode(r"\diagz::thomson_set_up:radial_pos").data()
    mask = np.ones(Z_CTS.size, dtype="bool")
    if remove_divertor:
        mask[np.where(Z_CTS <= -0.4)[0]] = False
    # now apply the mas to R and Z
    R_CTS, Z_CTS = R_CTS[mask], Z_CTS[mask]
    channels = range(0, len(Z_CTS))

    N_Te_TS = electrons.getNode(r"\results::thomson:te")

    t_Te_TS = electrons.getNode(r"\results::thomson:times").data()
    Te_TS = N_Te_TS.data()[mask, :]
    dev_Te_TS = electrons.getNode(r"\results::thomson:te:error_bar").data()[mask, :]
    dev_Te_TS[dev_Te_TS < 0] = 0
    # mask alredy in time because it run better for ELM
    _masktime = np.where((t_Te_TS >= t_min) & (t_Te_TS <= t_max))[0]
    t_Te_TS = t_Te_TS[_masktime]
    Te_TS = Te_TS[:, _masktime]
    dev_Te_TS = dev_Te_TS[:, _masktime]
    if interelm:
        _Node = electrons.getNode(r"\base::pd:pd_001")
        _dalpha_time = _Node.getDimensionAt().data()
        # restrict to the time bases
        _idx = np.where((_dalpha_time >= t_min) & (_dalpha_time <= t_max))[0]
        _dalpha = _Node.data()[_idx]
        _dalpha_time = _dalpha_time[_idx]
        ED = elm_detection.elm_detection(
            _dalpha_time,
            _dalpha,
            rho=kwargs.get("rho", 0.85),
            width=kwargs.get("width", 0.2),
            t_sep=kwargs.get("t_sep", 0.002),
            mode=kwargs.get("mode", "fractional"),
            mtime=kwargs.get("mtime", 500),
            hwidth=kwargs.get("hwidth", 8),
            dnpoint=kwargs.get("dnpoint", 5),
        )
        ED.run()
        maskElm = ED.filter_signal(
            t_Te_TS, inter_elm_range=kwargs.get("percentage", [0.7, 0.9])
        )
    else:
        maskElm = np.ones(t_Te_TS.size, dtype="bool")
    t_Te_TS = t_Te_TS[maskElm]
    Te_TS = ne_TS[:, maskElm]
    dev_Te_TS = dev_Te_TS[:, maskElm]

    t_grid, Z_grid = scipy.meshgrid(t_Te_TS, Z_CTS)
    t_grid, R_grid = scipy.meshgrid(t_Te_TS, R_CTS)
    t_grid, channel_grid = scipy.meshgrid(t_Te_TS, channels)

    Te = Te_TS.flatten()
    err_Te = dev_Te_TS.flatten()
    Z = scipy.atleast_2d(Z_grid.flatten())
    R = scipy.atleast_2d(R_grid.flatten())
    channels = channel_grid.flatten()
    t = scipy.atleast_2d(t_grid.flatten())

    X = scipy.hstack((t.T, R.T, Z.T))

    p.shot = shot
    p.efit_tree = eqtools.TCVLIUQEMATTree(shot)
    p.abscissa = "RZ"

    p.add_data(X, Te, err_y=err_Te, channels={1: channels, 2: channels})
    # Remove flagged points:
    p.remove_points(
        scipy.isnan(p.err_y)
        | scipy.isinf(p.err_y)
        | (p.err_y == 0.0)
        | (p.err_y == 1.0)
        | ((p.y == -1) & remove_zeros)
        | scipy.isnan(p.y)
        | scipy.isinf(p.y)
    )
    # if t_min is not None:
    #     p.remove_points(scipy.asarray(p.X[:, 0]).flatten() < t_min)
    # if t_max is not None:
    #     p.remove_points(scipy.asarray(p.X[:, 0]).flatten() > t_max)
    p.convert_abscissa(abscissa)

    if remove_edge:
        p.remove_edge_points()

    return p


def TeRCP(
    shot,
    abscissa="Rmid",
    t_min=None,
    t_max=None,
    remove_edge=False,
    rf=None,
    remove_divertor=False,
    rho_threshold=1,
    interelm=True,
    **kwargs
):
    """Returns a profile representing electron Temperature
    from the Reciprocating Langmuir probes.
    It computes correctly the
    strokes according to the t_min and t_max given

    Parameters
    ----------
    shot : int
        The shot number to load.
    abscissa : str, optional
        The abscissa to use for the data. The default is 'Rmid'.
    t_min : float, optional
        The smallest time to include. Default is None (no lower bound).
    t_max : float, optional
        The largest time to include. Default is None (no upper bound).
    remove_edge : bool, optional
        If True, will remove points that are outside the LCFS. It will convert
        the abscissa to psinorm if necessary. Default is False (keep edge).
    rho_threshold: by Default the RCP method does not consider the point inside the LCFS. By specifing a
        different value you can cut the profiles at different rho. useful if the probe profile is messed
        up close to the LCFS. Need to be specified in rho_poloidal or sqrtpsinorm.
    interelm : bool, optional. Default is False
        if True compute use only interelm 
    """
    p = BivariatePlasmaProfile(
        X_dim=2,
        X_units=["s", "m"],
        y_units="eV",
        X_labels=["$t$", r"$R_{mid}$"],
        y_label=r"$T_e$, RCP",
        weightable=False,
    )
    rf = MDSplus.Tree("tcv_shot", shot)
    p.efit_tree = eqtools.TCVLIUQEMATTree(shot)

    t = []
    R = []
    ne = []
    for Plunge in ("1", "2"):
        try:
            t = scipy.append(t, rf.getNode(r"\results::fp:t_" + Plunge).data())
            R = scipy.append(R, rf.getNode(r"\results::fp:r_" + Plunge).data())
            ne = scipy.append(ne, rf.getNode(r"\results::fp:te_" + Plunge).data())
        except:
            warnings.warn("Plunge " + Plunge + " for shot %5i" % shot + " not found")
            pass

    # for each time and R convert into Rmid but we need to restrict
    # for points inside the limiter
    rlim, zlim = p.efit_tree.getMachineCrossSection()
    _mask = scipy.where(scipy.asarray(R) <= rlim.max())[0]
    t, R, ne = (
        scipy.asarray(t)[_mask],
        scipy.asarray(R)[_mask],
        scipy.asarray(ne)[_mask],
    )
    # now we mask in time because it works better for interELM
    _maskTime = np.where((t >= t_min) & (t <= t_max))[0]
    t, R, ne = t[_maskTime], R[_maskTime], ne[_maskTime]
    if interelm:
        _Node = electrons.getNode(r"\base::pd:pd_001")
        _dalpha_time = _Node.getDimensionAt().data()
        # restrict to the time bases
        _idx = np.where((_dalpha_time >= t_min) & (_dalpha_time <= t_max))[0]
        _dalpha = _Node.data()[_idx]
        _dalpha_time = _dalpha_time[_idx]
        ED = elm_detection.elm_detection(
            _dalpha_time,
            _dalpha,
            rho=kwargs.get("rho", 0.85),
            width=kwargs.get("width", 0.2),
            t_sep=kwargs.get("t_sep", 0.002),
            mode=kwargs.get("mode", "fractional"),
            mtime=kwargs.get("mtime", 500),
            hwidth=kwargs.get("hwidth", 8),
            dnpoint=kwargs.get("dnpoint", 5),
        )
        ED.run()
        maskElm = ED.filter_signal(
            t, inter_elm_range=kwargs.get("percentage", [0.7, 0.9])
        )
    else:
        maskElm = np.ones(t.size, dtype="bool")
    t, R, ne = t[maskElm], R[maskElm], ne[maskElm]

    # for each time and R convert into Rmid
    Rmid = []
    for _t, _r in zip(t, R):
        Rmid.append(p.efit_tree.rz2rmid(_r, 0, _t))
    Rmid = scipy.asarray(Rmid)

    ne = ne.ravel()
    R = R.ravel()
    X = scipy.vstack((t, R)).T

    p.shot = shot
    p.abscissa = "Rmid"

    p.add_data(X, ne, err_y=0.1 * scipy.absolute(ne))

    # Remove flagged points:
    p.remove_points(p.y == 0)
    if t_min is not None:
        p.remove_points(scipy.asarray(p.X[:, 0]).flatten() < t_min)
    if t_max is not None:
        p.remove_points(scipy.asarray(p.X[:, 0]).flatten() > t_max)
    p.convert_abscissa("sqrtpsinorm")
    _ = p.remove_points(p.X[:, 1] < rho_threshold)
    if abscissa != "sqrtpsinorm":
        p.convert_abscissa(abscissa)

    if remove_edge:
        p.remove_edge_points()

    return p


def Te(shot, include=["TS", "RCP"], **kwargs):
    """Returns a profile representing electron temperature
    from the Thomson scattering and Reciprocating Langmuir Probe.

    Parameters
    ----------
    shot : int
        The shot number to load.
    include : list of str, optional
        The data sources to include. Valid options are:
            ======= ========================
            TS      Thomson scattering
            RCP     Reciprocating Langmuir Probe
            ======= ========================
    **kwargs
        All remaining parameters are passed to the individual loading methods.
    """

    p_list = []
    for system in include:
        if system == "TS":
            try:
                p_list.append(TeTS(shot, **kwargs))
            except:
                warnings.warn("Thomson profile not found")
        elif system == "RCP":
            try:
                p_list.append(TeRCP(shot, **kwargs))
            except:
                warnings.warn("RCP profile not found")
        else:
            raise ValueError("Unknown profile '%s'." % (system,))

    p = p_list.pop()
    for p_other in p_list:
        p.add_profile(p_other)

    return p


def read_plasma_csv(*args, **kwargs):
    """Returns a profile containing the data from a CSV file.

    If your data are bivariate, you must ensure that time ends up being the
    first column, either by putting it first in your CSV file or by specifying
    its name first in `X_names`.

    The CSV file can contain metadata lines of the form "name data" or
    "name data,data,...". The following metadata are automatically parsed into
    the correct fields:

        ========== ======================================================
        shot       shot number
        times      comma-separated list of times included in the data
        t_min      minimum time included in the data
        t_max      maximum time included in the data
        coordinate the abscissa the data are represented as a function of
        ========== ======================================================

    If you don't provide `coordinate` in the metadata, the program will try to
    use the last entry in X_labels to infer the abscissa. If this fails, it will
    simply set the abscissa to the title of the last entry in X_labels. If you
    provide your data as a function of R, Z it will look for the last two
    entries in X_labels to be R and Z once surrounding dollar signs and spaces
    are removed.

    Parameters are the same as :py:func:`read_csv`.
    """
    # TODO: Does not support transformed quantities!
    p = read_csv(*args, **kwargs)
    p.__class__ = BivariatePlasmaProfile
    metadata = dict([l.split(None, 1) for l in p.metadata])
    if "shot" in metadata:
        p.shot = int(metadata["shot"])
        p.efit_tree = eqtools.TCVLIUQEMATTree(p.shot)
    if "times" in metadata:
        p.times = [float(t) for t in metadata["times"].split(",")]
    if "t_max" in metadata:
        p.t_max = float(metadata["t_max"])
    if "t_min" in metadata:
        p.t_min = float(metadata["t_min"])
    if "coordinate" in metadata:
        p.abscissa = metadata["coordinate"]
    else:
        if (
            p.X_dim > 1
            and p.X_labels[-2].strip("$ ") == "R"
            and p.X_labels[-1].strip("$ ") == "Z"
        ):
            p.abscissa = "RZ"
        else:
            try:
                p.abscissa = _abscissa_mapping[p.X_labels[-1]]
            except KeyError:
                p.abscissa = p.X_labels[-1].strip("$ ")

    return p


def read_plasma_NetCDF(*args, **kwargs):
    """Returns a profile containing the data from a NetCDF file.

    The file can contain metadata attributes specified in the `metadata` kwarg.
    The following metadata are automatically parsed into the correct fields:

        ========== ======================================================
        shot       shot number
        times      comma-separated list of times included in the data
        t_min      minimum time included in the data
        t_max      maximum time included in the data
        coordinate the abscissa the data are represented as a function of
        ========== ======================================================

    If you don't provide `coordinate` in the metadata, the program will try to
    use the last entry in X_labels to infer the abscissa. If this fails, it will
    simply set the abscissa to the title of the last entry in X_labels. If you
    provide your data as a function of R, Z it will look for the last two
    entries in X_labels to be R and Z once surrounding dollar signs and spaces
    are removed.

    Parameters are the same as :py:func:`read_NetCDF`.
    """
    # TODO: Does not support transformed quantities!
    metadata = kwargs.pop("metadata", [])
    metadata = set(metadata + ["shot", "times", "t_max", "t_min", "coordinate"])
    p = read_NetCDF(*args, metadata=metadata, **kwargs)
    p.__class__ = BivariatePlasmaProfile
    if hasattr(p, "shot"):
        p.efit_tree = eqtools.TCVLIUQEMATTree(p.shot)
    if hasattr(p, "coordinate"):
        p.abscissa = p.coordinate
    else:
        if (
            p.X_dim > 1
            and p.X_labels[-2].strip("$ ") == "R"
            and p.X_labels[-1].strip("$ ") == "Z"
        ):
            p.abscissa = "RZ"
        else:
            try:
                p.abscissa = _abscissa_mapping[p.X_labels[-1]]
            except KeyError:
                p.abscissa = p.X_labels[-1].strip("$ ")

    return p
