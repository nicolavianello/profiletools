import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline as splint


class elm_detection(object):
    """
    Algorithm for detecting the onset of ELMs in edge light data.

    Required data:
        y  ->  edge light data array
        t  ->  time array
        data should be trimmed down to only the period
        containing the ELMs.

    Create class:
        elms = elm_detection( t, y )

    Run the algorithm:
        elms.run()

    Access results:
        elms.elm_times # times in shot where ELMs occur
        elms.elm_sep   # separation times between adjacent ELMs


    Code by Chris Bowman
    version: 0.3 (in development)
    updated: 14/10/2016
    Update by N. Vianello to include the variables in the init call
    Update by N. Walkden for an additional method to create mask on appropriate
        time basis
    """

    def __init__(
        self,
        t,
        y,
        rho=0.92,
        width=0.2,
        t_sep=0.002,
        dnpoint=5,
        mode="fractional",
        mtime=500,
        hwidth=8,
    ):
        """

        :param t: time basis of the signal
        :param y: signal to be analyzed
        :param rho: percentile used in threshold calculations. Default is 0.92
        :param width: half-width of the time window used for calculating
        threshold values. Default is 0.2
        :param t_sep: estimate of maximum duration (in seconds) of the positive-gradient
        :param dnpoint: number of data points which make up the initial sharp edge of an ELM feature.
        :param mode: Set if in the computation of the threshold
        in the derivative we use fractional values or absolute. Default is 'fractiona'
        :param mtime: number of time windows used to calculate threshold curve
        :param hwidth: half-width of gaussian smoothing window
        """
        self.t = t  # Be2 time data
        self.y = y  # Be2 intensity data

        # percentile used in threshold calculations.
        # currently this needs to be tuned on a shot-by-shot
        # basis, as it is sensitive to the average ELM separation.
        self.rho = rho

        # half-width of the time window used for calculating
        # threshold values. Must be long enough such that any
        # window of width 2*w contains several ELMs
        self.w = width

        # estimate of maximum duration (in seconds) of
        # the postive-gradient half of an elm feature
        self.t_sep = t_sep

        # An estimate of the number of data points which
        # make up the initial sharp edge of an ELM feature.
        self.d = dnpoint

        # settings
        self.mode = mode  #
        self.m = mtime  # number of time windows used to calculate threshold curve
        self.h = hwidth  # half-width of gaussian smoothing window
        self.dt = (
            self.t[len(self.t) // 2] - self.t[(len(self.t) // 2) - 1]
        )  # time resolution

    def run(self):
        """
        Generates a list of times at which ELMs occur in the
        (t, y) data provided.
        """
        self.get_alpha()  # calculate intensity constraint
        self.get_delta()  # calculate gradient constraint

        # this can be done in a one-step way which preserves
        # nu, and doesnt need a loop - fix later
        self.nu = self.alpha * self.delta  # combine the constraints
        self.q = np.zeros(len(self.y))
        for i in range(1, len(self.y) - 1):  # set all non-peak values of nu to zero
            if (self.nu[i] > self.nu[i - 1]) & (self.nu[i] > self.nu[i + 1]):
                self.q[i] = 1

        # now we need to remove all peaks originating from the same ELM.
        # need to work from back-to-front to avoid errors
        k = (np.where(self.q == 1)[0])[::-1]  # reversed array of possible elm indices
        self.elm_inds = list()
        di = np.ceil(self.t_sep / self.dt)

        # if the separation time between the current peak and the one which
        # preceeds it is less than self.t_sep, we assume that both come from
        # the same elm, and remove the current peak.
        for i in range(len(k) - 1):
            if (k[i] - k[i + 1]) > di:
                self.elm_inds.append(k[i])
        self.elm_inds.append(k[-1])  # includes first ELM

        self.elm_inds.reverse()  # flip list back to chronological order
        self.elm_times = list()
        for i in self.elm_inds:  # use list of elm indices to get elm times
            self.elm_times.append(self.t[i - self.d])

        self.elm_times = np.array(self.elm_times)
        self.elm_sep = np.diff(
            self.elm_times
        )  # get separation time between adjacent elms

    def get_alpha(self):
        """
        Calculates self.alpha - the likelihood that a point is
        part of an ELM peak based on its intensity value.
        """
        # create a series of times across the data which serve as the centre
        # of time windows with half-width self.w
        t_c = np.linspace(np.min(self.t), np.max(self.t), self.m)
        u = np.zeros(self.m)
        for i in range(self.m):
            # find all data in the current time-window
            booles = (self.t > t_c[i] - self.w) & (self.t < t_c[i] + self.w)
            # sort the y-data in the window
            order = np.sort(self.y[np.where(booles)])
            # estimate the self.rho'th percentile for the window
            u[i] = order[np.round(self.rho * len(order)).astype(int)]

        # smoothing of threshold level data
        u = self.smooth(u)

        # set threshold at ends of the data
        dt_c = t_c[1] - t_c[0]
        ind = np.round(self.w / dt_c).astype(int)
        u[:ind] = u[ind]
        u[-ind:] = u[-ind]

        # use a spline to determine
        spline = splint(t_c, u)
        self.tau = spline(self.t)
        self.alpha = np.array(self.y >= self.tau)

    def get_delta(self):
        """
        Calculates self.delta - the likelihood that a point is at the
        top of an ELM rise based on backwards fractional derivative
        """
        self.delta = np.zeros(len(self.y))
        if self.mode == "fractional":
            self.delta[self.d :] = (self.y[self.d :] - self.y[: -self.d]) / self.y[
                : -self.d
            ]
        else:
            self.delta[self.d :] = self.y[self.d :] - self.y[: -self.d]

        self.delta[
            np.where(self.delta < 0)
        ] = 0  # sets all negative derivatives to zero

    def smooth(self, z):
        """
        basic gaussian moving average used for smoothing
        the dynamic thresholding level.
        """
        Gx = np.linspace(-2, 2, 2 * self.h + 1)
        G = np.exp(-0.5 * (Gx) ** 2)
        G /= np.sum(G)

        z_s = np.zeros(len(z))
        for i in range(self.h, len(z) - self.h):
            z_s[i] = np.sum(G * z[i - self.h : i + self.h + 1])

        z_s[: self.h] = z_s[self.h + 1]
        z_s[-self.h :] = z_s[-(self.h + 1)]
        return z_s

    def filter_signal(self, sig_t, trange=[0, 100.0], inter_elm_range=[0.7, 0.9]):
        """
        Provide a mask using a given time-base to filter between elms. 
        param: inter_elm_range is the percentage range to filter between elms
        """

        elm_times = self.elm_times[1:][
            np.logical_and(
                self.elm_times[1:] > trange[0], self.elm_times[1:] < trange[1]
            )
        ]
        elm_seps = self.elm_sep[
            np.logical_and(
                self.elm_times[1:] > trange[0], self.elm_times[1:] < trange[1]
            )
        ]

        bool_arr = np.zeros(sig_t.size, dtype=bool)

        for i, t in enumerate(elm_times):
            bool_arr = np.logical_or(
                bool_arr,
                np.logical_and(
                    sig_t > (t - (elm_seps[i] * (1.0 - inter_elm_range[0]))),
                    sig_t < (t - (elm_seps[i] * (1.0 - inter_elm_range[1]))),
                ),
            )

        return bool_arr

    """
    END OF CORE FUNCTIONS
    the following are for diagnostic purposes
    """

    def check_threshold(self):
        """
        Overplots the (t, y) data and the threshold level.
        Allows the user to adjust self.rho such that the
        threshold cuts below all ELMs but stays above any
        non-ELM features in the data.
        """
        plt.plot(self.t, self.y)
        plt.plot(self.t, self.tau, lw=2, label="threshold level")

        plt.ylabel("amplitude", fontsize=14)
        plt.xlabel("time", fontsize=14)
        plt.title("diagnostic plot - check threshold level", fontsize=14)
        plt.tick_params(axis=u"both", labelsize=12)
        plt.grid()
        plt.legend(fontsize=12)

        plt.tight_layout()
        plt.show()
