###############################################################################
############################### LIBRARY IMPORTS ###############################
###############################################################################

import numpy as np

###############################################################################
################################## CLASSES ####################################
###############################################################################

class ArctanConvoluter():

    def __init__(self, e, e_edge, e_l, e_c, e_f, g_hole, g_max, p_dim = 150):

        self.e = e
        self.e_edge = e_edge
        self.e_l = e_l
        self.e_c = e_c
        self.e_f = e_f
        self.g_hole = g_hole
        self.g_max = g_max
        self.p_dim = p_dim

        e_pad_min = np.min(self.e) - (np.diff(self.e)[0] * self.p_dim)
        e_pad_max = np.max(self.e) + (np.diff(self.e)[0] * self.p_dim)
        self.e_pad = np.pad(self.e, self.p_dim, mode = 'linear_ramp',
                        end_values = (e_pad_min, e_pad_max))

        self.g = self.__arctan_g()
        self.h = self.__lorentz_h()

    def convolute(self, mu):

        mu_pad = np.pad(mu, self.p_dim, mode = 'edge')

        mu_conv = np.sum(self.h.T * np.where(self.e_pad < self.e_edge, 
                                             0.0, mu_pad), axis = 1)

        mu_conv = (np.sum(np.where(self.e_pad < self.e_edge, 0.0, mu_pad)) 
                   * mu_conv) / np.sum(mu_conv)

        mu_conv = mu_conv[self.p_dim:-self.p_dim]

        return mu_conv
                      
    def __arctan_g(self):

        e_rel = self.e_pad - self.e_edge
        gq = (e_rel - self.e_f) / self.e_c
        
        g = self.g_hole + (
            self.g_max * (
                0.5 + (
                    (1.0 / np.pi) * (
                        np.arctan((np.pi / 3.0) * (self.g_max / self.e_l) 
                                  * (gq - (1.0 / np.square(gq)))
                        )
                    )
                )
            )
        )

        return g

    def __lorentz_h(self):

        de = self.e_pad[:,np.newaxis] - self.e_pad[np.newaxis,:]

        h = (0.5 * self.g) / (np.square(de) + np.square(0.5 * self.g))
        h = np.where(de != 0.0, h, 1.0)

        return h

    @property
    def e(self):
        
        return self.__e

    @e.setter
    def e(self, e):

        if not isinstance(e, np.ndarray):
            str_ = 'e has to be a Numpy array; got {}'
            raise ValueError(str_.format(e))
        else:
            self.__e = e

    @property
    def e_edge(self):
        
        return self.__e_edge

    @e_edge.setter
    def e_edge(self, e_edge):

        if not isinstance(e_edge, float) or e_edge <= 0.0:
            str_ = 'e_edge has to be a) a float, and b) > 0; got {}'
            raise ValueError(str_.format(e_edge))
        else:
            self.__e_edge = e_edge

    @property
    def e_l(self):
        
        return self.__e_l

    @e_l.setter
    def e_l(self, e_l):

        if not isinstance(e_l, float):
            str_ = 'e_l has to be a float; got {}'
            raise ValueError(str_.format(e_l))
        else:
            self.__e_l = e_l

    @property
    def e_c(self):
        
        return self.__e_c

    @e_c.setter
    def e_c(self, e_c):

        if not isinstance(e_c, float):
            str_ = 'e_c has to be a float; got {}'
            raise ValueError(str_.format(e_c))
        else:
            self.__e_c = e_c

    @property
    def e_f(self):
        
        return self.__e_f

    @e_f.setter
    def e_f(self, e_f):

        if not isinstance(e_f, float):
            str_ = 'e_f has to be a float; got {}'
            raise ValueError(str_.format(e_f))
        else:
            self.__e_f = e_f

    @property
    def g_hole(self):
        
        return self.__g_hole

    @g_hole.setter
    def g_hole(self, g_hole):

        if not isinstance(g_hole, float):
            str_ = 'g_hole has to be a float; got {}'
            raise ValueError(str_.format(g_hole))
        else:
            self.__g_hole = g_hole

    @property
    def g_max(self):
        
        return self.__g_max

    @g_max.setter
    def g_max(self, g_max):

        if not isinstance(g_max, float):
            str_ = 'g_max has to be a float; got {}'
            raise ValueError(str_.format(g_max))
        else:
            self.__g_max = g_max

    @property
    def p_dim(self):
        
        return self.__p_dim

    @p_dim.setter
    def p_dim(self, p_dim):

        if not isinstance(p_dim, int) or p_dim < 0:
            str_ = 'p_dim has to be a) an integer, and b) >= 0; got {}'
            raise ValueError(str_.format(p_dim))
        else:
            self.__p_dim = p_dim