# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 18:42:48 2016

@author: charlie
"""
np.sum(x*r[:, np.newaxis], axis=0)
np.sum(z*r[:, np.newaxis], axis=0)
np.sum(xz*r[:, np.newaxis, np.newaxis], axis=0)
np.sum(zz*r[:, np.newaxis, np.newaxis], axis=0)
s1 = np.trace(xx, axis1=1, axis2=2)
s2 = -2*np.trace(xz.dot(W.T), axis1=1, axis2=2)
s3 = np.trace(zz*W.T.dot(W), axis1=1, axis2=2)
sigma_sq = np.sum(r*(s1 + s2 + s3))
