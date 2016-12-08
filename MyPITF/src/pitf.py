import numpy as np
import gmpy2
from gmpy2 import mpfr

class PITF:
    def __init__(self,alpha=0.0001,lamb=0.1,k=30,max_iter=100,data_shape=None,verbose=0):
        self.alpha = alpha
        self.lamb = lamb
        self.k = k
        self.max_iter = max_iter
        self.data_shape = data_shape
        self.verbose = verbose
        self.topN = 20

    def _init_latent_vectors_inc(self,data_shape, base_latent_vector): #put trained smaller latent vectors into current bigger vector
        latent_vector = {}
        latent_vector['u'] = np.random.normal(loc=0, scale=0.1, size=(data_shape[0],self.k))
        latent_vector['i'] = np.random.normal(loc=0, scale=0.1, size=(data_shape[1],self.k))
        latent_vector['tu'] = np.random.normal(loc=0, scale=0.1, size=(data_shape[2],self.k))
        latent_vector['ti'] = np.random.normal(loc=0, scale=0.1, size=(data_shape[2],self.k))

#         latent_vector['u'] = np.ones((data_shape[0],self.k))
#         latent_vector['i'] = np.ones((data_shape[1],self.k))
#         latent_vector['tu'] = np.ones((data_shape[2],self.k))
#         latent_vector['ti'] = np.ones((data_shape[2],self.k))

        latent_vector['u'][:len(base_latent_vector['u'])] = base_latent_vector['u']
        latent_vector['i'][:len(base_latent_vector['i'])] = base_latent_vector['i']
        latent_vector['tu'][:len(base_latent_vector['tu'])] = base_latent_vector['tu']
        latent_vector['ti'][:len(base_latent_vector['ti'])] = base_latent_vector['ti']
        
       
        return latent_vector

    def _init_latent_vectors(self,data_shape):
        latent_vector = {}
        latent_vector['u'] = np.random.normal(loc=0, scale=0.1, size=(data_shape[0],self.k))
        latent_vector['i'] = np.random.normal(loc=0, scale=0.1, size=(data_shape[1],self.k))
        latent_vector['tu'] = np.random.normal(loc=0, scale=0.1, size=(data_shape[2],self.k))
        latent_vector['ti'] = np.random.normal(loc=0, scale=0.1, size=(data_shape[2],self.k))
        return latent_vector

    def _calc_number_of_dimensions(self,data,validation):
        u_max = -1
        i_max = -1
        t_max = -1
        for u,i,t in data:
            if u > u_max: u_max = u
            if i > i_max: i_max = i
            if t > t_max: t_max = t
        if not validation is None:
            for u,i,t in validation:
                if u > u_max: u_max = u
                if i > i_max: i_max = i
                if t > t_max: t_max = t
        return (u_max+1,i_max+1,t_max+1)

    def _draw_negative_sample(self,t):
        r = np.random.randint(self.data_shape[2]) # sample random index
        while r==t:
            r = np.random.randint(self.data_shape[2]) # repeat while the same index is sampled
        return r

    def _sigmoid(self,x):
        mpfrExp = gmpy2.exp(-x)
        
        return float(1.0 / (1.0 + mpfrExp))

    def _score(self,data):
        if data is None: return "No validation data"
        correct = 0.
        for u,i,answer_t in data:
            predicted = self.predict_TopN(u,i)
            if predicted == answer_t: correct += 1
        return correct / data.shape[0]

    def fit_inc(self,data, base_latent_vector, validation=None):
        if self.data_shape is None: self.data_shape = self._calc_number_of_dimensions(data,validation)
        self.latent_vector_ = self._init_latent_vectors_inc(self.data_shape, base_latent_vector)
        remained_iter = self.max_iter
        while True:
            remained_iter -= 1
            np.random.shuffle(data)
            for u,i,t in data:
                if u < len(base_latent_vector['u']):
                    continue
                nt = self._draw_negative_sample(t)
                y_diff = self.y(u,i,t) - self.y(u,i,nt)
                delta = 1-self._sigmoid(y_diff)
                self.latent_vector_['u'][u] += self.alpha * (delta * (self.latent_vector_['tu'][t] - self.latent_vector_['tu'][nt]) - self.lamb * self.latent_vector_['u'][u])
                self.latent_vector_['tu'][t] += self.alpha * (delta * self.latent_vector_['u'][u] - self.lamb * self.latent_vector_['tu'][t])
                self.latent_vector_['tu'][nt] += self.alpha * (delta * -self.latent_vector_['u'][u] - self.lamb * self.latent_vector_['tu'][nt])
                #'''
                self.latent_vector_['i'][i] += self.alpha * (delta * (self.latent_vector_['ti'][t] - self.latent_vector_['ti'][nt]) - self.lamb * self.latent_vector_['i'][i])
                self.latent_vector_['ti'][t] += self.alpha * (delta * self.latent_vector_['i'][i] - self.lamb * self.latent_vector_['ti'][t])
                self.latent_vector_['ti'][nt] += self.alpha * (delta * -self.latent_vector_['i'][i] - self.lamb * self.latent_vector_['ti'][nt])
                #'''
            if self.verbose==1: print "%s\t%s" % (self.max_iter-remained_iter, self._score(validation))
            if remained_iter <= 0:
                break
        return self

    def fit(self,data,validation=None):
        if self.data_shape is None: self.data_shape = self._calc_number_of_dimensions(data,validation)
        self.latent_vector_ = self._init_latent_vectors(self.data_shape)
        remained_iter = self.max_iter
        while True:
            remained_iter -= 1
            np.random.shuffle(data)
            for u,i,t in data:
                nt = self._draw_negative_sample(t)
                y_diff = self.y(u,i,t) - self.y(u,i,nt)
                delta = 1-self._sigmoid(y_diff)
                self.latent_vector_['u'][u] += self.alpha * (delta * (self.latent_vector_['tu'][t] - self.latent_vector_['tu'][nt]) - self.lamb * self.latent_vector_['u'][u])
                self.latent_vector_['i'][i] += self.alpha * (delta * (self.latent_vector_['ti'][t] - self.latent_vector_['ti'][nt]) - self.lamb * self.latent_vector_['i'][i])
                self.latent_vector_['tu'][t] += self.alpha * (delta * self.latent_vector_['u'][u] - self.lamb * self.latent_vector_['tu'][t])
                self.latent_vector_['tu'][nt] += self.alpha * (delta * -self.latent_vector_['u'][u] - self.lamb * self.latent_vector_['tu'][nt])
                self.latent_vector_['ti'][t] += self.alpha * (delta * self.latent_vector_['i'][i] - self.lamb * self.latent_vector_['ti'][t])
                self.latent_vector_['ti'][nt] += self.alpha * (delta * -self.latent_vector_['i'][i] - self.lamb * self.latent_vector_['ti'][nt])
            if self.verbose==1: print "%s\t%s" % (self.max_iter-remained_iter, self._score(validation))
            if remained_iter <= 0:
                break
        return self

    def y(self,i,j,k):
        return self.latent_vector_['tu'][k].dot(self.latent_vector_['u'][i]) + self.latent_vector_['ti'][k].dot(self.latent_vector_['i'][j])

    def predict(self,i,j):
        y = self.latent_vector_['tu'].dot(self.latent_vector_['u'][i]) + self.latent_vector_['ti'].dot(self.latent_vector_['i'][j])
        return y.argmax()

    def predict_TopN(self,i,j):
        y = self.latent_vector_['tu'].dot(self.latent_vector_['u'][i]) + self.latent_vector_['ti'].dot(self.latent_vector_['i'][j])
        #return y.argmax()
        return y.argsort(axis=0)[:, self.topN:]
     
    def predict2(self,x):
        y = self.latent_vector_['u'][x[:,0]].dot(self.latent_vector_['tu'].T) + self.latent_vector_['i'][x[:,1]].dot(self.latent_vector_['ti'].T)
        return x[:,0], x[:,1], y.argmax(axis=1)
        #return y.argmax(axis=1)
        
    def predict2_topN(self,x, n):
        y = self.latent_vector_['u'][x[:,0]].dot(self.latent_vector_['tu'].T) + self.latent_vector_['i'][x[:,1]].dot(self.latent_vector_['ti'].T)
        return y.argsort(axis=1)[:, -n:]
        
        #return y.argmax(axis=1)
