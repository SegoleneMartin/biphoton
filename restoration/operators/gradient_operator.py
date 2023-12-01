from scipy.sparse.linalg import LinearOperator
import numpy as np


def generate_G(shape, res):

    N = shape[0] * shape[1] * shape[2]
    (n1, n2, n3) = shape

    def G(x):
        x = x.reshape(shape)
        Ghx = np.copy(x)
        Ghx[1:, :, :] = x[1:, :, :] - x[:n1-1, :, :]
        Gvx = np.copy(x)
        Gvx[:, 1:, :] = x[:, 1:, :] - x[:, :n2-1, :]
        Gtx = np.copy(x)
        Gtx[:, :, 1:] = x[:, :, 1:] - x[:, :, :n3-1]

        Gx = np.vstack([(Ghx/res[0]).reshape(N), (Gvx/res[1]).reshape(N), (Gtx/res[2]).reshape(N)]).T
        return(Gx.reshape(3*N))

    def G_T(x):
        x = x.reshape(N, 3)
        G_Thx = x[:, 0].reshape(shape)
        G_Thx[:n1-1, :, :] = G_Thx[:n1-1, :, :] - G_Thx[1:, :, :] 
        G_Tvx = x[:, 1].reshape(shape)
        G_Tvx[:, :n2-1, :] = G_Tvx[:, :n2-1, :] - G_Tvx[:, 1:, :] 
        G_Ttx = x[:, 2].reshape(shape)
        G_Ttx[:, :, :n3-1] =  G_Ttx[:, :, :n3-1] - G_Ttx[:, :, 1:]
        
        G_Tx = G_Thx/res[0] + G_Tvx/res[1] + G_Ttx/res[2]
        return(G_Tx.reshape(N))

    G = LinearOperator((3*N, N), matvec = G, rmatvec = G_T)
    return(G)
'''

def generate_G(shape, res):

    N = shape[0] * shape[1] 
    (n1, n2) = shape

    def G(x):
        x = x.reshape(shape)
        Ghx = np.copy(x)
        Ghx[1:, :, :] = x[1:, :, :] - x[:n1-1, :, :]
        Gvx = np.copy(x)
        Gvx[:, 1:, :] = x[:, 1:, :] - x[:, :n2-1, :]
        
        Gx = np.vstack([(Ghx/res[0]).reshape(N), (Gvx/res[1]).reshape(N)]).T
        return(Gx.reshape(2*N))

    def G_T(x):
        x = x.reshape(N, 3)
        G_Thx = x[:, 0].reshape(shape)
        G_Thx[:n1-1, :, :] = G_Thx[:n1-1, :, :] - G_Thx[1:, :, :] 
        G_Tvx = x[:, 1].reshape(shape)
        G_Tvx[:, :n2-1, :] = G_Tvx[:, :n2-1, :] - G_Tvx[:, 1:, :] 
        
        G_Tx = G_Thx/res[0] + G_Tvx/res[1]
        return(G_Tx.reshape(N))

    G = LinearOperator((2*N, N), matvec = G, rmatvec = G_T)
    return(G)
    
'''