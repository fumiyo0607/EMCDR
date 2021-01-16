class LatentFacorModel():
    def __init__(self, latent_factor_model, facter_size, train_num):
        self.model = latent_factor_model
        self.facter_size = facter_size
        self.train_num = train_num
    
    def info(self):
        print('Latent Facor Model : ', self.model)
        print('Factor Size : ', self.facter_size)
        print('Train Num : ', self.train_num)


class MLP():
    def __init__(self, hidden_layer_size, activation_func, train_num):
        self.hidden_layer_size = hidden_layer_size
        self.activation_func = activation_func
        self.train_num = train_num
    
    def info(self):
        print('Hidden Layer Size : ', self.hidden_layer_size)
        print('Activate Function : ', self.activation_func)
        print('Epochs : ', self.train_num)

class Mapping():
    def __init__(self, latent_facor_model: LatentFacorModel, mlp: MLP):
        self.latent_facor_model = latent_facor_model
        self.mlp = mlp

    def info(self):
        print('LatentFacorModel')
        self.latent_facor_model.info()
        print('MLP')
        self.mlp.info()