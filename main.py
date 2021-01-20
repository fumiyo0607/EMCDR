import create_data
import create_latent_factor
import mapping_latent_factor
import eval
import parameter_setting
import util

if __name__ == "__main__":
    ## create data

    ## parameter setting...
    setting_latent_factor = parameter_setting.LatentFacorModel(latent_factor_model='doc2vec', facter_size=50, train_num=300)
    setting_mlp = parameter_setting.MLP(hidden_layer_size=100, activation_func=None, train_num=500)
    setting = parameter_setting.Mapping(setting_latent_factor, setting_mlp)
    
    '''****************************
    Step 1. Training Latent Factors
    '''
    ## Doc2vec
    create_latent_factor.doc2vec_source(vector_size=setting_latent_factor.facter_size, epochs=setting_latent_factor.train_num)
    create_latent_factor.doc2vec_target(vector_size=setting_latent_factor.facter_size, epochs=setting_latent_factor.train_num)

    ## lda
    create_latent_factor.lda_source(topic_num=setting_latent_factor.facter_size, iters=setting_latent_factor.train_num)
    create_latent_factor.lda_target(topic_num=setting_latent_factor.facter_size, iters=setting_latent_factor.train_num)

    '''****************************
    Step 2. Mapping Latent Factors
    '''
    loss, Xt = mapping_latent_factor.mapping(setting=setting, learning_rate=0.1)

    '''****************************
    Step 3. Evaluation
    '''
    eval.doc2vec(setting=setting)
