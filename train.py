import torch
import torch.nn as nn

import gensim.models as gs
import argparse

from data_loader import DataLoader,get_tokens,get_sentences,load_sentences,get_characters
from maml import MetaLearn
import pickle as pkl

def main(args):
                
        lossFunction=nn.CrossEntropyLoss()
        
        hidden_size=args.hidden_size
        epsilon=args.epsilon
        training_mode=args.training_mode
        learning_rate=args.learning_rate
        
        epochs=args.epochs
        K=args.K_shot_learning
        N=args.N_way_learning
        inner_epoch=args.inner_gradient_update
        max_len=116
        
        manx_train,manx_test,manx_dev,irish_train,scottish_gaelic_train,serbian_train,slovak_train,slovenian_train,lithuanian_train,latvian_train,czech_train=load_sentences()
        tokens_dict,dict_token,n_tokens=get_tokens(irish_train)
        
        manx,manx_tags=get_sentences(manx_train,None,tokens_dict,max_len)
        manx_d,manx_tags_d=get_sentences(manx_dev,None,tokens_dict,max_len)
        manx_t,manx_tags_t=get_sentences(manx_test,None,tokens_dict,max_len)
        lithuanian,lithuanian_tags=get_sentences(lithuanian_train,None,tokens_dict,max_len)
        irish,irish_tags=get_sentences(irish_train,None,tokens_dict,max_len)
        serbian,serbian_tags=get_sentences(serbian_train,None,tokens_dict,max_len)
        latvian,latvian_tags=get_sentences(latvian_train,None,tokens_dict,max_len)
        # with open("sentences.pkl", "wb") as f:
        #         irish_tags = {"irish_tags":irish_tags}

        
        # a_file = open("data.pkl", "wb")

        # pickle.dump(dict_token, a_file)

        # a_file.close()
        

        a_file = open("data.pkl", "wb")

        pickle.dump(dict_token, a_file)

        a_file.close()
                
        manx=manx+manx_d+manx_t
        manx_tags=manx_tags+manx_tags_d+manx_tags_t
        
        slovak,slovak_tags=get_sentences(slovak_train,None,tokens_dict,max_len)
        czech,czech_tags=get_sentences(czech_train,None,tokens_dict,max_len)
        scottish_gaelic,scottish_gaelic_tags=get_sentences(scottish_gaelic_train,None,tokens_dict,max_len)
        slovenian,slovenian_tags=get_sentences(slovenian_train,None,tokens_dict,max_len)
        
        model_lithuanian=gs.Word2Vec(lithuanian,min_count=1,size=hidden_size)        
        model_manx=gs.Word2Vec(manx,min_count=1,size=hidden_size)   
        model_latvian=gs.Word2Vec(latvian,min_count=1,size=hidden_size) 
        model_irish=gs.Word2Vec(irish,min_count=1,size=hidden_size)   
        model_serbian=gs.Word2Vec(serbian,min_count=1,size=hidden_size)       
        model_slovenian=gs.Word2Vec(slovenian,min_count=1,size=hidden_size)  
        model_slovak=gs.Word2Vec(slovak,min_count=1,size=hidden_size)
        model_czech=gs.Word2Vec(czech,min_count=1,size=hidden_size)  
        model_scottish_gaelic=gs.Word2Vec(scottish_gaelic,min_count=1,size=hidden_size)  
        
        char_dict,n_chars=get_characters(manx+lithuanian+irish+latvian+serbian+slovak+czech+scottish_gaelic+slovenian)
        
        lithuanian_data_loader=DataLoader(lithuanian,None,lithuanian_tags,None,max_len,model_lithuanian)
        manx_data_loader=DataLoader(manx,None,manx_tags,None,max_len,model_manx)
        latvian_data_loader=DataLoader(latvian,None,latvian_tags,None,max_len,model_latvian)
        irish_data_loader=DataLoader(irish,None,irish_tags,None,max_len,model_irish)
        serbian_data_loader=DataLoader(serbian,None,serbian_tags,None,max_len,model_serbian)
        slovak_data_loader=DataLoader(slovak,None,slovak_tags,None,max_len,model_slovak)
        slovenian_data_loader=DataLoader(slovenian,None,slovenian_tags,None,max_len,model_slovenian)
        scottish_gaelic_data_loader=DataLoader(scottish_gaelic,None,scottish_gaelic_tags,None,max_len,model_scottish_gaelic)
        czech_data_loader=DataLoader(czech,None,czech_tags,None,max_len,model_czech)
        
        metaLearn=MetaLearn(lithuanian_data_loader,manx_data_loader,latvian_data_loader,irish_data_loader,serbian_data_loader,czech_data_loader,slovak_data_loader,slovenian_data_loader,scottish_gaelic_data_loader,lossFunction,hidden_size,
                            epochs,inner_epoch,max_len,n_tokens,tokens_dict,dict_token,char_dict,n_chars,N,K,learning_rate)
        
        if args.resume_training:
                model=torch.load(args.checkpoint_path)
                metaLearn.epochs=model['epoch']
                metaLearn.load_state_dict(model['model'])
                
                if args.resume_training_type=='MAML':
                        metaLearn.train()
                        _=metaLearn.test()
                elif args.resume_training_type=='Reptile':
                        metaLearn.train_Reptile()
                        _=metaLearn.test()
                        
        elif args.load_model:
                metaLearn.load_state_dict(torch.load(args.model_path))
                _=metaLearn.test()           
        
        if training_mode=='MAML':
                metaLearn.train()
                _=metaLearn.test()
        elif training_mode=='Reptile':
                metaLearn.train_Reptile(epsilon)
                _=metaLearn.test()
        else:
                raise(NotImplementedError('This algorithm has not been implemented'))
                
def setup():
        parser=argparse.ArgumentParser('Metalearning argument parser')
        
        parser.add_argument('--learning_rate',type=float,default=0.01)
        parser.add_argument('--hidden_size',type=int,default=256)
        parser.add_argument('--N_way_learning',type=int,default=2)
        parser.add_argument('--training_mode',type=str,default='MAML')
        parser.add_argument('--epsilon',type=float,default=0.1)
        parser.add_argument('--epochs',type=int,default=150)
        parser.add_argument('--load_model',type=bool,default=False)
        parser.add_argument('--model_path',type=str)
        parser.add_argument('--checkpoint_path',type=str)
        parser.add_argument('--resume_training_type',type=str,default='MAML')
        parser.add_argument('--resume_training',type=bool,default=False)
        parser.add_argument('--K_shot_learning',type=int,default=5)
        parser.add_argument('--inner_gradient_update',type=int,default=5)
        args=parser.parse_args()
        
        return args
    
if __name__=='__main__':
        args=setup()
        main(args)