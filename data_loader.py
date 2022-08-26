import pyconll
import os
import torch
from functions import preprocess_2

def load_sentences():
        manx_train=os.getcwd()+'/Data/gv_cadhan-ud-train.conllu'
        manx_test=os.getcwd()+'/Data/gv_cadhan-ud-test.conllu'
        manx_dev=os.getcwd()+'/Data/gv_cadhan-ud-dev.conllu'
        irish_train=os.getcwd()+'/Data/ga_idt-ud-train.conllu'
        scottish_gaelic_train=os.getcwd()+'/Data/gd_arcosg-ud-train.conllu'
        serbian_train=os.getcwd()+'/Data/sr_set-ud-train.conllu'
        slovak_train=os.getcwd()+'/Data/sk_snk-ud-train.conllu'
        slovenian_train=os.getcwd()+'/Data/sl_ssj-ud-train.conllu'
        lithuanian_train=os.getcwd()+'/Data/lt_alksnis-ud-train.conllu'
        latvian_train=os.getcwd()+'/Data/lv_lvtb-ud-train.conllu'
        czech_train=os.getcwd()+'/Data/cs_pdt-ud-train.conllu'
        
        sentences_manx_train=preprocess_2(pyconll.load_from_file(manx_train))
        sentences_manx_test=preprocess_2(pyconll.load_from_file(manx_test))
        sentences_manx_dev=preprocess_2(pyconll.load_from_file(manx_dev))
        sentences_irish_train=preprocess_2(pyconll.load_from_file(irish_train))
        sentences_scottish_gaelic_train=preprocess_2(pyconll.load_from_file(scottish_gaelic_train))
        sentences_serbian_train=preprocess_2(pyconll.load_from_file(serbian_train))
        sentences_slovak_train=preprocess_2(pyconll.load_from_file(slovak_train))
        sentences_slovenian_train=preprocess_2(pyconll.load_from_file(slovenian_train))
        sentences_lithuanian_train=preprocess_2(pyconll.load_from_file(lithuanian_train))
        sentences_latvian_train=preprocess_2(pyconll.load_from_file(latvian_train))
        sentences_czech_train=preprocess_2(pyconll.load_from_file(czech_train))

        return sentences_manx_train,sentences_manx_test,sentences_manx_dev,sentences_irish_train,sentences_scottish_gaelic_train,sentences_serbian_train,sentences_slovak_train,sentences_slovenian_train,sentences_lithuanian_train,sentences_latvian_train,sentences_czech_train


def get_sentences(sentences_train,sentences_test,tags,max_len):
        sentences_for_train=[]
        tags_for_train=[]
        
        for sentence in sentences_train:
                k=[]
                t=[]
                for token in sentence:
                        if token.form is not None:
                                if token.form=='।':
                                        k.append('.')
                                else:
                                        k.append(token.form)
                                t.append(tags[token.upos])
#                k.append('EOS')
#                t.append(tags['X'])
#                for _ in range(len(k),max_len):
#                        k.append('EOS')
#                        t.append(tags['X'])
                sentences_for_train.append(k)
                tags_for_train.append(t)

        return sentences_for_train,tags_for_train

def get_tokens(sentences):
        s=set()
        for sentence in sentences:
                for token in sentence:
                        s.add(token.upos)
        s.add('START')
        s.add('END')
        s=list(s)
        dict2={}
        dict1={}
        for i in range(len(s)):
                dict2[s[i]]=i
                dict1[i]=s[i]
        
        return dict2,dict1,len(s)
        
def get_characters(sentences):
        s=set()
        for sentence in sentences:
                for word in sentence:
                        for character in word:
                                s.add(character)
                                
        s=list(s)
        
        dict={}
        for i in range(len(s)):
                dict[s[i]]=i

        dict['pad']=len(s)
                
        return dict,len(s)+1


class DataLoader(object):
        def __init__(self,train_sentences,test_sentences,train_tags,test_tags,max_len,model):
                self.train=train_sentences
                self.test=test_sentences
                self.train_number=0
                self.test_number=0
                self.max_len=max_len
                self.model=model
                self.train_tags=train_tags
                self.test_tags=test_tags
                
        def load_next(self):
                sentence=self.train[self.train_number]
                tags=self.train_tags[self.train_number]
                l=[]
                
                for token in sentence:
                        l.append(self.model[token])
                        
                embedding=torch.tensor(l).view(1,len(sentence),-1) #.cuda()
                self.train_number=(self.train_number+1)%len(self.train)
                tags=torch.tensor(tags) #.cuda()
                
                return embedding,tags,sentence
            
        def load_next_test(self):
                sentence=self.test[self.test_number]
                tags=self.test_tags[self.test_number]
                l=[]
                
                for token in sentence:
                        l.append(self.model[token])
                        
                embedding=torch.tensor(l).view(1,len(sentence),-1) #.cuda()
                self.test_number=(self.test_number+1)%len(self.test)
                tags=torch.tensor(tags) #.cuda()
                
                return embedding,tags,sentence
            
            
class Data_Loader(object):
        def __init__(self,dataloaders,N,K,examples=3):
                self.counter=0
                self.K=K
                self.data=[]
                self.N=N

                for _ in range(examples):
                        for dataloader in dataloaders:
                                for i in range(self.K):
                                        self.data.append(dataloader.load_next())

        def load_next(self,reuse=False):
                data=self.data[self.counter]
                if reuse:
                        self.counter=(self.counter+1)%(self.N*self.K)
                else:
                        self.counter+=1

                return data

        def set_counter(self):
                self.counter=self.N*self.K