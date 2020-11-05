#https://arxiv.org/pdf/1703.07737.pdf
import torch
from collections import defaultdict

def hardTriplet(embeddings,classes,margin=0.1):

    c_embeddings=defaultdict(list)
    for i,cls in enumerate(classes):
        c_embeddings[cls].append(embeddings[i])

    for cls in c_embeddings:
        c_embeddings[cls] = torch.stack(c_embeddings[cls],dim=0)


    loss=[]
    for i,(cls,embeds) in enumerate(c_embeddings.items()):
        for a in range(embeds.size(0)):
            
            #el_loss=D(emb[None,...],torch.cat((embeds[:a],embeds[a+1:]),dim=0)).max()
            true_dist=D(embeds[a:a+1],embeds).max() #this is equivilent to the above, as the matching index will always dist=0

            negs=[]
            for j,(clsN,embedsN) in enumerate(c_embeddings.items()):
                if j==i:
                    continue
                negs.append( D(embeds[a:a+1],embedsN).min() )
            false_dist = torch.stack(negs).min()
            
            loss.append(max(margin+true_dist-false_dist,torch.tensor(0.0).to(embedsN.device)))

    return torch.stack(loss).mean()
        

def D(a,b):
    return (a-b).pow(2).sum(-1)
