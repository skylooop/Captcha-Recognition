from tqdm import tqdm
import torch
import config

def train(model, loader, opt):
    model.train()
    final_loss = 0
    tk = tqdm(loader, total = len(loader))
    for data in tk:
        for k,v in data.items():
            data[k] = v.to(config.device)
        opt.zero_grad()
        _, loss = model(**data)
        loss.backward()
        opt.step()
        final_loss+=loss.item()
    return final_loss/len(loader)

def eval(model, loader):
    model.eval()
    final_loss = 0
    fin_preds = []
    tk = tqdm(loader, total = len(loader))
    with torch.no_grad():
        for data in tk:
            for k,v in data.items():
                data[k] = v.to(config.device)
            _, loss = model(**data)
            final_loss+=loss.item()
            fin_preds.append(_)
    return fin_preds, final_loss/len(loader)

