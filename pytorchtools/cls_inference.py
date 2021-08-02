'''
Inference Classifier
'''
# edited by Alessandro Nicolosi - https://github.com/alenic


import torch
import numpy as np
from .surgery import ForwardMonitor

def cls_inference(model, data_loader, image_index=0, get_score=False, device="cuda"):
    model.eval()
    model.to(device)

    logits_list = []
    if get_score:
        score_list = []
    with torch.no_grad():
        for data in data_loader:
            image_batch = data[image_index].to(device)
            logits = model(image_batch)
            logits_list.append(logits.cpu().numpy().squeeze())
            if get_score:
                score_list.append(torch.softmax(logits, 1).cpu().numpy().squeeze())

    if get_score:
        return np.vstack(logits_list), np.vstack(score_list)
    
    return np.vstack(logits_list)


def cls_inference_ensambles(model_list, data_loader, image_index=0, get_score=False, device="cuda"):
    assert isinstance(model_list, list)

    n_models = len(model_list)
    logits_list = []
    if get_score:
        score_list = []
    for i in range(n_models):
        if get_score:
            logits_np, score_np = cls_inference(model_list[i], data_loader, image_index=image_index, get_score=True, device=device)
            score_list.append(score_np)
        else:
            logits_np = cls_inference(model_list[i], data_loader, image_index=image_index, get_score=False, device=device)
        
        logits_list.append(logits_np)
    
    if get_score:
        return logits_list, score_list
    
    return logits_list


def cls_inference_embedding(
    model,
    data_loader,
    embedding_layer_name,
    image_index=0,
    get_logits=False,
    get_score=False,
    device="cuda",
    verbose=False,
):
    """
    return order: emb, logits, score
    """
    model.eval()
    model.to(device)

    monitor = ForwardMonitor(model, verbose=verbose)
    monitor.add_layer(embedding_layer_name)

    embedding_list = []
    if get_logits:
        logits_list = []
    if get_score:
        score_list = []
    with torch.no_grad():
        for data in data_loader:
            image_batch = data[image_index].to(device)
            logits = model(image_batch)

            embedding_list.append(
                monitor.get_layer(embedding_layer_name).cpu().numpy().squeeze()
            )

            if get_logits:
                logits_list.append(logits.cpu().numpy().squeeze())
            if get_score:
                score_list.append(torch.softmax(logits, 1).cpu().numpy().squeeze())

    return_values = []

    return_values.append(np.vstack(embedding_list))

    if get_logits:
        return_values.append(np.vstack(logits_list))

    if get_score:
        return_values.append(np.vstack(score_list))

    if len(return_values) == 1:
        return return_values[0]

    return tuple(return_values)