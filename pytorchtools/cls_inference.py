"""
Inference Classifier
"""

# edited by Alessandro Nicolosi - https://github.com/alenic


import torch
import numpy as np
import tqdm

from pytorchtools.surgery import ForwardMonitor


def cls_inference(
    model, data_loader, image_index=0, get_score=False, device="cuda", use_amp=False
):
    model.eval()
    model.to(device)

    logits_list = []
    if get_score:
        score_list = []
    with torch.no_grad():
        for data in data_loader:
            image_batch = data[image_index].to(device)
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(image_batch)
            else:
                logits = model(image_batch)
            logits_list.append(logits.cpu().numpy().squeeze())
            if get_score:
                score_list.append(torch.softmax(logits, 1).cpu().numpy().squeeze())

    if get_score:
        return np.vstack(logits_list), np.vstack(score_list)

    return np.vstack(logits_list)


def cls_inference_ensambles(
    model_list,
    data_loader,
    image_index=0,
    get_score=False,
    device="cuda",
    use_amp=False,
):
    assert isinstance(model_list, list)

    n_models = len(model_list)
    logits_list = []
    if get_score:
        score_list = []
    for i in range(n_models):
        if get_score:
            logits_np, score_np = cls_inference(
                model_list[i],
                data_loader,
                image_index=image_index,
                get_score=True,
                device=device,
            )
            score_list.append(score_np)
        else:
            logits_np = cls_inference(
                model_list[i],
                data_loader,
                image_index=image_index,
                get_score=False,
                device=device,
                use_amp=use_amp,
            )

        logits_list.append(logits_np)

    if get_score:
        return logits_list, score_list

    return logits_list


def cls_inference_embedding(
    model,
    data_loader,
    embedding_layer_name=None,
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

    if embedding_layer_name is not None:
        monitor = ForwardMonitor(model, verbose=verbose)
        monitor.add_layer(embedding_layer_name)

    embedding_list = []
    if get_logits:
        logits_list = []
    if get_score:
        score_list = []
    with torch.no_grad():
        for _, data in enumerate(tqdm.tqdm(data_loader)):
            image_batch = data[image_index].to(device)
            out = model(image_batch)

            if embedding_layer_name is not None:
                emb = (
                    monitor.get_layer(embedding_layer_name)
                    .squeeze(-1)
                    .squeeze(-1)
                    .cpu()
                    .numpy()
                )

            else:
                emb = out.squeeze(-1).squeeze(-1).cpu().numpy()

            embedding_list.append(emb)
            if get_logits:
                logits_list.append(out.squeeze(-1).squeeze(-1).cpu().numpy())
            if get_score:
                score_list.append(
                    torch.softmax(out, 1).squeeze(-1).squeeze(-1).cpu().numpy()
                )

    return_values = []

    return_values.append(np.vstack(embedding_list))

    if get_logits:
        return_values.append(np.vstack(logits_list))

    if get_score:
        return_values.append(np.vstack(score_list))

    if len(return_values) == 1:
        return return_values[0]

    return tuple(return_values)
