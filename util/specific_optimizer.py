def scaffold_tuning_gradients(model, server_controls, client_controls):
    for name, param in model.named_parameters():
        if param.requires_grad==True:
            param.grad.data += (server_controls[name] - client_controls[name])